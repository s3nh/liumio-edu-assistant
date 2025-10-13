# Classic PyTorch finetuning (no TRL), QLoRA with a manual training loop
# - 4-bit quantization using bitsandbytes
# - LoRA adapters via PEFT
# - Optional sequence packing for efficiency
# - Periodic on-the-fly generation for quick quality checks
#
# How to run:
#   pip install -U datasets transformers peft bitsandbytes accelerate
#   python assistant/train.py
#
# Notes:
# - This script assumes a single-GPU setup. device_map="auto" works for a single GPU as well.
# - If you want multi-GPU or distributed training, prefer Hugging Face Accelerate or DDP.
# - Edit the CONFIG section below to adjust hyperparameters and test prompts.

import math
import os
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# --------------- CONFIG ---------------

MODEL_NAME = "Qwen/Qwen3-8B"
DATASET = "ajibawa-2023/Education-Young-Children"
OUTPUT_DIR = "out/qlora-edu-children"

SEED = 42
MAX_SEQ_LEN = 1024

PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0

WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"  # "cosine" or "linear"

LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3

PACKING = True  # Concatenate samples and chunk into MAX_SEQ_LEN sequences
BF16 = True     # Use bfloat16 autocast if supported, else float16

# Generation preview during training
GEN_EVERY_STEPS = 500  # set to 0 to disable
GEN_MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9

# Prompts to quickly sanity-check training progress
CHECK_PROMPTS = [
    "How can teachers encourage young children to share toys with classmates?",
    "What activities help develop fine motor skills in early childhood?",
    "Explain positive reinforcement for young learners in simple terms.",
]

# LoRA configuration
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --------------- UTILITIES ---------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(model: torch.nn.Module):
    trainable, total = 0, 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def rotate_checkpoints(output_dir: str, save_total_limit: int):
    if save_total_limit is None or save_total_limit <= 0:
        return
    if not os.path.exists(output_dir):
        return
    # Checkpoints named step-XXXX
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("step-")]
    if len(checkpoints) <= save_total_limit:
        return
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    to_delete = checkpoints[:-save_total_limit]
    for ckpt in to_delete:
        path = os.path.join(output_dir, ckpt)
        try:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Warning: failed to delete old checkpoint {path}: {e}")


def build_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Map dataset fields into a single 'text' field for Causal LM training.
    Explicitly supports ('prompt', 'text') pairs. Falls back to ('question','answer') or raw 'text'.
    """
    if "prompt" in example and "text" in example and example["prompt"] is not None and example["text"] is not None:
        return {"text": f"<s>[INST] {example['prompt']} [/INST] {example['text']}</s>"}
    if "question" in example and "answer" in example and example["question"] is not None and example["answer"] is not None:
        return {"text": f"<s>[INST] {example['question']} [/INST] {example['answer']}</s>"}
    if "text" in example and example["text"] is not None:
        return {"text": str(example["text"])}
    return {"text": ""}


class PackedCausalLMDataset(Dataset):
    """
    Concatenate all tokenized samples and split into fixed-length chunks.
    This approximates packing behavior for better compute efficiency.
    """
    def __init__(self, tokenizer: AutoTokenizer, texts: List[str], max_seq_len: int):
        eos_id = tokenizer.eos_token_id
        assert eos_id is not None, "Tokenizer must have an eos_token_id."

        # Tokenize and concatenate
        all_ids: List[int] = []
        for t in texts:
            ids = tokenizer(
                t,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )["input_ids"]
            # Ensure separation between samples
            if len(ids) == 0 or ids[-1] != eos_id:
                ids = ids + [eos_id]
            all_ids.extend(ids)

        # Chunk into fixed-length sequences
        total_len = (len(all_ids) // max_seq_len) * max_seq_len
        all_ids = all_ids[:total_len]
        self.input_ids = torch.tensor(
            [all_ids[i:i + max_seq_len] for i in range(0, total_len, max_seq_len)],
            dtype=torch.long
        )
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.long)
        # For Causal LM, labels are input_ids; no need to shift here (handled by model)
        self.labels = self.input_ids.clone()

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@dataclass
class DynamicCollator:
    tokenizer: AutoTokenizer
    max_seq_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids.clone()
        # Mask out padding tokens in labels
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@torch.no_grad()
def preview_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
):
    model.eval()
    print("\n===== Preview generation =====")
    for i, p in enumerate(prompts):
        print(f"\n[Prompt {i+1}] {p}")
        inputs = tokenizer(p, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"[Output {i+1}] {out}")
    print("===== End preview =====\n")
    model.train()


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Device: {device} | autocast dtype: {amp_dtype}")

    # Load dataset
    ds = load_dataset(DATASET)
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    ds_train = ds[train_split].map(build_example, remove_columns=ds[train_split].column_names)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Quantization (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # works for single-GPU as well
    )
    model.config.use_cache = False  # Important for training
    model.config.pad_token_id = tok.pad_token_id

    # Prepare for k-bit finetuning + LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_trainable_parameters(model)

    # Gradient checkpointing (optional, usually helps memory)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Data
    if PACKING:
        texts = ds_train["text"]
        train_dataset = PackedCausalLMDataset(tok, texts, MAX_SEQ_LEN)
        train_loader = DataLoader(
            train_dataset,
            batch_size=PER_DEVICE_BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
        )
    else:
        collator = DynamicCollator(tok, MAX_SEQ_LEN)
        train_loader = DataLoader(
            ds_train,
            batch_size=PER_DEVICE_BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True,
        )

    # Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_training_steps = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = max(1, int(WARMUP_RATIO * total_training_steps))
    if LR_SCHEDULER_TYPE.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    last_log_step = 0
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, batch in enumerate(train_loader, start=1):
            # Ensure tensors on correct device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            with autocast_ctx(dtype=amp_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            loss.backward()
            running_loss += loss.item()

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                if LOGGING_STEPS and global_step % LOGGING_STEPS == 0:
                    avg_loss = (running_loss / (global_step - last_log_step)) if (global_step - last_log_step) > 0 else running_loss
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    elapsed = time.time() - start_time
                    print(f"[step {global_step}] loss={avg_loss:.4f} ppl={ppl:.2f} lr={scheduler.get_last_lr()[0]:.6f} elapsed={elapsed:.1f}s")
                    running_loss = 0.0
                    last_log_step = global_step

                # Preview generation
                if GEN_EVERY_STEPS and global_step % GEN_EVERY_STEPS == 0:
                    preview_generation(model, tok, CHECK_PROMPTS, device)

                # Save checkpoint
                if SAVE_STEPS and global_step % SAVE_STEPS == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"step-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)  # saves LoRA adapters
                    tok.save_pretrained(ckpt_dir)
                    rotate_checkpoints(OUTPUT_DIR, SAVE_TOTAL_LIMIT)

        # End of epoch preview
        preview_generation(model, tok, CHECK_PROMPTS, device)

    # Final save
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Adapters saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
