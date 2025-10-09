 train_qlora.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen3-8B"  # change if needed
DATASET = "ajibawa-2023/Education-Young-Children"
OUTPUT_DIR = "out/qlora-edu-children"

ds = load_dataset(DATASET)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Simple mapping (adjust to actual fields)
def build(example):
    if "prompt" in example and "text" in example:
        return {"prompt": f"<s>[INST] {example['prompt']} [/INST]", "completion": f"{example['text']}</s>"}
    elif "text" in example:
        return {"text": example["text"]}
    return {"text": ""}

train_ds = ds["train"].map(build)

from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],  # adapt to model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#model = get_peft_model(model, lora_config)

cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    args=cfg,
    #packing=True  # Packs multiple short samples into one sequence for efficiency
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)