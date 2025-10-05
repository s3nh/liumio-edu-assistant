# Example of inference using mock model
from typing import List, Dict, Union
import io
import asyncio
import aiofiles

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from PIL import Image

from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
MODEL_NAME: str = "opendatalab/MineU2.5-2509-1.2B"
async_llm = AsyncLLM.from_engine_args(
    AsyncEngineArgs(    
        model = MODEL_NAME, 
        logits_processors = [MinerULogitsProcessor],

    )
)

client = MinerUClient(
    backend="vllm-async-engine", 
    vllm_async_llm=async_llm,
)

async def main():
    image_path = "tests/data/sample_image.png"
    async with aiofiles.open(image_path, 'rb') as f:
        image_data = await f.read()

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    extracted_blocks = await client.aio_two_step_extract(image)
    print(extracted_blocks)


asyncio.run(main())
async_llm.shutdown()

