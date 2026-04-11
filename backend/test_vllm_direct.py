"""Test Qwen3-ASR forward pass step by step, outside the vLLM engine."""
import os
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import faulthandler
faulthandler.enable()

print("Step 1: Basic CUDA ops...")
x = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
y = x @ w
torch.cuda.synchronize()
print(f"  OK: matmul {y.shape}")

print("Step 2: Import vLLM model class...")
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
print("  OK")

print("Step 3: Load vLLM engine with CUDA_LAUNCH_BLOCKING to get sync crash...")
from vllm import LLM
try:
    llm = LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
        max_model_len=512,
        max_num_batched_tokens=512,
        skip_mm_profiling=True,
        limit_mm_per_prompt={"audio": 1},
        compilation_config={"custom_ops": ["none"]},
    )
    print("  Model loaded successfully!")
except Exception as e:
    print(f"  Model load failed: {e}")
    import traceback
    traceback.print_exc()
