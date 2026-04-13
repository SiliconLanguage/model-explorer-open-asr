"""Quick smoke test: can vLLM V0 + fork load a model on this GPU?"""
import os, sys

# Force V0 engine BEFORE importing vllm
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

from vllm import LLM, SamplingParams  # noqa: E402

print(f"vLLM version: {__import__('vllm').__version__}")
print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")
print(f"VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")

model = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
print(f"\nLoading model: {model}")

try:
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("Engine created! Running inference...")
    out = llm.generate(["Hello world"], SamplingParams(max_tokens=5))
    print(f"SUCCESS: {out[0].outputs[0].text!r}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
