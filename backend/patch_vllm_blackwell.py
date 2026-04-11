"""Patch vLLM attention backend priorities for Blackwell (SM 12.0+).

vLLM 0.18.0 ships FA2 with sm_80 PTX (forward-compatible to sm_120
via CUDA JIT) and FA3 with sm_90a cubins (architecture-specific, NOT
forward-compatible).  FlashInfer and Triton-based backends segfault on
SM 12.0 due to a Triton 3.6.0 `ir.builder` NULL-deref bug.

This patch:
1. Decoder attention: puts FLASH_ATTN first (FA2 via PTX forward-
   compilation) — genuine GPU-accelerated fused kernels, no Triton.
2. ViT / encoder attention: puts TORCH_SDPA first (PyTorch native,
   confirmed stable on SM 12.0 via hardware baseline test).
"""
import pathlib, glob, sys

candidates = glob.glob("/opt/venv/**/vllm/platforms/cuda.py", recursive=True)
if not candidates:
    print("vLLM cuda.py not found – skipping patch", file=sys.stderr)
    sys.exit(0)

cuda_py = pathlib.Path(candidates[0])
src = cuda_py.read_text()

# 1. Decoder (non-MLA) attention: replace the SM >= 12 branch.
#    FLASH_ATTN (FA2) first — its sm_80 PTX is forward-compiled by the
#    CUDA driver to sm_120 at runtime.  FLASHINFER/TRITON_ATTN/
#    FLEX_ATTENTION all ultimately invoke Triton JIT which segfaults.
OLD_NON_MLA = (
    "        if device_capability.major >= 12:\n"
    "            return [\n"
    "                AttentionBackendEnum.FLASHINFER,\n"
    "                AttentionBackendEnum.FLEX_ATTENTION,\n"
    "                AttentionBackendEnum.TRITON_ATTN,\n"
    "                AttentionBackendEnum.FLASH_ATTN,\n"
    "            ]"
)
NEW_NON_MLA = (
    "        if device_capability.major >= 12:\n"
    "            return [\n"
    "                AttentionBackendEnum.FLASH_ATTN,\n"
    "            ]"
)

if OLD_NON_MLA in src:
    src = src.replace(OLD_NON_MLA, NEW_NON_MLA)
    print("Patched decoder attention: FLASH_ATTN first for SM 12.0+")
else:
    print("WARNING: decoder attention pattern not found – may already be patched",
          file=sys.stderr)

# 2. ViT / encoder attention: add SM 12.0+ branch with TORCH_SDPA first.
OLD_VIT = (
    "        if cls.has_device_capability(80):\n"
    "            return [\n"
    "                AttentionBackendEnum.FLASH_ATTN,\n"
    "                AttentionBackendEnum.TRITON_ATTN,\n"
    "                AttentionBackendEnum.TORCH_SDPA,\n"
    "                AttentionBackendEnum.FLASHINFER,\n"
    "            ]"
)
NEW_VIT = (
    "        if cls.has_device_capability(120):\n"
    "            return [\n"
    "                AttentionBackendEnum.TORCH_SDPA,\n"
    "                AttentionBackendEnum.FLASH_ATTN,\n"
    "            ]\n"
    "        if cls.has_device_capability(80):\n"
    "            return [\n"
    "                AttentionBackendEnum.FLASH_ATTN,\n"
    "                AttentionBackendEnum.TRITON_ATTN,\n"
    "                AttentionBackendEnum.TORCH_SDPA,\n"
    "                AttentionBackendEnum.FLASHINFER,\n"
    "            ]"
)

if OLD_VIT in src:
    src = src.replace(OLD_VIT, NEW_VIT)
    print("Patched ViT attention: TORCH_SDPA first for SM 12.0+")
else:
    print("WARNING: ViT attention pattern not found – may already be patched",
          file=sys.stderr)

cuda_py.write_text(src)
print("vLLM Blackwell attention patch applied to", cuda_py)
