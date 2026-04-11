#!/usr/bin/env python3
"""PyTorch Hardware Baseline Test — MLOps Isolation Strategy

Proves the GPU, driver, and PyTorch stack are stable on SM 12.0 (Blackwell)
by exercising operations representative of a transformer forward pass.
If all tests pass, the fault is in the pre-compiled vLLM wheel, not the
hardware or driver stack.
"""
import sys
import time
import traceback

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16
PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)


def test_cuda_info():
    """Print device info and verify SM 12.0."""
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    mem = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
    arch_list = torch.cuda.get_arch_list()
    detail = f"{name}, SM {cap[0]}.{cap[1]}, {mem:.1f} GiB, archs={arch_list}"
    report("CUDA device info", cap[0] >= 12, detail)


def test_large_matmul():
    """4096×4096 bf16 matmul — exercises tensor cores."""
    a = torch.randn(4096, 4096, dtype=DTYPE, device=DEVICE)
    b = torch.randn(4096, 4096, dtype=DTYPE, device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = c.shape == (4096, 4096) and c.isfinite().all().item()
    report("Large matmul (4096² bf16)", ok, f"{dt:.1f} ms")


def test_sdpa_attention():
    """Multi-head scaled dot-product attention (PyTorch native SDPA)."""
    B, H, S, D = 2, 32, 1024, 128  # batch, heads, seq_len, head_dim
    q = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, H, S, D, dtype=DTYPE, device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = out.shape == (B, H, S, D) and out.isfinite().all().item()
    report("SDPA attention (2×32×1024×128 bf16)", ok, f"{dt:.1f} ms")


def test_rmsnorm():
    """RMSNorm-style operation (manual, no fused kernel)."""
    hidden = 4096
    x = torch.randn(2, 1024, hidden, dtype=DTYPE, device=DEVICE)
    weight = torch.ones(hidden, dtype=DTYPE, device=DEVICE)
    eps = 1e-6
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps) * weight
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = normed.shape == x.shape and normed.isfinite().all().item()
    report("RMSNorm (2×1024×4096 bf16)", ok, f"{dt:.1f} ms")


def test_softmax_crossentropy():
    """Softmax + cross-entropy loss — exercises numerically sensitive ops."""
    vocab = 152064  # Qwen3 vocab size
    logits = torch.randn(2, 512, vocab, dtype=torch.float32, device=DEVICE)
    targets = torch.randint(0, vocab, (2, 512), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab), targets.view(-1)
    )
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = loss.isfinite().item()
    report("Softmax + CrossEntropy (vocab=152064)", ok, f"loss={loss.item():.4f}, {dt:.1f} ms")


def test_memory_stress():
    """Allocate ~4 GiB, verify, free — simulates model weight loading."""
    chunks = []
    total_gb = 4.0
    chunk_gb = 0.5
    n_chunks = int(total_gb / chunk_gb)
    elems = int(chunk_gb * 1024**3 / 2)  # bf16 = 2 bytes
    try:
        for i in range(n_chunks):
            t = torch.randn(elems, dtype=DTYPE, device=DEVICE)
            chunks.append(t)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        ok = True
        detail = f"allocated {n_chunks}×{chunk_gb} GiB = {total_gb} GiB, peak={peak:.2f} GiB"
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        detail = str(e)
    finally:
        del chunks
        torch.cuda.empty_cache()
    report("Memory stress (4 GiB alloc/free)", ok, detail)


def test_conv1d_audio():
    """1-D convolution representative of audio feature extraction."""
    # Simulates Whisper-style mel extraction frontend
    B, C_in, L = 2, 1, 16000 * 30  # 30 seconds of 16 kHz audio
    C_out, K, S = 80, 400, 160
    x = torch.randn(B, C_in, L, dtype=torch.float32, device=DEVICE)
    conv = torch.nn.Conv1d(C_in, C_out, K, stride=S).to(device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = conv(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = out.isfinite().all().item()
    report("Conv1d audio frontend (30s @ 16kHz)", ok, f"out={list(out.shape)}, {dt:.1f} ms")


def test_embedding_gather():
    """Embedding lookup — exercises memory-bound gather patterns."""
    vocab, dim = 152064, 2048
    emb = torch.nn.Embedding(vocab, dim).to(dtype=DTYPE, device=DEVICE)
    ids = torch.randint(0, vocab, (2, 2048), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = emb(ids)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    ok = out.shape == (2, 2048, dim) and out.isfinite().all().item()
    report("Embedding gather (152064×2048 bf16)", ok, f"{dt:.1f} ms")


if __name__ == "__main__":
    print("=" * 72)
    print("PyTorch Hardware Baseline — SM 12.0 (Blackwell) Isolation Test")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 72)
    
    tests = [
        test_cuda_info,
        test_large_matmul,
        test_sdpa_attention,
        test_rmsnorm,
        test_softmax_crossentropy,
        test_memory_stress,
        test_conv1d_audio,
        test_embedding_gather,
    ]
    
    for fn in tests:
        try:
            fn()
        except Exception:
            report(fn.__name__, False, traceback.format_exc().splitlines()[-1])
    
    print("=" * 72)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    if FAIL == 0:
        print("VERDICT: GPU + driver + PyTorch stack is STABLE on SM 12.0.")
        print("         The fault lies in the pre-compiled vLLM 0.18.0 wheel.")
    else:
        print("VERDICT: Hardware/driver instability detected — fix GPU stack first.")
    print("=" * 72)
    sys.exit(FAIL)
