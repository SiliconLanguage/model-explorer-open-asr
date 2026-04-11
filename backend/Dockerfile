# ── Base image: CUDA 12.8 + cuDNN (devel) for PyTorch, vLLM, and FlashInfer ──
# The devel variant provides nvcc for FlashInfer JIT compilation of attention
# kernels.  CUDA 12.8 is the minimum with native Blackwell (SM 12.0) support.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL maintainer="Open-ASR Model Explorer"
LABEL description="FastAPI backend for hybrid ASR inference testbed (vLLM server-side routing)"

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
        python3.11 \
        python3.11-dev \
    python3.11-venv \
        python3-pip \
        libsndfile1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` and `pip` resolve to 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create an isolated Python 3.11 environment and use its pip for installs.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade "pip<26" \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Patch vLLM attention backend priorities for Blackwell (SM 12.0) ──────────
# FlashAttention 2 segfaults during encoder cache profiling on SM 12.0 because
# the bundled CUDA 12.1 toolkit predates the Blackwell architecture.  Swap the
# priority so FlashInfer (main attn) and TORCH_SDPA (vit/encoder attn) are
# tried first; FLASH_ATTN becomes a last resort.
COPY patch_vllm_blackwell.py /tmp/patch_vllm_blackwell.py
RUN python /tmp/patch_vllm_blackwell.py \
    && find /opt/venv -path '*/vllm/platforms/__pycache__' -exec rm -rf {} + 2>/dev/null \
    && rm /tmp/patch_vllm_blackwell.py \
    || true

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py .

# ── Runtime environment variables ─────────────────────────────────────────────
# vLLM: disable usage statistics ping
ENV VLLM_NO_USAGE_STATS=1
# Hugging Face model cache directory (mount a volume here for persistence)
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_CACHE=/hf_cache

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
