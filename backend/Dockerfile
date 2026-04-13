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

# ── Copy application source ───────────────────────────────────────────────────
COPY app.py .
COPY worker.py .

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
