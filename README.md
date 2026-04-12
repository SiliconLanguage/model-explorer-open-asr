---
title: Open-ASR Model Explorer
emoji: 🎙
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Open-ASR Model Explorer

[![Deploy to Hugging Face Spaces](https://img.shields.io/badge/🤗%20Deploy-Hugging%20Face%20Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![vLLM](https://img.shields.io/badge/vLLM-0.19-purple?style=for-the-badge)](https://vllm.ai)

A hybrid inference testbed for evaluating top open-source ASR models (Cohere, Qwen3-ASR, IBM Granite) featuring vLLM server-side routing and transformers.js WebGPU client-side inference.

## Quick Start

### 1) Environment Setup

Create a local environment file from the template:

```bash
cp .env.example .env
```

Set required values in `.env`:

- `HF_TOKEN` — required for gated server-side models (for example Cohere and Granite)
- `GPU_MEMORY_UTILIZATION=0.65` (start conservative; tune upward once stable)
- `OPENASR_ENGINE_INIT_TIMEOUT_S=120` (first cold boot may include compilation)
- `OPENASR_VLLM_MAX_MODEL_LEN=8192`
- `OPENASR_VLLM_ENFORCE_EAGER=true` (recommended on WSL to avoid compile-path crashes)
- `ALLOW_MOCK_FALLBACK=false` (recommended to avoid masking real backend failures)
- `PORT=7860` (for Hugging Face Spaces Docker runtime)

### 2) Start the Full Stack

```bash
docker compose up --build
```

Frontend is served on `http://localhost:3000` and backend API on `http://localhost:8000`.

### 3) Agent Protocol

When infrastructure or documentation changes, follow this protocol:

- `/sync-infrastructure` — sync deployment documentation from `docker-compose.yml` and `nginx.conf` into the AgentWiki topology docs.
- `/document-logic` — analyze implementation logic and update conceptual docs (for example LocalAgreement-2 behavior and router integration).

### Verification Checklist

- [ ] **WebGPU**: WebGPU models run client-side and stream stable/unstable transcript updates in the UI.
- [ ] **Streaming**: Server-side SSE path streams token updates continuously without proxy buffering delays.
- [ ] **vLLM**: Backend starts with expected scheduler/runtime settings and serves `/health` successfully.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (React/Vite)                 │
│                                                         │
│  ┌──────────────┐   Strategy Router   ┌──────────────┐  │
│  │ Model Select │──────────────────►  │ InferRouter  │  │
│  └──────────────┘                     └──────┬───────┘  │
│                                              │          │
│                           ┌──────────────────┴──────┐   │
│                           │                         │   │
│                    "WebGPU" in name?         else       │
│                           │                         │   │
│                    ┌──────▼──────┐        ┌────────▼─┐  │
│                    │  WebWorker  │        │  fetch() │  │
│                    │ (transforms │        │  SSE     │  │
│                    │  .js ONNX)  │        └────┬─────┘  │
│                    │ LocalAgrmt2 │             │        │
│                    └──────┬──────┘             │        │
│                           │                   │         │
│  ┌────────────────────────▼───────────────────▼──────┐  │
│  │              Metrics Dashboard                    │  │
│  │          TTFT · ITL · RTFx                        │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────┘
                               │ HTTP POST (multipart/SSE)
                    ┌──────────▼──────────┐
                    │   FastAPI Backend   │
                    │                     │
                    │  Audio Normaliser   │
                    │  (pad/truncate 30s) │
                    │         │           │
                    │  ┌──────▼───────┐   │
                    │  │  vLLM Engine │   │
                    │  │  chunked     │   │
                    │  │  prefill     │   │
                    │  │  max_tokens  │   │
                    │  │  =2048       │   │
                    │  │  gpu_util    │   │
                    │  │  =0.88       │   │
                    │  └──────────────┘   │
                    └─────────────────────┘
```

### Strategy-Based Routing

| Model | Execution Mode | Description |
|---|---|---|
| `openai/whisper-base` | **Server (HF-GPU / HF-CPU)** | Standard HF transformers pipeline |
| `CohereLabs/cohere-transcribe-03-2026` | **Server (HF-GPU / HF-CPU)** | Custom model wrapper (`trust_remote_code`) |
| `Qwen3-ASR-1.7B` | **Server (HF-GPU)** | Via `qwen-asr` package (architecture not in transformers) |
| `ibm-granite/granite-4.0-1b-speech` | **Server (HF-GPU / vLLM Cloud)** | Chat-template + `<\|audio\|>` multimodal wrapper |
| `Xenova/whisper-tiny` / `Xenova/whisper-base` | **Client (WebGPU)** | Runs in-browser via transformers.js ONNX |
| `onnx-community/cohere-transcribe-03-2026-ONNX` | **Client (WebGPU)** | Cohere in-browser via transformers.js ONNX |

### WebGPU Client-Side Features
- **Zero-server transcription** – audio never leaves the browser
- **Persistent ONNX cache** – model weights cached via the browser Cache API
- **LocalAgreement-2 streaming** – unconfirmed partial hypotheses are held back until two consecutive passes agree on a prefix

### Backend vLLM Configuration
- **Chunked Prefill** – `enable_chunked_prefill=True` prevents long audio prefills from starving decode steps
- **Pre-batching normalisation** – all audio padded/truncated to 30 s before entering the scheduler
- **Resource constraints** – `max_num_batched_tokens=512`, env-driven `gpu_memory_utilization`, auto-derived `max_model_len`

### Hardware Constraints: vLLM on SM 12.0 (Blackwell)

> **Affects:** Local development on NVIDIA RTX 5070 Ti and other SM 12.0 (Blackwell) GPUs.

**The Issue.** While vLLM 0.19.0 is the target production engine, running it locally on SM 12.0 hardware fails due to an upstream `ir.builder` NULL-dereference bug in the bundled Triton 3.6.0 compiler. This crashes all native attention kernels — FlashAttention 2, FlashInfer 0.6.6, and every Triton-based backend (TRITON_ATTN, FLEX_ATTENTION). The NVFP4 kernel guard patches in vLLM 0.19.0 (PR #38423, #38126) address quantization paths but do not fix the attention backend segfaults.

**Graceful Degradation.** The testbed handles this without crashing. When a vLLM route is requested on unsupported local hardware, the backend returns `503 Service Unavailable` with a descriptive message. Users should select an **HF-GPU** variant from the model dropdown to process audio via the standard Hugging Face transformers pipeline, which uses native PyTorch CUDA primitives and runs reliably on SM 12.0.

**Future Resolution.** Once Triton ≥ 3.6.1 ships with the `ir.builder` fix and is bundled into a future vLLM release, the local SM 12.0 constraint will be lifted and vLLM will serve as the primary engine on all hardware.

For the full architectural decision record, see [ADR-001: vLLM vs. Hugging Face Inference Routing](https://github.com/SiliconLanguage/model-explorer-open-asr-AgentWiki/blob/main/Architecture/ADR-001-vLLM_Constraints.md).

---

## Project Structure

```
model-explorer-open-asr/
├── docker-compose.yml     # Full-stack orchestration (backend + frontend)
├── backend/
│   ├── app.py             # FastAPI application with vLLM integration
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # CUDA 12.8 container image (Blackwell-compatible)
└── frontend/
    ├── Dockerfile         # Multi-stage Vite build → Nginx serve
    ├── nginx.conf         # Nginx: API proxy + SSE streaming + SPA fallback
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── components/
        │   ├── ModelSelector.jsx
        │   ├── AudioRecorder.jsx
        │   ├── MetricsDashboard.jsx
        │   └── TranscriptDisplay.jsx
        ├── services/
        │   └── inferenceRouter.js   # Strategy-based routing logic
        └── workers/
            └── webgpu.worker.js     # transformers.js + LocalAgreement-2
```

---

## Metrics

| Metric | Definition |
|---|---|
| **TTFT** (Time-to-First-Token) | Time from request submission to the first generated token |
| **ITL** (Inter-Token Latency) | Mean time between successive tokens during the decode phase |
| **RTFx** (Real-Time Factor) | `audio_duration / processing_time` – values > 1 indicate faster-than-real-time |

---

## Local Development

### Prerequisites
- Node.js ≥ 18
- Python 3.11
- (Optional) NVIDIA GPU with CUDA 12.1 for vLLM inference

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Swagger docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.  
API requests are proxied to `http://localhost:8000` via Vite's dev proxy.

---

## Docker Setup

### Backend only

```bash
cd backend
docker build -t open-asr-backend .
docker run --gpus all -p 8000:8000 \
  -v $HOME/.cache/huggingface:/hf_cache \
  open-asr-backend
```

### Full stack with Docker Compose

```bash
# From repo root
docker compose up --build
```

The UI will be available at `http://localhost:3000` and the API at `http://localhost:8000`.

---

## Deploy to Hugging Face Spaces

1. Fork this repository.
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with the **Docker** SDK.
3. Set the following Space secrets:
   - `HF_TOKEN` – your Hugging Face token (for gated model downloads)
4. The Space Docker container exposes port `7860` (Hugging Face default).
5. Push your fork – Spaces will automatically build the Docker image and start the server.

[![Deploy to Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-lg.svg)](https://huggingface.co/spaces)

---

## API Reference

### `POST /transcribe`
Upload audio for full (non-streaming) transcription.

| Field | Type | Description |
|---|---|---|
| `audio` | `File` | Audio file (wav, mp3, ogg, flac, webm) |
| `model` | `string` | Model key, e.g. `Qwen3-ASR-1.7B` |

Response:
```json
{
  "transcript": "Hello, world!",
  "model": "Qwen3-ASR-1.7B",
  "ttft_ms": 123.4,
  "itl_ms": 8.2,
  "rtfx": 4.56
}
```

### `POST /transcribe/stream`
Same as above but returns **Server-Sent Events** with token-by-token streaming.

### `GET /health`
Returns readiness details such as `status`, `mode`, `loaded_models`, and `failed_models`.
`status=ok` means at least one real vLLM model is ready; `status=degraded` means no real model is currently serving.

### `GET /models`
Returns the list of available server-side model keys.

