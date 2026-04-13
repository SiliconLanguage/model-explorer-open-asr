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
- `WHISPER_MODEL=large-v3` (worker model size: `large-v3`, `medium`, `small`, `base`)
- `NUM_WORKERS=4` (parallel GPU inference threads in the worker)
- `CORS_ALLOWED_ORIGIN=*` (lock to domain in production)

### 2) Start the Full Stack

```bash
docker compose up --build
```

This launches five services:

| Service | Role | Port |
|---|---|---|
| **gateway** | Caddy reverse-proxy, TLS termination | `:80` / `:443` |
| **frontend** | React SPA + Nginx API proxy | `:3000` → Nginx `:80` |
| **backend** | FastAPI — job enqueue, audio normalisation, REST API | `:8000` |
| **valkey** | Valkey (Redis-compatible) job queue + hash store | `:6379` |
| **worker** | faster-whisper GPU inference (BLPOP consumer) | — |

Frontend: `http://localhost:3000` · Backend API: `http://localhost:8000` · Swagger: `http://localhost:8000/docs`

### 3) Running Services Individually

```bash
# Backend + Valkey + Worker only (headless API mode)
docker compose up --build backend valkey worker

# Frontend only (requires healthy backend)
docker compose up --build frontend
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Browser (React / Vite)                         │
│                                                                     │
│  ┌──────────────┐   Strategy Router   ┌──────────────┐             │
│  │ Model Select │──────────────────►  │ InferRouter  │             │
│  └──────────────┘                     └──────┬───────┘             │
│                                              │                     │
│                           ┌──────────────────┴──────┐              │
│                    "WebGPU" in name?          else                  │
│                           │                         │              │
│                    ┌──────▼──────┐       ┌──────────▼─────────┐    │
│                    │  WebWorker  │       │ p-limit pool (8)   │    │
│                    │ (ONNX +     │       │ micro-batch upload  │    │
│                    │  LocalAg2)  │       │ + progress bar      │    │
│                    └──────┬──────┘       └──────────┬─────────┘    │
│                           │                         │              │
│  ┌────────────────────────▼─────────────────────────▼────────────┐ │
│  │  Metrics Dashboard: TTFT · ITL · RTFx  ·  Upload Progress    │ │
│  │  Jobs Sidebar: Progress bar  ·  Status  ·  Playback          │ │
│  └───────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │ HTTPS (Caddy TLS)
                    ┌──────────────▼──────────────┐
                    │        Caddy Gateway        │
                    │  TLS · HTTP/3 · Routing     │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │ /api/*                                  │ /*
    ┌─────────▼──────────┐                   ┌──────────▼─────────┐
    │     Nginx Proxy    │                   │   Nginx Static     │
    │  (strips /api/)    │                   │   (React SPA)      │
    └─────────┬──────────┘                   └────────────────────┘
              │
    ┌─────────▼──────────┐     ┌─────────────────┐     ┌──────────────────┐
    │   FastAPI Backend  │────►│     Valkey       │◄────│  Worker (GPU)    │
    │                    │     │  (job queue +    │     │  faster-whisper  │
    │  • POST /transcribe│     │   hash store)    │     │  large-v3        │
    │    /batch           │     │                 │     │                  │
    │  • Audio normalize │     │  LPUSH ──► BLPOP│     │  ThreadPool(4)   │
    │  • Job enqueue     │     │  HSET / HGETALL │     │  CUDA inference  │
    │  • /jobs CRUD      │     │  EXPIRE 24h     │     │                  │
    │  • /audio playback │     └─────────────────┘     │  Spool cleanup   │
    └────────────────────┘                              └──────────────────┘
              │
    ┌─────────▼──────────┐
    │  /data/audio_spool │  (Docker named volume, shared)
    └────────────────────┘
```

### Data Flow: Batch Upload (945 files)

1. **Browser** splits files into micro-batches of 5 via `p-limit(8)` — max 8 concurrent HTTP requests
2. **Nginx** proxies each `POST /api/transcribe/batch` → FastAPI (strips `/api/` prefix)
3. **FastAPI** normalises audio to 16 kHz mono WAV, writes to `/data/audio_spool`, creates Valkey hash with 24h TTL, `LPUSH` job ID to queue
4. **Worker** `BLPOP`s job IDs, runs faster-whisper inference in a thread pool, writes transcript + metrics back to Valkey hash
5. **Frontend** polls `GET /jobs/{id}` for status updates, displays progress bar
6. **Cleanup**: Spool files are deleted after successful transcription; Valkey hashes expire after 24h; orphan sweep runs on `DELETE /jobs`

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
├── docker-compose.yml        # 5-service orchestration (gateway, frontend, backend, valkey, worker)
├── Caddyfile                 # Caddy reverse-proxy: TLS + API/UI routing
├── backend/
│   ├── app.py                # FastAPI: job enqueue, audio normalisation, CRUD
│   ├── worker.py             # BLPOP consumer: faster-whisper GPU inference
│   ├── requirements.txt
│   └── Dockerfile            # CUDA 12.8 (Blackwell-compatible)
└── frontend/
    ├── Dockerfile            # Multi-stage Vite build → Nginx serve
    ├── nginx.conf            # API proxy + SSE streaming + SPA fallback
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
        │   ├── StagedFiles.jsx       # Staging area + upload progress bar
        │   ├── JobsList.jsx           # Jobs sidebar + completion progress bar
        │   ├── MetricsDashboard.jsx
        │   └── TranscriptDisplay.jsx
        ├── services/
        │   └── inferenceRouter.js    # Strategy routing + p-limit batch upload
        └── workers/
            └── webgpu.worker.js      # transformers.js + LocalAgreement-2
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
Synchronous full transcription (in-process HF pipeline).

### `POST /transcribe/stream`
Same as above but returns **Server-Sent Events** with token-by-token streaming.

### `POST /transcribe/async`
Enqueue a single file for background transcription. Returns `{ job_id, status: "accepted" }`.

### `POST /transcribe/batch`
Enqueue multiple files. Returns `{ jobs: [{ id, filename }, ...] }`.

### `GET /jobs?session_id=...`
List all jobs scoped to a session.

### `GET /jobs/{job_id}`
Poll job status. Returns full hash: `status`, `transcript`, `segments`, `ttft_ms`, `itl_ms`, `processing_time_s`, ...

### `DELETE /jobs?session_id=...`
Delete all session jobs + sweep orphaned spool files.

### `DELETE /jobs/{job_id}`
Delete a single job and its spool file.

### `POST /jobs/{job_id}/resubmit`
Re-enqueue a completed/failed job.

### `GET /models`
List available models with metadata.

### `GET /audio/{filename}`
Serve audio from the spool directory (browser playback).

### `GET /health`
Returns readiness details such as `status`, `mode`, `loaded_models`, and `failed_models`.
`status=ok` means at least one real model is ready; `status=degraded` means no real model is currently serving.

### `GET /models`
Returns the list of available server-side model keys.

