---
title: Open-ASR Model Explorer
emoji: 🎙
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Open-ASR Model Explorer

[![Deploy to Hugging Face Spaces](https://img.shields.io/badge/🤗%20Deploy-Hugging%20Face%20Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![vLLM](https://img.shields.io/badge/vLLM-0.18-purple?style=for-the-badge)](https://vllm.ai)

A hybrid inference testbed for evaluating top open-source ASR models (Cohere, Qwen3-ASR, IBM Granite) featuring vLLM server-side routing and transformers.js WebGPU client-side inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (React/Vite)                  │
│                                                          │
│  ┌──────────────┐   Strategy Router   ┌──────────────┐  │
│  │ Model Select │──────────────────►  │ InferRouter  │  │
│  └──────────────┘                     └──────┬───────┘  │
│                                              │           │
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
│                           │                   │        │
│  ┌────────────────────────▼───────────────────▼──────┐ │
│  │              Metrics Dashboard                    │ │
│  │          TTFT · ITL · RTFx                        │ │
│  └───────────────────────────────────────────────────┘ │
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
| `Cohere-transcribe-03-2026` | **Server (vLLM)** | Routes to FastAPI backend |
| `Qwen3-ASR-1.7B` | **Server (vLLM)** | Routes to FastAPI backend |
| `ibm-granite/granite-4.0-1b-speech` | **Server (vLLM)** | Routes to FastAPI backend |
| `Cohere-Transcribe-WebGPU` | **Client (WebGPU)** | Runs in-browser via transformers.js |
| `Whisper-Large-v3-Turbo-WebGPU` | **Client (WebGPU)** | Whisper Large v3 Turbo in-browser via transformers.js |

### WebGPU Client-Side Features
- **Zero-server transcription** – audio never leaves the browser
- **Persistent ONNX cache** – model weights cached via the browser Cache API
- **LocalAgreement-2 streaming** – unconfirmed partial hypotheses are held back until two consecutive passes agree on a prefix

### Backend vLLM Configuration
- **Chunked Prefill** – `enable_chunked_prefill=True` prevents long audio prefills from starving decode steps
- **Pre-batching normalisation** – all audio padded/truncated to 30 s before entering the scheduler
- **Resource constraints** – `max_num_batched_tokens=2048`, `gpu_memory_utilization=0.88`

---

## Project Structure

```
model-explorer-open-asr/
├── docker-compose.yml     # Full-stack orchestration (backend + frontend)
├── backend/
│   ├── app.py             # FastAPI application with vLLM integration
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # CUDA 12.1 container image
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
4. Push your fork – Spaces will automatically build the Docker image and start the server.

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
Returns `{ "status": "ok", "loaded_models": [...] }`.

### `GET /models`
Returns the list of available server-side model keys.

