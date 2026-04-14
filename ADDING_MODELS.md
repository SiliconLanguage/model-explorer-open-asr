# Adding a New ASR Model

This guide walks through the steps to add a new ASR model to the Open-ASR Model Explorer. There are three integration paths depending on where the model runs.

---

## Integration Paths

| Path | Engine | Runs In | Concurrency | Example |
|---|---|---|---|---|
| **A. HF Transformers** | `hf-gpu` / `hf-cpu` | Backend (`app.py`) | Per-request | Qwen3-ASR, Granite |
| **B. Batch Worker** | Custom in `worker.py` | Worker container | Configurable | VibeVoice-ASR, faster-whisper |
| **C. WebGPU (ONNX)** | Browser-side | Client | N/A | Whisper Tiny, Cohere ONNX |

Pick the path that matches your model, then follow the steps below.

---

## Path A: HF Transformers (Server-Side)

Use this when the model works with `transformers` AutoModel/pipeline and fits alongside vllm in the backend image (i.e., requires `transformers<5`).

### Step 1 — Register in backend

Add the model to `SUPPORTED_MODELS` in `backend/app.py`:

```python
SUPPORTED_MODELS: dict[str, str] = {
    # ... existing models ...
    "your-org/your-model": "your-org/your-model",
}
```

The `/models` endpoint auto-deduplicates, so one entry with the full HF ID is sufficient.

### Step 2 — Add to frontend dropdown

Add an entry to the `MODELS` array in `frontend/src/components/ModelSelector.jsx`:

```javascript
{ value: 'your-model-hf-gpu', id: 'your-org/your-model', label: 'Your Model Name (HF-GPU)', mode: 'server', engine: 'hf-gpu' },
```

Key fields:
- `value` — unique slug (used as React key and localStorage persistence)
- `id` — must match the key you added to `SUPPORTED_MODELS`
- `label` — display name in the dropdown
- `mode` — `'server'` for backend models
- `engine` — `'hf-gpu'`, `'hf-cpu'`, or `'faster_whisper'`

### Step 3 — Rebuild frontend

```bash
docker compose up -d --build frontend
```

### Step 4 — Verify

Select the model in the dropdown and transcribe a sample. The backend will download the model weights on first use (cached in the `hf_cache` volume).

---

## Path B: Batch Worker (Custom Engine)

Use this when the model needs its own inference logic, different dependencies, or dedicated GPU memory (e.g., the 9B VibeVoice-ASR fills 24 GB A10G).

### Step 1 — Add dependencies

Add any new Python packages to `backend/requirements-worker.txt`:

```
your-new-package>=1.0.0
```

> **Important:** The worker has its own `Dockerfile.worker` and `requirements-worker.txt`, separate from the backend. This avoids dependency conflicts (e.g., vllm requires `transformers<5`, but VibeVoice needs `transformers>=5.3`).

### Step 2 — Add loader and transcriber to worker

In `backend/worker.py`, add:

1. **Configuration** — new env var for the model ID:
   ```python
   YOUR_MODEL = os.getenv("YOUR_MODEL", "your-org/your-model")
   ```

2. **Loader function** — lazy-load the model once at startup:
   ```python
   _your_model = None

   def _load_your_model():
       global _your_model
       if _your_model is not None:
           return _your_model
       # ... load model ...
       return _your_model
   ```

3. **Transcriber function** — must return a dict with this shape:
   ```python
   def _transcribe_your_model(audio_path: str, language: str | None) -> dict:
       return {
           "transcript": "full text",
           "segments": [{"start": 0.0, "end": 1.5, "text": "..."}],
           "duration_s": 10.5,
           "language_detected": "en",
           "ttft_ms": 150.0,       # or None
           "itl_ms": 25.0,         # or None
       }
   ```

4. **Route in `_process_job()`** — add an `elif` branch:
   ```python
   if ASR_ENGINE == "your_engine":
       result = _transcribe_your_model(str(audio_path), language)
   ```

5. **Startup in `_main()`** — add engine-specific pool and model preload:
   ```python
   if ASR_ENGINE == "your_engine":
       pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="your-engine")
       await loop.run_in_executor(pool, _load_your_model)
       concurrency = 1  # adjust based on VRAM
   ```

### Step 3 — Add env vars to docker-compose

In `docker-compose.yml`, add to the worker service environment:

```yaml
- ASR_ENGINE=${ASR_ENGINE:-whisper}
- YOUR_MODEL=${YOUR_MODEL:-your-org/your-model}
```

### Step 4 — Register in backend + frontend

Same as Path A, Steps 1–2. If the model has a custom engine name, also add a badge and tooltip in `ModelSelector.jsx`:

```javascript
// In BADGE object:
'your-engine-server': { label: 'Your Engine · Batch Worker', color: '#dc2626' },

// In modelBadge():
if (model.engine === 'your_engine') return BADGE['your-engine-server'];

// In modelTitle():
if (model.engine === 'your_engine') {
  return 'Description of your engine';
}
```

### Step 5 — Rebuild and switch

```bash
ASR_ENGINE=your_engine docker compose up -d --build worker
docker compose up -d --build frontend
```

---

## Path C: WebGPU (Client-Side ONNX)

Use this for models that run entirely in the browser via ONNX Runtime Web + WebGPU.

### Step 1 — Publish ONNX model

Ensure the model is available as an ONNX export on Hugging Face (e.g., `onnx-community/your-model-ONNX`).

### Step 2 — Add to frontend dropdown

```javascript
{ value: 'your-model-webgpu', id: 'onnx-community/your-model-ONNX', label: 'Your Model (WebGPU)', mode: 'webgpu' },
```

WebGPU models use `mode: 'webgpu'` and are routed to the Web Worker (`workers/webgpu.worker.js`) — no backend changes needed.

### Step 3 — Rebuild frontend

```bash
docker compose up -d --build frontend
```

---

## Concurrency Guidelines

| Model Size | GPU | Recommended Concurrency |
|---|---|---|
| < 1B params | A10G 24 GB | `NUM_WORKERS=4` (ThreadPoolExecutor) |
| 1–3B params | A10G 24 GB | 2–3 concurrent |
| 8–9B params | A10G 24 GB | 1 (sequential) |
| > 10B params | A10G 24 GB | 1 + reduced `tokenizer_chunk_size` |

Set concurrency in the `_main()` function of `worker.py` based on your model's VRAM footprint.

---

## File Checklist

| File | Path A | Path B | Path C |
|---|---|---|---|
| `backend/app.py` (`SUPPORTED_MODELS`) | ✅ | ✅ | — |
| `frontend/src/components/ModelSelector.jsx` | ✅ | ✅ | ✅ |
| `backend/worker.py` | — | ✅ | — |
| `backend/requirements-worker.txt` | — | ✅ | — |
| `docker-compose.yml` (worker env) | — | ✅ | — |
| Rebuild frontend | ✅ | ✅ | ✅ |
| Rebuild worker | — | ✅ | — |
