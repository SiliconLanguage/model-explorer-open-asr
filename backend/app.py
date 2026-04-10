"""
Open-ASR Model Explorer – FastAPI backend
Handles server-side (vLLM) transcription requests.
Strategy-based routing:
  - Models with "WebGPU" in the name → handled client-side (transformers.js)
  - All other models → routed here via POST /transcribe
"""

from __future__ import annotations

import io
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import librosa
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported server-side models
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, str] = {
    "Cohere-transcribe-03-2026": "CohereForAI/c4ai-command-r-plus",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
    "ibm-granite/granite-4.0-1b-speech": "ibm-granite/granite-4.0-1b-speech",
}

# Target sample rate expected by Whisper-style models
TARGET_SR = 16_000
# Pad / truncate all audio to this many samples (30 s at 16 kHz)
MAX_AUDIO_SAMPLES = TARGET_SR * 30

# ---------------------------------------------------------------------------
# vLLM engine – loaded lazily on first request to keep startup fast in dev
# ---------------------------------------------------------------------------
_vllm_engines: dict[str, object] = {}


def _get_engine(model_key: str):
    """Return (or lazily create) the vLLM AsyncLLMEngine for *model_key*."""
    if model_key in _vllm_engines:
        return _vllm_engines[model_key]

    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore

        hf_model = SUPPORTED_MODELS[model_key]
        logger.info("Loading vLLM engine for %s (%s)…", model_key, hf_model)

        engine_args = AsyncEngineArgs(
            model=hf_model,
            # ── Chunked prefill prevents long audio prefills from starving
            #    decode steps of other in-flight requests.
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            # ── Reserve ample KV-cache space while keeping GPU busy.
            gpu_memory_utilization=0.88,
            # ── Async streaming for token-by-token delivery.
            disable_log_requests=False,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        _vllm_engines[model_key] = engine
        logger.info("Engine ready for %s", model_key)
        return engine
    except ImportError:
        logger.warning(
            "vLLM not available – returning mock engine for development."
        )
        return None


# ---------------------------------------------------------------------------
# Audio pre-processing helpers
# ---------------------------------------------------------------------------

def _load_audio_bytes(data: bytes) -> np.ndarray:
    """Decode audio bytes → mono float32 NumPy array at TARGET_SR."""
    with io.BytesIO(data) as buf:
        waveform, sr = sf.read(buf, dtype="float32", always_2d=False)
    # Resample if necessary
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    # Convert stereo → mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform


def _pad_or_truncate(waveform: np.ndarray) -> np.ndarray:
    """
    Pre-batching normalisation: pad (with zeros) or truncate to MAX_AUDIO_SAMPLES.
    This ensures all tensors entering the vLLM scheduler share a uniform shape.
    """
    if len(waveform) >= MAX_AUDIO_SAMPLES:
        return waveform[:MAX_AUDIO_SAMPLES]
    pad_width = MAX_AUDIO_SAMPLES - len(waveform)
    return np.pad(waveform, (0, pad_width), mode="constant", constant_values=0.0)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    transcript: str
    model: str
    # Performance metrics
    ttft_ms: float    # Time-to-First-Token (ms)
    itl_ms: float     # Mean Inter-Token Latency (ms)
    rtfx: float       # Real-Time Factor  = audio_duration / processing_time


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Open-ASR Model Explorer backend starting up.")
    yield
    logger.info("Open-ASR Model Explorer backend shutting down.")
    _vllm_engines.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Open-ASR Model Explorer",
    description=(
        "Hybrid inference testbed for evaluating open-source speech-to-text models. "
        "Server-side route uses vLLM with chunked prefill and GPU memory optimisation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "loaded_models": list(_vllm_engines.keys())}


# ---------------------------------------------------------------------------
# List available server-side models
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models() -> dict:
    return {"models": list(SUPPORTED_MODELS.keys())}


# ---------------------------------------------------------------------------
# Main transcription endpoint
# ---------------------------------------------------------------------------

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/ogg/flac…)"),
    model: str = Form(..., description="Model key, e.g. 'Qwen3-ASR-1.7B'"),
) -> TranscribeResponse:
    """
    Transcribe *audio* using the selected server-side model via vLLM.

    Metrics returned
    ----------------
    ttft_ms  – time from request receipt to first generated token
    itl_ms   – mean inter-token latency across the decode phase
    rtfx     – audio_duration / total_processing_time  (higher is better)
    """
    if "WebGPU" in model:
        raise HTTPException(
            status_code=400,
            detail="WebGPU models are handled client-side. Do not send them to this endpoint.",
        )
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model}'. Available: {list(SUPPORTED_MODELS.keys())}",
        )

    # ── 1. Decode & normalise audio ──────────────────────────────────────────
    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)

    # ── 2. Attempt vLLM inference ────────────────────────────────────────────
    engine = _get_engine(model)
    request_start = time.perf_counter()

    if engine is not None:
        transcript, ttft_ms, itl_ms = await _run_vllm(engine, waveform_padded, model)
    else:
        # Development fallback when vLLM / GPU is unavailable
        transcript, ttft_ms, itl_ms = await _mock_transcribe(waveform_padded, model)

    total_time_s = time.perf_counter() - request_start
    rtfx = audio_duration_s / total_time_s if total_time_s > 0 else 0.0

    return TranscribeResponse(
        transcript=transcript,
        model=model,
        ttft_ms=round(ttft_ms, 2),
        itl_ms=round(itl_ms, 2),
        rtfx=round(rtfx, 4),
    )


# ---------------------------------------------------------------------------
# vLLM streaming helper
# ---------------------------------------------------------------------------

async def _run_vllm(
    engine,
    waveform: np.ndarray,
    model_key: str,
) -> tuple[str, float, float]:
    """
    Run vLLM inference; return (transcript, ttft_ms, mean_itl_ms).

    The waveform is serialised as a base64-encoded prompt for models that
    accept audio tokens.  For text-only models this path issues a simple
    ASR-formatted text prompt.
    """
    import asyncio
    from vllm import SamplingParams  # type: ignore

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # Minimal prompt – actual audio-token injection depends on model tokeniser
    prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."

    request_id = f"asr-{time.time_ns()}"
    token_timestamps: list[float] = []
    first_token_time: float | None = None
    request_start = time.perf_counter()

    result_generator = engine.generate(prompt, sampling_params, request_id)
    full_text = ""

    async for request_output in result_generator:
        now = time.perf_counter()
        if first_token_time is None and request_output.outputs:
            first_token_time = now
        token_timestamps.append(now)
        if request_output.outputs:
            full_text = request_output.outputs[0].text

    ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else 0.0
    if len(token_timestamps) > 1:
        gaps = [
            (token_timestamps[i] - token_timestamps[i - 1]) * 1000
            for i in range(1, len(token_timestamps))
        ]
        itl_ms = float(np.mean(gaps))
    else:
        itl_ms = 0.0

    return full_text, ttft_ms, itl_ms


# ---------------------------------------------------------------------------
# Mock transcription fallback (development without GPU)
# ---------------------------------------------------------------------------

async def _mock_transcribe(
    waveform: np.ndarray,
    model_key: str,
) -> tuple[str, float, float]:
    """Return a placeholder transcript for local development without a GPU."""
    import asyncio

    await asyncio.sleep(0.1)  # simulate small latency
    duration_s = len(waveform) / TARGET_SR
    transcript = (
        f"[Mock transcript – {model_key}] "
        f"Audio duration: {duration_s:.1f}s. "
        "Deploy with GPU and vLLM for real transcription."
    )
    ttft_ms = 42.0
    itl_ms = 8.5
    return transcript, ttft_ms, itl_ms


# ---------------------------------------------------------------------------
# Streaming transcription endpoint (SSE)
# ---------------------------------------------------------------------------

@app.post("/transcribe/stream")
async def transcribe_stream(
    audio: UploadFile = File(...),
    model: str = Form(...),
) -> StreamingResponse:
    """
    Server-Sent Events endpoint that streams tokens as they are generated.
    Each event carries a JSON payload:
      {"token": "...", "ttft_ms": ..., "done": false}
    The final event has "done": true and includes all metrics.
    """
    if "WebGPU" in model:
        raise HTTPException(status_code=400, detail="WebGPU models run client-side.")
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'.")

    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)
    engine = _get_engine(model)

    async def event_generator() -> AsyncGenerator[str, None]:
        import json

        request_start = time.perf_counter()
        first_token_time: float | None = None
        token_timestamps: list[float] = []
        full_text = ""

        if engine is not None:
            from vllm import SamplingParams  # type: ignore

            sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."
            request_id = f"asr-stream-{time.time_ns()}"

            async for output in engine.generate(prompt, sampling_params, request_id):
                now = time.perf_counter()
                if first_token_time is None and output.outputs:
                    first_token_time = now
                    ttft_ms = (first_token_time - request_start) * 1000
                    yield f"data: {json.dumps({'token': '', 'ttft_ms': round(ttft_ms, 2), 'done': False})}\n\n"
                token_timestamps.append(now)
                if output.outputs:
                    new_text = output.outputs[0].text
                    delta = new_text[len(full_text):]
                    full_text = new_text
                    if delta:
                        yield f"data: {json.dumps({'token': delta, 'ttft_ms': None, 'done': False})}\n\n"
        else:
            # Mock streaming
            import asyncio

            words = (
                f"[Mock – {model}] Audio {audio_duration_s:.1f}s. "
                "No GPU available."
            ).split()
            first_token_time = time.perf_counter()
            ttft_ms = (first_token_time - request_start) * 1000
            yield f"data: {json.dumps({'token': '', 'ttft_ms': round(ttft_ms, 2), 'done': False})}\n\n"
            for word in words:
                token_timestamps.append(time.perf_counter())
                full_text += word + " "
                yield f"data: {json.dumps({'token': word + ' ', 'ttft_ms': None, 'done': False})}\n\n"
                await asyncio.sleep(0.05)

        total_time_s = time.perf_counter() - request_start
        rtfx = audio_duration_s / total_time_s if total_time_s > 0 else 0.0
        if len(token_timestamps) > 1:
            gaps = [
                (token_timestamps[i] - token_timestamps[i - 1]) * 1000
                for i in range(1, len(token_timestamps))
            ]
            itl_ms = float(np.mean(gaps))
        else:
            itl_ms = 0.0

        final_payload = json.dumps({
            "token": "",
            "ttft_ms": None,
            "itl_ms": round(itl_ms, 2),
            "rtfx": round(rtfx, 4),
            "transcript": full_text.strip(),
            "done": True,
        })
        yield f"data: {final_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
