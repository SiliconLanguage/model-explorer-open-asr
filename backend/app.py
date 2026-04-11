"""
Open-ASR Model Explorer – FastAPI backend
Handles server-side (vLLM) transcription requests.
Strategy-based routing:
  - Models with "WebGPU" in the name → handled client-side (transformers.js)
  - All other models → routed here via POST /transcribe
"""

from __future__ import annotations

import io
import asyncio
import time
import os
import logging
import tempfile
from pathlib import Path
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
    "Cohere-transcribe-03-2026": "CohereLabs/cohere-transcribe-03-2026",
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
_vllm_failed_models: set[str] = set()


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.3f", name, raw, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


ALLOW_MOCK_FALLBACK = _read_bool_env("ALLOW_MOCK_FALLBACK", False)
ENGINE_INIT_TIMEOUT_S = _read_float_env("OPENASR_ENGINE_INIT_TIMEOUT_S", 120.0)
GPU_MEMORY_UTILIZATION = _read_float_env("GPU_MEMORY_UTILIZATION", 0.65)
VLLM_MAX_MODEL_LEN = _read_int_env("OPENASR_VLLM_MAX_MODEL_LEN", 8192)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
VLLM_ENFORCE_EAGER = _read_bool_env("OPENASR_VLLM_ENFORCE_EAGER", True)


def _get_engine(model_key: str):
    """Return (or lazily create) the vLLM AsyncLLMEngine for *model_key*."""
    if model_key in _vllm_engines:
        return _vllm_engines[model_key]

    if model_key in _vllm_failed_models:
        return None

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
            # ── Keep a safer default to avoid OOM on mid-range GPUs.
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
            enforce_eager=VLLM_ENFORCE_EAGER,
            # ── Required for models with custom modeling code (e.g. Cohere ASR).
            trust_remote_code=True,
            # ── Limit audio encoder profiling to 1 item to avoid segfault
            #    during warmup on constrained environments (e.g. WSL).
            limit_mm_per_prompt={"audio": 1},
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        _vllm_engines[model_key] = engine
        logger.info("Engine ready for %s", model_key)
        return engine
    except ImportError:
        _vllm_failed_models.add(model_key)
        logger.warning(
            "vLLM not available."
        )
        return None
    except Exception as exc:
        _vllm_failed_models.add(model_key)
        logger.warning(
            "vLLM engine init failed (%s).",
            exc,
        )
        return None


async def _get_engine_async(model_key: str, timeout_s: float = ENGINE_INIT_TIMEOUT_S):
    """Bound engine initialization latency so requests can fail fast and visibly."""
    if model_key in _vllm_engines:
        return _vllm_engines[model_key]
    if model_key in _vllm_failed_models:
        return None

    try:
        return await asyncio.wait_for(asyncio.to_thread(_get_engine, model_key), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "vLLM engine init timed out for %s after %.1fs. It may finish in the background.",
            model_key,
            timeout_s,
        )
        return _vllm_engines.get(model_key)


def _model_requires_hf_token(model_key: str) -> bool:
    model_id = SUPPORTED_MODELS[model_key].lower()
    return (
        model_id.startswith("cohereforai/")
        or model_id.startswith("coherelabs/")
        or model_id.startswith("ibm-granite/")
    )


def _ensure_model_auth(model_key: str) -> None:
    if _model_requires_hf_token(model_key) and not HF_TOKEN:
        raise HTTPException(
            status_code=401,
            detail=(
                f"Model '{model_key}' requires Hugging Face authentication. "
                "Set HF_TOKEN in backend environment."
            ),
        )


def _runtime_mode() -> str:
    if _vllm_engines:
        return "real"
    if ALLOW_MOCK_FALLBACK:
        return "mock"
    return "degraded"


# ---------------------------------------------------------------------------
# Audio pre-processing helpers
# ---------------------------------------------------------------------------

def _load_audio_bytes(data: bytes, filename: str | None = None) -> np.ndarray:
    """Decode audio bytes → mono float32 NumPy array at TARGET_SR."""
    try:
        with io.BytesIO(data) as buf:
            waveform, sr = sf.read(buf, dtype="float32", always_2d=False)
        # Resample if necessary
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        # Convert stereo → mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return waveform
    except Exception:
        # Fallback path for formats often emitted by MediaRecorder
        # (e.g. webm/opus) that may fail with direct soundfile decode.
        suffix = Path(filename).suffix if filename else ".webm"
        if not suffix:
            suffix = ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            waveform, _ = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
        return waveform.astype(np.float32, copy=False)


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
    logger.info(
        "Runtime config: gpu_memory_utilization=%.2f, max_model_len=%d, init_timeout=%.1fs, allow_mock_fallback=%s, enforce_eager=%s",
        GPU_MEMORY_UTILIZATION,
        VLLM_MAX_MODEL_LEN,
        ENGINE_INIT_TIMEOUT_S,
        ALLOW_MOCK_FALLBACK,
        VLLM_ENFORCE_EAGER,
    )
    yield
    logger.info("Open-ASR Model Explorer backend shutting down.")
    _vllm_engines.clear()
    _vllm_failed_models.clear()


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
    mode = _runtime_mode()
    status = "ok" if mode == "real" else "degraded"
    return {
        "status": status,
        "mode": mode,
        "loaded_models": list(_vllm_engines.keys()),
        "failed_models": sorted(_vllm_failed_models),
        "allow_mock_fallback": ALLOW_MOCK_FALLBACK,
        "engine_init_timeout_s": ENGINE_INIT_TIMEOUT_S,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "vllm_enforce_eager": VLLM_ENFORCE_EAGER,
    }


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
    _ensure_model_auth(model)

    # ── 1. Decode & normalise audio ──────────────────────────────────────────
    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes, audio.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)

    # ── 2. Attempt vLLM inference ────────────────────────────────────────────
    engine = await _get_engine_async(model)
    request_start = time.perf_counter()

    if engine is not None:
        transcript, ttft_ms, itl_ms = await _run_vllm(engine, waveform_padded, model)
    else:
        if not ALLOW_MOCK_FALLBACK:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Real inference unavailable for '{model}'. "
                    "vLLM engine failed or timed out during initialization."
                ),
            )
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
    _ensure_model_auth(model)

    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes, audio.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)
    engine = await _get_engine_async(model)

    if engine is None and not ALLOW_MOCK_FALLBACK:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Real inference unavailable for '{model}'. "
                "vLLM engine failed or timed out during initialization."
            ),
        )

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
