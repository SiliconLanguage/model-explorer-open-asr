"""
Open-ASR Model Explorer – FastAPI backend
Handles server-side (vLLM) transcription requests.
Strategy-based routing:
  - Models with "WebGPU" in the name → handled client-side (transformers.js)
  - All other models → routed here via POST /transcribe
"""

from __future__ import annotations

import gc
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
from transformers import pipeline

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import gradio as gr

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
    "CohereLabs/cohere-transcribe-03-2026": "CohereLabs/cohere-transcribe-03-2026",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
    "ibm-granite/granite-4.0-1b-speech": "ibm-granite/granite-4.0-1b-speech",
    "openai/whisper-base": "openai/whisper-base",
}

# Target sample rate expected by Whisper-style models
TARGET_SR = 16_000
# Pad / truncate all audio to this many samples (30 s at 16 kHz)
MAX_AUDIO_SAMPLES = TARGET_SR * 30

# Language name → ISO-639-1 code (used by Cohere .transcribe())
LANGUAGE_ISO_MAP: dict[str, str] = {
    "english": "en",
    "chinese": "zh",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "hindi": "hi",
}

# ---------------------------------------------------------------------------
# Singleton Model Manager – at most ONE model lives in VRAM at a time
# ---------------------------------------------------------------------------


class ModelManager:
    """Aggressive VRAM GC – evicts the resident model before loading a new one."""

    def __init__(self) -> None:
        self.current_model_id: str | None = None
        self.current_engine_mode: str | None = None
        self._engine: object | None = None  # vLLM engine OR HF pipeline/wrapper
        self._engine_type: str | None = None  # "vllm" | "hf"
        self._failed: set[str] = set()

    # -- cache key ----------------------------------------------------------

    @staticmethod
    def _cache_key(model_key: str, engine_mode: str) -> str:
        return f"{model_key}::{engine_mode}"

    # -- VRAM purge ---------------------------------------------------------

    def purge(self) -> None:
        """Delete the resident model and reclaim all cached VRAM."""
        if self._engine is not None:
            logger.info(
                "VRAM purge: evicting %s (%s)…",
                self.current_model_id,
                self.current_engine_mode,
            )
            del self._engine
            self._engine = None

        self.current_model_id = None
        self.current_engine_mode = None
        self._engine_type = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("VRAM purge complete.")

    # -- swap check ---------------------------------------------------------

    def _needs_swap(self, model_key: str, engine_mode: str) -> bool:
        return self.current_model_id is not None and (
            self.current_model_id != model_key
            or self.current_engine_mode != engine_mode
        )

    # -- public accessors ---------------------------------------------------

    def is_loaded(self) -> bool:
        return self._engine is not None

    def loaded_info(self) -> dict:
        return {
            "current_model_id": self.current_model_id,
            "current_engine_mode": self.current_engine_mode,
            "failed": sorted(self._failed),
        }

    # -- clear (shutdown) ---------------------------------------------------

    def clear(self) -> None:
        self.purge()
        self._failed.clear()


_model_manager = ModelManager()


class _CohereASRWrapper:
    """Thin wrapper so Cohere's custom model/processor pair can live in ModelManager."""
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model


class _Qwen3ASRWrapper:
    """Thin wrapper so qwen-asr's Qwen3ASRModel can live in ModelManager."""
    def __init__(self, model):
        self.model = model


class _GraniteSpeechWrapper:
    """Thin wrapper for IBM Granite Speech (chat-template + <|audio|> prompt)."""
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        self.tokenizer = processor.tokenizer


# Language name → display name expected by Qwen3-ASR .transcribe(language=...)
QWEN_LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


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
GPU_MEMORY_UTILIZATION = _read_float_env("GPU_MEMORY_UTILIZATION", 0.4)
VLLM_MAX_MODEL_LEN = _read_int_env("OPENASR_VLLM_MAX_MODEL_LEN", 8192)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
VLLM_ENFORCE_EAGER = _read_bool_env("OPENASR_VLLM_ENFORCE_EAGER", True)
ENGINE_CHOICES = {"vllm", "hf-gpu", "hf-cpu"}


def _normalize_engine(engine: str | None) -> str:
    if engine is None:
        return "vllm"
    value = engine.strip().lower()
    if value not in ENGINE_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid engine '{engine}'. Expected one of {sorted(ENGINE_CHOICES)}.",
        )
    return value


def _build_granite_speech_model(hf_model: str, engine_mode: str) -> _GraniteSpeechWrapper:
    """Load IBM Granite Speech via AutoProcessor + AutoModelForSpeechSeq2Seq."""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True)
    if engine_mode == "hf-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("hf-gpu requested but CUDA is not available.")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model, device_map="cuda:0", torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif engine_mode == "hf-cpu":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model, trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported engine mode '{engine_mode}' for Granite Speech.")
    model.eval()
    return _GraniteSpeechWrapper(processor, model)


def _build_qwen3_asr_model(hf_model: str, engine_mode: str) -> _Qwen3ASRWrapper:
    """Load Qwen3-ASR via the qwen-asr package (handles its own architecture registration)."""
    from qwen_asr import Qwen3ASRModel

    kwargs: dict = {
        "max_new_tokens": 512,
        "max_inference_batch_size": 1,
    }
    if engine_mode == "hf-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("hf-gpu requested but CUDA is not available.")
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "cuda:0"
    elif engine_mode == "hf-cpu":
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = "cpu"
    else:
        raise ValueError(f"Unsupported engine mode '{engine_mode}' for Qwen3-ASR.")

    model = Qwen3ASRModel.from_pretrained(hf_model, **kwargs)
    return _Qwen3ASRWrapper(model)


def _build_cohere_model(hf_model: str, engine_mode: str) -> _CohereASRWrapper:
    """Load Cohere ASR via AutoProcessor + AutoModelForSpeechSeq2Seq (trust_remote_code)."""
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True)
    if engine_mode == "hf-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("hf-gpu requested but CUDA is not available.")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to("cuda:0")
    elif engine_mode == "hf-cpu":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model, trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported engine mode '{engine_mode}' for Cohere.")
    model.eval()
    return _CohereASRWrapper(processor, model)


def _build_hf_pipeline(model_key: str, engine_mode: str):
    hf_model = SUPPORTED_MODELS[model_key]

    # Qwen3-ASR custom path — uses qwen-asr package
    if "qwen3-asr" in hf_model.lower() or "qwen/qwen3-asr" in hf_model.lower():
        return _build_qwen3_asr_model(hf_model, engine_mode)

    # IBM Granite Speech custom path — chat-template + <|audio|> prompt
    if "granite" in hf_model.lower() and "speech" in hf_model.lower():
        return _build_granite_speech_model(hf_model, engine_mode)

    # Cohere custom path — uses .transcribe() instead of generic pipeline()
    if "cohere-transcribe" in hf_model.lower():
        return _build_cohere_model(hf_model, engine_mode)

    if engine_mode == "hf-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("hf-gpu requested but CUDA is not available.")
        return pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            trust_remote_code=True,
            device=0,
            torch_dtype=torch.float16,
        )
    if engine_mode == "hf-cpu":
        return pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            trust_remote_code=True,
            device="cpu",
        )
    raise ValueError(f"Unsupported HF engine mode '{engine_mode}'.")


def _get_hf_pipeline(model_key: str, engine_mode: str):
    cache_key = _model_manager._cache_key(model_key, engine_mode)
    if _model_manager.current_model_id == model_key and _model_manager.current_engine_mode == engine_mode:
        return _model_manager._engine
    if cache_key in _model_manager._failed:
        return None

    # Different model is resident – purge VRAM first
    if _model_manager._needs_swap(model_key, engine_mode):
        _model_manager.purge()

    try:
        logger.info("Loading transformers ASR pipeline for %s (%s)...", model_key, engine_mode)
        asr_pipeline = _build_hf_pipeline(model_key, engine_mode)
        _model_manager._engine = asr_pipeline
        _model_manager.current_model_id = model_key
        _model_manager.current_engine_mode = engine_mode
        _model_manager._engine_type = "hf"
        logger.info("Transformers pipeline ready for %s (%s)", model_key, engine_mode)
        return asr_pipeline
    except Exception as exc:
        _model_manager._failed.add(cache_key)
        logger.warning("Transformers pipeline init failed for %s (%s): %s", model_key, engine_mode, exc)
        return None


async def _get_hf_pipeline_async(model_key: str, engine_mode: str, timeout_s: float = ENGINE_INIT_TIMEOUT_S):
    cache_key = _model_manager._cache_key(model_key, engine_mode)
    if _model_manager.current_model_id == model_key and _model_manager.current_engine_mode == engine_mode:
        return _model_manager._engine
    if cache_key in _model_manager._failed:
        return None

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_get_hf_pipeline, model_key, engine_mode),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Transformers pipeline init timed out for %s (%s) after %.1fs.",
            model_key,
            engine_mode,
            timeout_s,
        )
        if _model_manager.current_model_id == model_key and _model_manager.current_engine_mode == engine_mode:
            return _model_manager._engine
        return None


async def _get_best_hf_pipeline_async(model_key: str):
    pipeline_gpu = await _get_hf_pipeline_async(model_key, "hf-gpu")
    if pipeline_gpu is not None:
        return pipeline_gpu, "hf-gpu"
    pipeline_cpu = await _get_hf_pipeline_async(model_key, "hf-cpu")
    if pipeline_cpu is not None:
        return pipeline_cpu, "hf-cpu"
    return None, None


def _get_engine(model_key: str):
    """Return (or lazily create) the vLLM AsyncLLMEngine for *model_key*."""
    cache_key = _model_manager._cache_key(model_key, "vllm")
    if _model_manager.current_model_id == model_key and _model_manager._engine_type == "vllm":
        return _model_manager._engine

    if cache_key in _model_manager._failed:
        return None

    # Different model is resident – purge VRAM first
    if _model_manager._needs_swap(model_key, "vllm"):
        _model_manager.purge()

    try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore

        hf_model = SUPPORTED_MODELS[model_key]
        logger.info("Loading vLLM engine for %s (%s)…", model_key, hf_model)

        engine_args = AsyncEngineArgs(
            model=hf_model,
            # ── Chunked prefill prevents long audio prefills from starving
            #    decode steps of other in-flight requests.
            enable_chunked_prefill=True,
            max_num_batched_tokens=512,
            # ── Keep a safer default to avoid OOM on mid-range GPUs.
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            # ── Let vLLM auto-derive max_model_len from the model config
            #    so we don't exceed max_position_embeddings for any model.
            enforce_eager=VLLM_ENFORCE_EAGER,
            # ── Required for models with custom modeling code (e.g. Cohere ASR).
            trust_remote_code=True,
            # ── Limit audio encoder profiling to 1 item to reduce profiling
            #    burden on constrained environments.
            limit_mm_per_prompt={"audio": 1},
            # ── Skip encoder cache profiling — the forward pass during
            #    profiling triggers segfaults on SM 12.0.
            skip_mm_profiling=True,
            # ── Force FlashAttention 2 backend.  FA2 ships with sm_80 PTX
            #    that the CUDA driver JIT-compiles for supported GPUs.
            #    NOTE: All attention backends still segfault on SM 12.0
            #    (Blackwell) due to Triton 3.6.0 ir.builder bug.  This
            #    config is for non-Blackwell cloud deployments.
            attention_backend="FLASH_ATTN",
            # ── Disable Triton-compiled custom fused kernels (norm_quant,
            #    act_quant) that segfault on SM 12.0 via Triton 3.6.0 bug.
            compilation_config={"custom_ops": ["none"]},
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        _model_manager._engine = engine
        _model_manager.current_model_id = model_key
        _model_manager.current_engine_mode = "vllm"
        _model_manager._engine_type = "vllm"
        logger.info("Engine ready for %s", model_key)
        return engine
    except ImportError:
        _model_manager._failed.add(_model_manager._cache_key(model_key, "vllm"))
        logger.warning(
            "vLLM not available."
        )
        return None
    except Exception as exc:
        _model_manager._failed.add(_model_manager._cache_key(model_key, "vllm"))
        logger.warning(
            "vLLM engine init failed (%s).",
            exc,
        )
        return None


async def _get_engine_async(model_key: str, timeout_s: float = ENGINE_INIT_TIMEOUT_S):
    """Bound engine initialization latency so requests can fail fast and visibly."""
    cache_key = _model_manager._cache_key(model_key, "vllm")
    if _model_manager.current_model_id == model_key and _model_manager._engine_type == "vllm":
        return _model_manager._engine
    if cache_key in _model_manager._failed:
        return None

    try:
        return await asyncio.wait_for(asyncio.to_thread(_get_engine, model_key), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "vLLM engine init timed out for %s after %.1fs. It may finish in the background.",
            model_key,
            timeout_s,
        )
        if _model_manager.current_model_id == model_key and _model_manager._engine_type == "vllm":
            return _model_manager._engine
        return None


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
    if _model_manager.is_loaded():
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
    _model_manager.clear()


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
# Gradio UI – mounted at / for HF Spaces iframe
# ---------------------------------------------------------------------------

def _gradio_transcribe(audio_path: str, model_name: str, engine: str, language: str):
    """Synchronous wrapper called by Gradio – posts to the local FastAPI endpoint."""
    import httpx

    if not audio_path:
        return "Please upload or record an audio file."

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    port = int(os.environ.get("PORT", "7860"))
    resp = httpx.post(
        f"http://127.0.0.1:{port}/transcribe",
        files={"audio": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": model_name, "engine": engine, "language": language},
        timeout=120.0,
    )
    if resp.status_code != 200:
        return f"Error {resp.status_code}: {resp.text}"
    result = resp.json()
    metrics = (
        f"\n\n---\nTTFT: {result.get('ttft_ms', 'N/A')} ms  |  "
        f"ITL: {result.get('itl_ms', 'N/A')} ms  |  "
        f"RTFx: {result.get('rtfx', 'N/A')}"
    )
    return result.get("transcript", "") + metrics


_SERVER_MODELS = [k for k in SUPPORTED_MODELS.keys() if "/" not in k or k.startswith("ibm-") or k.startswith("openai/")]

_SAMPLES_DIR = Path("/app/samples") if Path("/app/samples").exists() else Path("samples")

demo = gr.Interface(
    fn=_gradio_transcribe,
    inputs=[
        gr.Audio(type="filepath", label="Upload or record audio"),
        gr.Dropdown(
            choices=list(SUPPORTED_MODELS.keys()),
            value="openai/whisper-base",
            label="Model",
        ),
        gr.Dropdown(
            choices=["hf-gpu", "hf-cpu", "vllm"],
            value="hf-gpu",
            label="Engine",
        ),
        gr.Dropdown(
            choices=["english", "chinese", "spanish", "french", "german",
                     "japanese", "korean", "hindi", "arabic", "portuguese"],
            value="english",
            label="Language",
        ),
    ],
    outputs=gr.Textbox(label="Transcript", lines=8),
    title="Open-ASR Model Explorer",
    description=(
        "Hybrid inference testbed for open-source ASR models. "
        "Upload audio or record from your microphone, pick a model and engine, then transcribe.\n\n"
        "**Engine options:** `hf-gpu` (HuggingFace Transformers on GPU), "
        "`hf-cpu` (CPU-only fallback), `vllm` (vLLM optimised serving).\n\n"
        "**Note:** WebGPU client-side models (Xenova/whisper-*, onnx-community/cohere-*) "
        "run in-browser via transformers.js and are available in the "
        "[full React UI](https://github.com/SiliconLanguage/model-explorer-open-asr) only."
    ),
    examples=[
        [str(_SAMPLES_DIR / "english.wav"), "openai/whisper-base", "hf-gpu", "english"],
        [str(_SAMPLES_DIR / "chinese.wav"), "openai/whisper-base", "hf-gpu", "chinese"],
        [str(_SAMPLES_DIR / "french.wav"), "openai/whisper-base", "hf-gpu", "french"],
        [str(_SAMPLES_DIR / "spanish.wav"), "openai/whisper-base", "hf-gpu", "spanish"],
        [str(_SAMPLES_DIR / "japanese.wav"), "openai/whisper-base", "hf-gpu", "japanese"],
        [str(_SAMPLES_DIR / "hindi.wav"), "openai/whisper-base", "hf-gpu", "hindi"],
    ],
    flagging_mode="never",
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    mode = _runtime_mode()
    status = "ok" if mode == "real" else "degraded"
    mgr_info = _model_manager.loaded_info()
    return {
        "status": status,
        "mode": mode,
        "current_model_id": mgr_info["current_model_id"],
        "current_engine_mode": mgr_info["current_engine_mode"],
        "failed_models": mgr_info["failed"],
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
    engine: str | None = Form(None, description="Execution engine: vllm | hf-gpu | hf-cpu"),
    language: str = Form("english", description="Language name, e.g. 'english', 'chinese', 'spanish'"),
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
    selected_engine = _normalize_engine(engine)
    iso_lang = LANGUAGE_ISO_MAP.get(language.strip().lower(), "en")

    # ── 1. Decode & normalise audio ──────────────────────────────────────────
    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes, audio.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)

    # ── 2. Run on the selected engine (no silent fallback) ────────────────────────
    vllm_engine = await _get_engine_async(model) if selected_engine == "vllm" else None
    request_start = time.perf_counter()

    if selected_engine == "vllm":
        if vllm_engine is not None:
            transcript, ttft_ms, itl_ms = await _run_vllm(vllm_engine, waveform_padded, model)
        else:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"vLLM engine failed to initialise for '{model}'. "
                    "On SM 12.0 (Blackwell) all attention backends segfault "
                    "due to Triton 3.6.0 bug. Select an HF-GPU variant or "
                    "deploy to a non-Blackwell cloud GPU."
                ),
            )
    else:
        hf_pipeline = await _get_hf_pipeline_async(model, selected_engine)
        if hf_pipeline is not None:
            transcript, ttft_ms, itl_ms = await _run_hf_pipeline(hf_pipeline, waveform_padded, iso_lang)
        else:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Transformers engine '{selected_engine}' failed to load for '{model}'. "
                    "Check backend logs for details."
                ),
            )

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

    Adapts the prompt and multimodal data format per model:
    - Granite Speech: chat-template with <|audio|> token + multi_modal_data
    - Qwen-style: <|audio_bos|><|AUDIO|><|audio_eos|> prompt
    """
    import asyncio
    from vllm import SamplingParams, TokensPrompt  # type: ignore

    hf_model = SUPPORTED_MODELS.get(model_key, model_key)
    is_granite = "granite" in hf_model.lower() and "speech" in hf_model.lower()

    if is_granite:
        # Granite Speech: chat-template prompt + multimodal audio data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        question = "can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": f"<|audio|>{question}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

        # vLLM AsyncEngine expects multimodal data alongside the prompt
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": (waveform, TARGET_SR)},
        }
    else:
        # Qwen / generic ASR prompt
        prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        inputs = {"prompt": prompt}

    request_id = f"asr-{time.time_ns()}"
    token_timestamps: list[float] = []
    first_token_time: float | None = None
    request_start = time.perf_counter()

    result_generator = engine.generate(inputs, sampling_params, request_id)
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


async def _run_granite_speech(wrapper: _GraniteSpeechWrapper, waveform: np.ndarray, language: str = "en") -> tuple[str, float, float]:
    """Run IBM Granite Speech via chat-template prompt + processor + generate."""
    request_start = time.perf_counter()

    def _run_sync():
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        prompt = wrapper.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        # Granite processor expects (text, audio_tensor, device=...)
        wav_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
        device = next(wrapper.model.parameters()).device
        model_inputs = wrapper.processor(
            prompt, wav_tensor, device=device, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model_outputs = wrapper.model.generate(
                **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1,
            )
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        text = wrapper.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True,
        )
        return text[0] if text else ""

    try:
        transcript = await asyncio.to_thread(_run_sync)
    except Exception as exc:
        logger.warning("Granite Speech inference failed: %s", exc)
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        return (
            "[HF fallback] Granite Speech model loaded, but inference failed. "
            "Try a different model.",
            elapsed_ms,
            0.0,
        )
    elapsed_ms = (time.perf_counter() - request_start) * 1000
    return str(transcript).strip(), elapsed_ms, 0.0


async def _run_qwen3_asr(wrapper: _Qwen3ASRWrapper, waveform: np.ndarray, language: str = "en") -> tuple[str, float, float]:
    """Run Qwen3-ASR via the qwen-asr package's .transcribe() method."""
    request_start = time.perf_counter()
    qwen_lang = QWEN_LANGUAGE_MAP.get(language)  # None → auto-detect

    def _run_sync():
        results = wrapper.model.transcribe(
            audio=[(waveform, TARGET_SR)],
            language=qwen_lang,
        )
        return results[0].text if results else ""

    try:
        transcript = await asyncio.to_thread(_run_sync)
    except Exception as exc:
        logger.warning("Qwen3-ASR inference failed: %s", exc)
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        return (
            "[HF fallback] Qwen3-ASR model loaded, but inference failed. "
            "Try a different model.",
            elapsed_ms,
            0.0,
        )
    elapsed_ms = (time.perf_counter() - request_start) * 1000
    return str(transcript).strip(), elapsed_ms, 0.0


async def _run_cohere(wrapper: _CohereASRWrapper, waveform: np.ndarray, language: str = "en") -> tuple[str, float, float]:
    """Run Cohere ASR via its custom .transcribe() method."""
    COHERE_SUPPORTED_LANGS = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}
    cohere_lang = language if language in COHERE_SUPPORTED_LANGS else "en"
    request_start = time.perf_counter()

    def _run_sync():
        texts = wrapper.model.transcribe(
            processor=wrapper.processor,
            audio_arrays=[waveform],
            sample_rates=[TARGET_SR],
            language=cohere_lang,
        )
        return texts[0] if texts else ""

    try:
        transcript = await asyncio.to_thread(_run_sync)
    except Exception as exc:
        logger.warning("Cohere inference failed: %s", exc)
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        return (
            "[HF fallback] Cohere model loaded, but inference failed. "
            "Try a different model.",
            elapsed_ms,
            0.0,
        )
    elapsed_ms = (time.perf_counter() - request_start) * 1000
    return str(transcript).strip(), elapsed_ms, 0.0


async def _run_hf_pipeline(asr_pipeline, waveform: np.ndarray, language: str = "en") -> tuple[str, float, float]:
    """Run transformers ASR and return (transcript, ttft_ms, mean_itl_ms)."""
    # Dispatch Qwen3-ASR models to their custom runner
    if isinstance(asr_pipeline, _Qwen3ASRWrapper):
        return await _run_qwen3_asr(asr_pipeline, waveform, language)

    # Dispatch IBM Granite Speech to their custom runner
    if isinstance(asr_pipeline, _GraniteSpeechWrapper):
        return await _run_granite_speech(asr_pipeline, waveform, language)

    # Dispatch Cohere models to their custom runner
    if isinstance(asr_pipeline, _CohereASRWrapper):
        return await _run_cohere(asr_pipeline, waveform, language)

    request_start = time.perf_counter()

    def _run_sync():
        try:
            gen_kwargs: dict = {"task": "transcribe"}
            if language:
                gen_kwargs["language"] = language
            return asr_pipeline(waveform, generate_kwargs=gen_kwargs)
        except Exception as exc:
            # Some custom ASR architectures (e.g. Cohere ASR) require explicit
            # generate() usage instead of the generic pipeline forward call.
            if "decoder_input_ids" not in str(exc) and "input_ids" not in str(exc):
                raise
            feature_extractor = getattr(asr_pipeline, "feature_extractor", None)
            tokenizer = getattr(asr_pipeline, "tokenizer", None)
            if feature_extractor is None or tokenizer is None:
                raise
            model = asr_pipeline.model
            inputs = feature_extractor(
                waveform,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            generation_inputs = dict(inputs)
            decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
            if decoder_start_token_id is not None and "decoder_input_ids" not in generation_inputs:
                first_tensor = next(iter(inputs.values()))
                batch_size = int(first_tensor.shape[0])
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    int(decoder_start_token_id),
                    dtype=torch.long,
                    device=model.device,
                )
                generation_inputs["decoder_input_ids"] = decoder_input_ids
                generation_inputs["decoder_attention_mask"] = torch.ones_like(decoder_input_ids)
            with torch.no_grad():
                generated_ids = model.generate(**generation_inputs)
            text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return {"text": text}

    try:
        result = await asyncio.to_thread(_run_sync)
    except Exception as exc:
        logger.warning("Transformers inference failed; returning graceful fallback transcript: %s", exc)
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        return (
            "[HF fallback] Model loaded, but inference failed for this audio/configuration. "
            "Try hf-cpu or a different model.",
            elapsed_ms,
            0.0,
        )
    elapsed_ms = (time.perf_counter() - request_start) * 1000
    transcript = str(result.get("text", "")).strip() if isinstance(result, dict) else str(result).strip()
    return transcript, elapsed_ms, 0.0


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
    engine: str | None = Form(None),
    language: str = Form("english"),
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
    selected_engine = _normalize_engine(engine)
    iso_lang = LANGUAGE_ISO_MAP.get(language.strip().lower(), "en")

    raw_bytes = await audio.read()
    try:
        waveform = _load_audio_bytes(raw_bytes, audio.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}") from exc

    audio_duration_s = len(waveform) / TARGET_SR
    waveform_padded = _pad_or_truncate(waveform)
    vllm_engine = await _get_engine_async(model) if selected_engine == "vllm" else None
    hf_pipeline = None
    if selected_engine in {"hf-gpu", "hf-cpu"}:
        hf_pipeline = await _get_hf_pipeline_async(model, selected_engine)

    if selected_engine == "vllm" and vllm_engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"vLLM engine failed to initialise for '{model}'. "
                "On SM 12.0 (Blackwell) all attention backends segfault "
                "due to Triton 3.6.0 bug. Select an HF-GPU variant or "
                "deploy to a non-Blackwell cloud GPU."
            ),
        )
    if selected_engine in {"hf-gpu", "hf-cpu"} and hf_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Transformers engine '{selected_engine}' failed to load for '{model}'. "
                "Check backend logs for details."
            ),
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        import json

        request_start = time.perf_counter()
        first_token_time: float | None = None
        token_timestamps: list[float] = []
        full_text = ""

        if vllm_engine is not None:
            from vllm import SamplingParams  # type: ignore

            hf_model = SUPPORTED_MODELS.get(model, model)
            is_granite = "granite" in hf_model.lower() and "speech" in hf_model.lower()

            if is_granite:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
                question = "can you transcribe the speech into a written format?"
                chat = [{"role": "user", "content": f"<|audio|>{question}"}]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
                sampling_params = SamplingParams(temperature=0.2, max_tokens=256)
                inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": (waveform_padded, TARGET_SR)},
                }
            else:
                prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the speech."
                sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
                inputs = {"prompt": prompt}

            request_id = f"asr-stream-{time.time_ns()}"

            async for output in vllm_engine.generate(inputs, sampling_params, request_id):
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
        elif hf_pipeline is not None:
            transcript, _, _ = await _run_hf_pipeline(hf_pipeline, waveform_padded, iso_lang)
            first_token_time = time.perf_counter()
            ttft_ms = (first_token_time - request_start) * 1000
            token_timestamps.append(first_token_time)
            yield f"data: {json.dumps({'token': '', 'ttft_ms': round(ttft_ms, 2), 'done': False})}\n\n"
            full_text = transcript
            token_timestamps.append(time.perf_counter())
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

        final_ttft_ms = (
            round((first_token_time - request_start) * 1000, 2)
            if first_token_time is not None
            else None
        )
        final_payload = json.dumps({
            "token": "",
            "ttft_ms": final_ttft_ms,
            "itl_ms": round(itl_ms, 2),
            "rtfx": round(rtfx, 4),
            "transcript": full_text.strip(),
            "done": True,
        })
        yield f"data: {final_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Mount Gradio UI AFTER all FastAPI routes so it doesn't shadow them
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(app, demo, path="/")
