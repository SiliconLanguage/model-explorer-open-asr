"""
Scribe v1 — Python Inference Worker (gRPC Server)

Implements the ScribeEngine gRPC service backed by faster-whisper.
Handles both FastTranscribe (synchronous streaming) and SubmitBatchJob
(queued via Valkey).

Environment variables:
  GRPC_PORT           — gRPC listen port (default: 50051)
  DEFAULT_MODEL       — Default faster-whisper model size (default: large-v3)
  DEVICE              — "cuda" or "cpu" (default: auto-detect)
  COMPUTE_TYPE        — "float16", "int8", etc. (default: auto)
  VALKEY_ENDPOINT     — Valkey address for batch results (default: valkey:6379)
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from concurrent import futures

import grpc
import numpy as np
import soundfile as sf

import scribe_pb2
import scribe_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("scribe-worker")

# ── Configuration ────────────────────────────────────────────────────────────

GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "auto")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto")
TARGET_SR = 16_000

# ── Model cache ──────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}


def _detect_device() -> tuple[str, str]:
    """Auto-detect CUDA availability and return (device, compute_type)."""
    if DEVICE != "auto":
        ct = COMPUTE_TYPE if COMPUTE_TYPE != "auto" else ("float16" if DEVICE == "cuda" else "int8")
        return DEVICE, ct
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
    except ImportError:
        pass
    return "cpu", "int8" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE


def get_model(model_name: str):
    """Lazily load a faster-whisper model, caching by name."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from faster_whisper import WhisperModel

    device, compute_type = _detect_device()
    logger.info("Loading faster-whisper '%s' on %s (%s)…", model_name, device, compute_type)
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    elapsed = time.perf_counter() - t0
    logger.info("Model '%s' ready in %.1fs", model_name, elapsed)
    _model_cache[model_name] = model
    return model


def transcribe_audio(
    waveform: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
) -> tuple[str, list[dict], float]:
    """
    Run faster-whisper inference.
    Returns (full_text, segments_list, duration_sec).
    """
    model = get_model(model_name)
    duration_sec = len(waveform) / TARGET_SR

    lang = language if language and language != "auto" else None
    segments_iter, info = model.transcribe(
        waveform,
        language=lang,
        beam_size=5,
        vad_filter=True,
    )

    segments = []
    texts = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": "",
            "confidence": seg.avg_logprob,
        })
        texts.append(seg.text.strip())

    full_text = " ".join(texts)
    return full_text, segments, duration_sec


# ── gRPC Service ─────────────────────────────────────────────────────────────

class ScribeEngineServicer(scribe_pb2_grpc.ScribeEngineServicer):
    """Implements the ScribeEngine gRPC service."""

    def FastTranscribe(self, request_iterator, context):
        """
        Client-to-server streaming RPC.
        Accumulates audio chunks, runs inference, returns a single response.
        """
        request_id = str(uuid.uuid4())
        audio_buffer = bytearray()
        model_name = DEFAULT_MODEL
        language = ""

        for chunk in request_iterator:
            audio_buffer.extend(chunk.data)
            if chunk.model:
                model_name = chunk.model
            if chunk.language:
                language = chunk.language

        logger.info("[%s] FastTranscribe: %d bytes, model=%s, lang=%s",
                    request_id, len(audio_buffer), model_name, language)

        if not audio_buffer:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No audio data received")
            return scribe_pb2.TranscribeResponse()

        # Decode audio bytes to waveform
        try:
            waveform = _bytes_to_waveform(bytes(audio_buffer))
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Cannot decode audio: {e}")
            return scribe_pb2.TranscribeResponse()

        # Run inference
        t0 = time.perf_counter()
        try:
            full_text, segments, duration_sec = transcribe_audio(
                waveform, model_name, language or None
            )
        except Exception as e:
            logger.error("[%s] Inference failed: %s", request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference error: {e}")
            return scribe_pb2.TranscribeResponse()

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Transcription complete: %.1fs, %d segments, %.1f RTFx",
                    request_id, elapsed, len(segments),
                    duration_sec / elapsed if elapsed > 0 else 0)

        # Build response
        proto_segments = []
        for seg in segments:
            proto_segments.append(scribe_pb2.Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg["speaker"],
                confidence=seg["confidence"],
            ))

        return scribe_pb2.TranscribeResponse(
            request_id=request_id,
            duration_sec=duration_sec,
            result=scribe_pb2.TranscriptResult(
                text_display=full_text,
                segments=proto_segments,
            ),
        )

    def SubmitBatchJob(self, request, context):
        """
        Standard RPC — enqueue a batch job.
        The actual processing is handled by batch_consumer.py.
        """
        job_id = str(uuid.uuid4())
        logger.info("[%s] Batch job submitted: model=%s uri=%s",
                    job_id, request.model, request.input_uri)

        # Store in Valkey via the batch consumer's shared connection
        try:
            import redis
            r = redis.Redis.from_url(
                f"redis://{os.getenv('VALKEY_ENDPOINT', 'valkey:6379')}"
            )
            r.xadd("scribe_jobs", {
                "job_id": job_id,
                "input_uri": request.input_uri,
                "model": request.model or DEFAULT_MODEL,
                "language": request.language or "",
                "callback_url": request.callback_url or "",
                "status": "QUEUED",
            })
            r.hset(f"scribe:job:{job_id}", mapping={
                "status": "QUEUED",
                "input_uri": request.input_uri,
                "model": request.model or DEFAULT_MODEL,
            })
        except Exception as e:
            logger.warning("[%s] Valkey enqueue failed: %s", job_id, e)

        return scribe_pb2.BatchJobResponse(
            job_id=job_id,
            status="QUEUED",
        )


def _bytes_to_waveform(data: bytes) -> np.ndarray:
    """Decode audio bytes (WAV, FLAC, OGG, etc.) to 16kHz mono float32."""
    try:
        waveform, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        # Fallback: try treating as raw 16-bit PCM at 16kHz
        waveform = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        sr = TARGET_SR

    # Mix to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    return waveform


# ── Server entrypoint ────────────────────────────────────────────────────────

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    scribe_pb2_grpc.add_ScribeEngineServicer_to_server(
        ScribeEngineServicer(), server
    )
    addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(addr)
    logger.info("Scribe Worker gRPC server starting on %s", addr)

    # Pre-warm default model
    try:
        get_model(DEFAULT_MODEL)
    except Exception as e:
        logger.warning("Pre-warm failed for '%s': %s", DEFAULT_MODEL, e)

    server.start()
    logger.info("Scribe Worker ready — waiting for requests")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
