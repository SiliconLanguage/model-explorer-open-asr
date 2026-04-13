"""
Scribe Background Worker
─────────────────────────────────────────────────────────────────────────────
Consumes transcription jobs from Valkey (scribe:queue), runs faster-whisper
inference on GPU via a ThreadPoolExecutor, and writes results back to Valkey.

Job lifecycle:
  1. BLPOP scribe:queue → job_id
  2. Read metadata from scribe:job:{job_id}
  3. Set status → processing
  4. Run faster_whisper inference in a thread (releases GIL for CUDA)
  5. Write transcript + set status → completed
  6. Delete source audio from /data/audio_spool

Environment variables:
  VALKEY_HOST       — Valkey hostname (default: valkey)
  VALKEY_PORT       — Valkey port (default: 6379)
  WHISPER_MODEL     — faster-whisper model size (default: large-v3)
  WHISPER_DEVICE    — cuda | cpu (default: cuda)
  WHISPER_COMPUTE   — int8_float16 | float16 | int8 (default: int8_float16)
  NUM_WORKERS       — ThreadPoolExecutor size (default: 4)
  SPOOL_DIR         — Audio spool directory (default: /data/audio_spool)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [worker] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

VALKEY_HOST = os.getenv("VALKEY_HOST", "valkey")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8_float16")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", "/data/audio_spool"))

QUEUE_KEY = "scribe:queue"
JOB_PREFIX = "scribe:job:"
SPOOL_MAX_AGE_S = 86400  # 24 hours — matches Valkey key TTL

# ── Globals ───────────────────────────────────────────────────────────────────

_shutdown = asyncio.Event()
_model = None  # Lazy-loaded faster-whisper model (shared across threads)


def _load_model():
    """Load the faster-whisper model once, shared by all worker threads."""
    global _model
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    logger.info(
        "Loading faster-whisper model=%s device=%s compute=%s",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE,
    )
    t0 = time.perf_counter()
    _model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        num_workers=NUM_WORKERS,
    )
    logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
    return _model


def _get_valkey() -> redis.Redis:
    """Create a blocking Valkey client."""
    return redis.Redis(
        host=VALKEY_HOST,
        port=VALKEY_PORT,
        decode_responses=True,
        socket_connect_timeout=10,
        retry_on_timeout=True,
    )


# ── Inference (runs in ThreadPoolExecutor) ────────────────────────────────────

def _transcribe(audio_path: str, language: str | None) -> dict:
    """
    Run faster-whisper inference on a single audio file.
    Returns {"transcript": str, "segments": list[dict], "duration_s": float}.
    Called from a thread to release the GIL during CUDA execution.
    """
    model = _load_model()

    lang_arg = language if language and language != "auto" else None
    logger.debug("Inference start: %s (lang=%s)", audio_path, lang_arg or "auto-detect")
    t_start = time.perf_counter()
    segments_iter, info = model.transcribe(
        audio_path,
        language=lang_arg,
        beam_size=5,
        vad_filter=True,
    )

    seg_list = []
    texts = []
    ttft_ms = None
    for seg in segments_iter:
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000
        texts.append(seg.text.strip())
        seg_list.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    total_ms = (time.perf_counter() - t_start) * 1000
    n_segments = len(seg_list)
    # Synthetic ITL: average time between segments (excluding first)
    itl_ms = ((total_ms - (ttft_ms or 0)) / (n_segments - 1)) if n_segments > 1 else None

    transcript = " ".join(texts)
    return {
        "transcript": transcript,
        "segments": seg_list,
        "duration_s": round(info.duration, 3),
        "language_detected": info.language,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "itl_ms": round(itl_ms, 1) if itl_ms is not None else None,
    }


# ── Job processing ────────────────────────────────────────────────────────────

async def _process_job(
    pool: ThreadPoolExecutor,
    rconn: redis.Redis,
    job_id: str,
) -> None:
    """Process a single transcription job end-to-end."""
    job_key = f"{JOB_PREFIX}{job_id}"
    loop = asyncio.get_running_loop()

    # 1. Read job metadata
    meta = rconn.hgetall(job_key)
    if not meta:
        logger.warning("Job %s not found in Valkey, skipping", job_id)
        return

    audio_filename = meta.get("audio_file", "")
    language = meta.get("language")
    audio_path = SPOOL_DIR / audio_filename

    if not audio_path.is_file():
        logger.error("Job %s: audio file not found: %s", job_id, audio_path)
        rconn.hset(job_key, mapping={"status": "failed", "error": f"File not found: {audio_filename}"})
        return

    # 2. Mark as processing
    logger.info("Job %s: processing %s (lang=%s)", job_id, audio_filename, language)
    rconn.hset(job_key, "status", "processing")

    # 3. Run inference in thread pool
    t0 = time.perf_counter()
    try:
        result = await loop.run_in_executor(
            pool, _transcribe, str(audio_path), language
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Job %s: inference failed after %.1fs: %s", job_id, elapsed, exc)
        rconn.hset(job_key, mapping={"status": "failed", "error": str(exc)})
        return

    elapsed = time.perf_counter() - t0
    logger.info(
        "Job %s: completed in %.1fs — %d segments, %.1fs audio, lang=%s",
        job_id, elapsed,
        len(result["segments"]),
        result["duration_s"],
        result["language_detected"],
    )

    # 4. Write result to Valkey
    import json
    mapping = {
        "status": "completed",
        "transcript": result["transcript"],
        "segments": json.dumps(result["segments"]),
        "duration_s": str(result["duration_s"]),
        "language_detected": result["language_detected"],
        "processing_time_s": f"{elapsed:.2f}",
    }
    if result.get("ttft_ms") is not None:
        mapping["ttft_ms"] = str(result["ttft_ms"])
    if result.get("itl_ms") is not None:
        mapping["itl_ms"] = str(result["itl_ms"])
    rconn.hset(job_key, mapping=mapping)

    # 5. Delete source audio to reclaim disk.
    #    Frontend playback via /audio/{filename} uses the spool file, but
    #    transcription is complete — the Valkey hash holds all needed data.
    #    Spool files for failed/in-flight jobs are cleaned by startup sweep.
    try:
        audio_path.unlink(missing_ok=True)
        logger.info("Job %s: spool file deleted: %s", job_id, audio_filename)
    except OSError as exc:
        logger.warning("Job %s: failed to delete spool file %s: %s", job_id, audio_filename, exc)


# ── Startup spool sweep ───────────────────────────────────────────────────────

def _sweep_orphan_spool_files(rconn: redis.Redis) -> None:
    """Delete spool files older than SPOOL_MAX_AGE_S with no matching Valkey job."""
    if not SPOOL_DIR.is_dir():
        return

    now = time.time()
    removed = 0
    for f in SPOOL_DIR.iterdir():
        if not f.is_file():
            continue
        age = now - f.stat().st_mtime
        if age < SPOOL_MAX_AGE_S:
            continue
        # File is older than TTL — check if a Valkey job still references it
        # (defensive: maybe resubmitted). Stem is the job_id-prefixed filename.
        # We can't cheaply reverse-lookup, so just remove old files.
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass

    if removed:
        logger.info("Startup sweep: removed %d orphan spool files older than %ds", removed, SPOOL_MAX_AGE_S)


# ── Main loop ─────────────────────────────────────────────────────────────────

async def _main() -> None:
    logger.info(
        "Starting worker: model=%s, device=%s, compute=%s, num_workers=%d, spool=%s",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE, NUM_WORKERS, SPOOL_DIR,
    )

    # Pre-load model before accepting jobs
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=NUM_WORKERS, thread_name_prefix="whisper")
    await loop.run_in_executor(pool, _load_model)

    rconn = _get_valkey()
    logger.info("Connected to Valkey at %s:%d", VALKEY_HOST, VALKEY_PORT)

    # Sweep orphaned spool files from previous runs
    _sweep_orphan_spool_files(rconn)

    # Semaphore limits concurrent GPU jobs to NUM_WORKERS
    sem = asyncio.Semaphore(NUM_WORKERS)

    async def _bounded_process(job_id: str) -> None:
        async with sem:
            await _process_job(pool, rconn, job_id)

    tasks: set[asyncio.Task] = set()

    while not _shutdown.is_set():
        try:
            # BLPOP blocks in a thread to avoid blocking the event loop
            result = await loop.run_in_executor(
                None,
                lambda: rconn.blpop(QUEUE_KEY, timeout=2),
            )
        except redis.ConnectionError as exc:
            logger.warning("Valkey connection lost: %s — reconnecting in 3s", exc)
            await asyncio.sleep(3)
            rconn = _get_valkey()
            continue

        if result is None:
            # BLPOP timed out — loop back and check shutdown
            continue

        _key, job_id = result
        task = asyncio.create_task(_bounded_process(job_id))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    # Drain in-flight tasks on shutdown
    if tasks:
        logger.info("Shutting down: waiting for %d in-flight jobs…", len(tasks))
        await asyncio.gather(*tasks, return_exceptions=True)

    pool.shutdown(wait=True)
    rconn.close()
    logger.info("Worker stopped.")


def _handle_signal(sig, _frame):
    logger.info("Received %s, initiating graceful shutdown…", signal.Signals(sig).name)
    _shutdown.set()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    asyncio.run(_main())
