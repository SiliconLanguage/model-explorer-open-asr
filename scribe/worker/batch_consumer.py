"""
Scribe v1 — Batch Consumer

Pulls jobs from the Valkey "scribe_jobs" stream, processes audio files
via faster-whisper, and writes results back to Valkey.

Runs as a separate process alongside scribe_server.py.

Environment variables:
  VALKEY_ENDPOINT  — Valkey address (default: valkey:6379)
  DEFAULT_MODEL    — Default model size (default: large-v3)
  CONSUMER_GROUP   — Consumer group name (default: scribe-workers)
  CONSUMER_NAME    — This consumer's name (default: worker-1)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import tempfile

import numpy as np
import redis
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("batch-consumer")

VALKEY_ENDPOINT = os.getenv("VALKEY_ENDPOINT", "valkey:6379")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "large-v3")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "scribe-workers")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "worker-1")
TARGET_SR = 16_000
STREAM_KEY = "scribe_jobs"


def get_valkey() -> redis.Redis:
    return redis.Redis.from_url(f"redis://{VALKEY_ENDPOINT}")


def ensure_consumer_group(r: redis.Redis):
    """Create the consumer group if it doesn't exist."""
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created consumer group '%s' on stream '%s'", CONSUMER_GROUP, STREAM_KEY)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group '%s' already exists", CONSUMER_GROUP)
        else:
            raise


def fetch_audio(uri: str) -> np.ndarray:
    """Download audio from a URI and decode to 16kHz mono float32."""
    logger.info("Fetching audio from %s", uri)

    with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
        if uri.startswith(("http://", "https://")):
            urllib.request.urlretrieve(uri, tmp.name)
        elif uri.startswith("s3://"):
            import boto3
            bucket, key = uri[5:].split("/", 1)
            boto3.client("s3").download_file(bucket, key, tmp.name)
        else:
            # Local file path
            tmp.name = uri

        waveform, sr = sf.read(tmp.name, dtype="float32")

    # Mix to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    return waveform


def process_job(r: redis.Redis, job_data: dict) -> None:
    """Process a single batch job."""
    job_id = job_data.get("job_id", "unknown")
    input_uri = job_data.get("input_uri", "")
    model_name = job_data.get("model", DEFAULT_MODEL)
    language = job_data.get("language", "") or None

    logger.info("[%s] Processing batch job: model=%s uri=%s", job_id, model_name, input_uri)

    # Update status to PROCESSING
    r.hset(f"scribe:job:{job_id}", "status", "PROCESSING")

    try:
        # Fetch and decode audio
        waveform = fetch_audio(input_uri)

        # Import transcription from the server module
        from scribe_server import transcribe_audio
        full_text, segments, duration_sec = transcribe_audio(waveform, model_name, language)

        # ── Embed segments for semantic search ───────────────────────────────
        try:
            from embedder import embed_and_index
            embed_and_index(r, job_id, segments)
        except Exception as e:
            logger.warning("[%s] Embedding failed (non-fatal): %s", job_id, e)

        # Write result
        result = {
            "job_id": job_id,
            "status": "COMPLETED",
            "duration_sec": duration_sec,
            "transcript": full_text,
            "segments": segments,
        }
        r.set(f"scribe:result:{job_id}", json.dumps(result))
        r.hset(f"scribe:job:{job_id}", mapping={
            "status": "COMPLETED",
            "duration_sec": str(duration_sec),
        })

        logger.info("[%s] Batch job completed: %d segments, %.1fs audio",
                    job_id, len(segments), duration_sec)

        # Optional callback
        callback_url = job_data.get("callback_url", "")
        if callback_url:
            try:
                req = urllib.request.Request(
                    callback_url,
                    data=json.dumps(result).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10)
                logger.info("[%s] Callback sent to %s", job_id, callback_url)
            except Exception as e:
                logger.warning("[%s] Callback failed: %s", job_id, e)

    except Exception as e:
        logger.error("[%s] Batch job failed: %s", job_id, e)
        r.hset(f"scribe:job:{job_id}", mapping={
            "status": "FAILED",
            "error": str(e),
        })


def consume_loop():
    """Main consumer loop — blocks and processes jobs from the Valkey stream."""
    r = get_valkey()
    ensure_consumer_group(r)

    logger.info("Batch consumer '%s' starting (group: %s)", CONSUMER_NAME, CONSUMER_GROUP)

    while True:
        try:
            # Read new messages from the stream
            entries = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=1,
                block=5000,  # 5s block
            )

            if not entries:
                continue

            for stream_name, messages in entries:
                for msg_id, data in messages:
                    # Decode bytes to str
                    job_data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in data.items()
                    }

                    process_job(r, job_data)

                    # Acknowledge the message
                    r.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)

        except redis.exceptions.ConnectionError as e:
            logger.warning("Valkey connection lost: %s. Retrying in 5s…", e)
            time.sleep(5)
            r = get_valkey()
        except Exception as e:
            logger.error("Consumer error: %s", e)
            time.sleep(1)


if __name__ == "__main__":
    consume_loop()
