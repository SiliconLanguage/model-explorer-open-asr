"""
Scribe v1 — Transcript Embedder + Valkey Semantic Search

Vectorises transcript segments via sentence-transformers and indexes
them in Valkey using VECTOR HNSW for semantic search.

Used by:
  - batch_consumer.py   (at job completion)
  - Query endpoint       (future /search API)

Environment variables:
  VALKEY_ENDPOINT  — Valkey address (default: valkey:6379)
  EMBED_MODEL      — Sentence-transformers model (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import redis

logger = logging.getLogger("embedder")

VALKEY_ENDPOINT = os.getenv("VALKEY_ENDPOINT", "valkey:6379")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
VECTOR_DIM = 384  # all-MiniLM-L6-v2 produces 384-d vectors
INDEX_NAME = "scribe_segments"
PREFIX = "scribe:seg:"

# ── Lazy model singleton ─────────────────────────────────────────────────────

_model = None


def get_embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("Embedding model ready (dim=%d)", VECTOR_DIM)
    return _model


# ── Index management ──────────────────────────────────────────────────────────

def ensure_index(r: redis.Redis) -> None:
    """Create the Valkey HNSW vector index if it doesn't exist."""
    try:
        r.execute_command("FT.INFO", INDEX_NAME)
        return  # index exists
    except redis.exceptions.ResponseError:
        pass

    r.execute_command(
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH",
        "PREFIX", "1", PREFIX,
        "SCHEMA",
            "job_id",    "TAG",
            "text",      "TEXT",
            "start",     "NUMERIC", "SORTABLE",
            "end",       "NUMERIC", "SORTABLE",
            "speaker",   "TAG",
            "embedding", "VECTOR", "HNSW", "6",
                         "TYPE", "FLOAT32",
                         "DIM", str(VECTOR_DIM),
                         "DISTANCE_METRIC", "COSINE",
    )
    logger.info("Created FT index '%s' (HNSW, dim=%d, COSINE)", INDEX_NAME, VECTOR_DIM)


# ── Embed + Index ─────────────────────────────────────────────────────────────

def embed_and_index(
    r: redis.Redis,
    job_id: str,
    segments: list[dict],
) -> int:
    """Embed transcript segments and store in Valkey for semantic search.

    Args:
        r: Valkey connection.
        job_id: The batch job ID.
        segments: List of dicts with keys: text, start, end, speaker, confidence.

    Returns:
        Number of segments indexed.
    """
    if not segments:
        return 0

    ensure_index(r)
    model = get_embed_model()

    texts = [seg.get("text", "") for seg in segments]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    pipe = r.pipeline(transaction=False)
    for i, (seg, emb) in enumerate(zip(segments, embeddings)):
        key = f"{PREFIX}{job_id}:{i}"
        pipe.hset(key, mapping={
            "job_id":    job_id,
            "text":      seg.get("text", ""),
            "start":     str(seg.get("start", 0.0)),
            "end":       str(seg.get("end", 0.0)),
            "speaker":   seg.get("speaker", ""),
            "embedding": np.array(emb, dtype=np.float32).tobytes(),
        })
    pipe.execute()

    logger.info("[%s] Indexed %d segments in Valkey HNSW", job_id, len(segments))
    return len(segments)


# ── Semantic Search ───────────────────────────────────────────────────────────

def search(
    r: redis.Redis,
    query: str,
    top_k: int = 10,
    job_id: Optional[str] = None,
) -> list[dict]:
    """KNN semantic search over indexed transcript segments.

    Args:
        r: Valkey connection.
        query: Natural-language search query.
        top_k: Number of results to return.
        job_id: Optional filter to scope search to a single job.

    Returns:
        List of dicts with keys: key, text, start, end, speaker, score.
    """
    model = get_embed_model()
    qvec = model.encode([query], normalize_embeddings=True)[0]
    blob = np.array(qvec, dtype=np.float32).tobytes()

    # Build the filter
    filter_clause = f"@job_id:{{{job_id}}}" if job_id else "*"

    # FT.SEARCH with KNN
    raw = r.execute_command(
        "FT.SEARCH", INDEX_NAME,
        f"({filter_clause})=>[KNN {top_k} @embedding $vec AS score]",
        "PARAMS", "2", "vec", blob,
        "SORTBY", "score",
        "RETURN", "5", "text", "start", "end", "speaker", "score",
        "DIALECT", "2",
    )

    # Parse RediSearch response: [total, key1, [field, val, ...], key2, ...]
    results = []
    total = raw[0]
    i = 1
    while i < len(raw):
        key = raw[i].decode() if isinstance(raw[i], bytes) else raw[i]
        fields = raw[i + 1]
        entry = {"key": key}
        for j in range(0, len(fields), 2):
            fname = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
            fval = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]
            entry[fname] = fval
        results.append(entry)
        i += 2

    return results
