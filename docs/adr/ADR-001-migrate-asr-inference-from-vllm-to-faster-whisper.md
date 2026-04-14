# ADR-001: Migrate ASR Inference Engine from vLLM to faster-whisper

- Status: Accepted
- Date: 2026-04-14

## Context

We initially prototyped the ASR pipeline using vLLM due to its strong ecosystem momentum and performance reputation in large-scale inference. During single-node GPU validation for Open-ASR, the stack showed instability under concurrent workloads, including silent deadlocks and CUDA OOM failures.

The product direction has since shifted toward a resilient asynchronous data plane: FastAPI ingestion, Valkey queue/state, and decoupled Python workers. This architecture sustained high local burst traffic and better aligned with operational reliability goals.

## Decision

We are fully deprecating vLLM for audio transcription and standardizing on faster-whisper (CTranslate2 engine) behind an asynchronous Valkey-backed queue.

## Technical Rationale

### 1. VRAM Allocation Greed

vLLM is optimized for large datacenter deployments and aggressively pre-allocates GPU memory (often near 90%) for KV cache behavior. On single-node and edge GPU profiles, this leaves insufficient headroom for Whisper encoder workloads, increasing OOM risk under concurrency.

### 2. Architectural Mismatch

vLLM's main optimization, PagedAttention, is highly effective for decoder-only autoregressive LLM serving. Whisper is encoder-decoder and heavily constrained by audio encoding compute, not the same decode-time memory fragmentation pattern. The overhead of paged attention management does not return equivalent throughput gains for this ASR workload.

### 3. Feature Immaturity for ASR Operations

The current vLLM ASR path does not provide production-grade behavior needed for this data plane, especially robust and reliable word-level timestamp output suitable for downstream subtitle formats (SRT/VTT) and audit use cases.

### 4. Compute Efficiency on Single GPUs

faster-whisper (CTranslate2) offers native quantization paths (including INT8) and efficient transformer execution tuned for practical single-GPU serving. It supports higher concurrent file throughput with lower orchestration overhead and more predictable runtime behavior.

## Consequences

### Positive

- Improved reliability under bursty asynchronous ingestion patterns
- Better GPU utilization efficiency for target deployment shapes
- Cleaner separation of concerns between ingestion, queueing, and processing
- Reduced operational risk from model-server-specific instability

### Trade-offs

- Reduced leverage of vLLM-specific scheduling features
- Need to maintain worker-level lifecycle controls and queue semantics explicitly

## Implementation Notes

- Ingestion: FastAPI receives job metadata and enqueues Valkey state
- Queue/state: Valkey holds queue list and job hash lifecycle
- Processing: Worker pods consume queue items and run faster-whisper inference
- Observability: Processing metrics and queue health drive scaling and alerting

## Follow-up

- Maintain this ADR as the canonical record for ASR inference-engine policy
- Revisit only when objective evidence shows equal or better reliability for our ASR workload profile
