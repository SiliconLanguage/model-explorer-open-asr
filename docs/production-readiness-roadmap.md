# SiliconLanguage: Open-ASR Model Explorer Production Readiness Roadmap

## Diagram

- Production architecture diagram: [diagrams/production-readiness-architecture.md](diagrams/production-readiness-architecture.md)

## Executive Summary

The validated asynchronous prototype has proven burst handling, but it still contains production risks: local spooling, single failure domains, and limited observability and policy enforcement. The production target is a stateless, spoolless, zero-copy ingestion plane where large audio payloads bypass the API process and flow directly to object storage. Control and state are centralized in managed, highly available services, while compute is event-scaled and health-gated.

This roadmap transitions Open-ASR from a validated prototype into a resilient enterprise-grade service by implementing a hardened traffic front door, decoupled S3 ingestion, reliable queue semantics, multi-AZ GPU worker operations, and full telemetry from request edge through inference completion.

## Pillar 1: Traffic Management & Security

### Architecture

Deprecate direct public access to FastAPI. Put API Gateway and WAF in front of all ingress:

1. External Client -> Load Balancer/WAF -> API Gateway
2. API Gateway -> private FastAPI service
3. FastAPI remains auth-agnostic for ingestion orchestration and job lifecycle APIs

Recommended gateway stack options:

- Kong
- APISIX
- AWS API Gateway

### Authentication

Implement server-to-server stateless JWT validation:

1. Client signs requests locally using client_secret (secret never sent over network)
2. Gateway validates JWT and forwards identity context downstream
3. Auth Microservice maintains hashed client credentials and rotates secrets
4. Valid client hash material is synced into Valkey for O(1) lookup by gateway auth path

### Client Experience

- Clients receive short-lived signed access to ingestion APIs
- Identity is bound to tenant and scope claims
- No long-lived credentials are exposed to backend ingestion handlers

### Rate Limiting

Apply strict tiered controls at API Gateway:

- Requests per minute
- MB per day
- Concurrent in-flight jobs per tenant
- Burst protection and abuse blocking

## Pillar 2: Data Plane, Storage & State

### Architecture

Deprecate local /data/audio_spool shared volume. It is a SPOF and scaling bottleneck.

### Implementation

Adopt S3 presigned URL ingestion:

1. FastAPI generates presigned upload URL
2. Client uploads audio directly to S3
3. S3 event emits notification via SQS or Lambda
4. Event handler pushes job id and S3 URI to Valkey HA queue

The queue stores metadata pointers only, not large binary payloads.

### Lifecycle

- Configure S3 lifecycle to delete processed audio objects after 24h (or defined TTL)
- Maintain object tags for audit and retention policy automation
- Enforce encryption and bucket policy boundaries

## Pillar 3: Queue & Worker Resilience

### Reliable Job Processing

Move from basic pop-only semantics to reliable delivery with explicit completion state:

1. Claim job with processing state
2. Execute transcription from S3 URI
3. Write transcript and metrics
4. ACK completion with final status

### Dead Letter Queue (DLQ)

Define poison-file strategy to avoid queue blockage:

- Retry failed jobs with capped attempts
- After 3 failures, route to DLQ
- Preserve error metadata and source URI for operator inspection
- Keep primary queue unblocked under pathological inputs

## Pillar 4: Infrastructure & Autoscaling (HA/BC)

### Infrastructure

Deploy in multi-AZ high availability topology:

- Managed Valkey HA cluster
- Distributed GPU worker pools across AZs
- No single host or zone outage should halt service

### Autoscaling

Do not CPU-scale GPU workers. Use event-driven scaling via KEDA:

- KEDA watches Valkey queue depth
- Queue depth is the primary scaling signal
- Scale-out and scale-in tuned with cooldown/stabilization windows

## Pillar 5: Observability, GPU Health & Failover

### Golden Signals

Build dashboards and alerts for:

- Queue depth and queue age
- Worker throughput and failure rate
- GPU utilization and memory pressure
- End-to-end latency from ingest request to completed transcript

### Hardware Monitoring

Deploy NVIDIA GPU Operator and DCGM Exporter on worker nodes:

- Stream VRAM usage, thermal limits, and Xid errors to Prometheus
- Surface per-node and per-pod GPU health in Grafana

### Failover Logic

Bind worker readiness/liveness probes to DCGM health status:

- If GPU degrades, readiness fails
- Pod stops pulling new jobs from Valkey
- Healthy pods continue processing, enabling automatic workload reroute

### Tracing

Implement OpenTelemetry end to end:

- Assign Trace ID at ingress
- Propagate through API Gateway, FastAPI, S3 event, Valkey queue, and worker
- Use trace spans to measure ingestion, queue wait, inference, and completion latency

## Delivery Sequence (Recommended)

1. WAF + API Gateway + JWT enforcement
2. S3 presigned upload and spoolless ingestion cutover
3. Reliable queue semantics and DLQ policy
4. Multi-AZ deployment and KEDA-based GPU autoscaling
5. DCGM health-gated probes, Prometheus/Grafana, and OTel tracing
6. Production burn-in with chaos/failover drills and SLO validation
