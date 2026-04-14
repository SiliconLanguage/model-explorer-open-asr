# Production Readiness Architecture Diagram

```mermaid
flowchart LR
  %% Ingestion Flow
  C[External Client] --> WAF[Load Balancer / WAF]
  WAF --> GW[API Gateway\nAuth + Rate Limiting]
  GW --> F[FastAPI Backend\nAuth-agnostic]
  F -->|Generate Presigned URL| C
  C -->|Direct Audio Upload| S3[(AWS S3 Bucket)]
  S3 --> EVT[S3 Event Notification\nSQS or Lambda]
  EVT -->|Push Job ID + S3 URI| V[(Valkey HA Cluster\nscribe:queue + job hashes)]

  %% Processing Flow
  subgraph AZ1[Multi-AZ Worker Plane]
    W1[GPU Worker Pod A\nFaster-Whisper]
    W2[GPU Worker Pod B\nFaster-Whisper]
  end

  subgraph AZ2[Multi-AZ Worker Plane]
    W3[GPU Worker Pod C\nFaster-Whisper]
    W4[GPU Worker Pod D\nFaster-Whisper]
  end

  V -->|Pop Job| W1
  V -->|Pop Job| W2
  V -->|Pop Job| W3
  V -->|Pop Job| W4

  W1 -->|Read Audio| S3
  W2 -->|Read Audio| S3
  W3 -->|Read Audio| S3
  W4 -->|Read Audio| S3

  W1 -->|Update completed/failed| V
  W2 -->|Update completed/failed| V
  W3 -->|Update completed/failed| V
  W4 -->|Update completed/failed| V

  %% Decoupled Auth Service
  AUTH[Auth Microservice\nclient_id/client_secret hashing] -->|Sync valid client hash set| V
  V -->|Zero-latency credential lookup| GW

  %% Control Plane
  KEDA[KEDA\nEvent-driven Autoscaling] -->|Scale GPU worker pods\nby queue depth| W1
  KEDA --> W2
  KEDA --> W3
  KEDA --> W4
  V -->|Queue depth metric| KEDA

  DCGM[NVIDIA GPU Operator + DCGM Exporter] --> PROBE[K8s Readiness/Liveness\nProbe Gates]
  PROBE -->|If degraded GPU\nstop pulling new jobs| W1
  PROBE --> W2
  PROBE --> W3
  PROBE --> W4

  OTEL[OpenTelemetry Collector]
  GW --> OTEL
  F --> OTEL
  EVT --> OTEL
  V --> OTEL
  W1 --> OTEL
  W2 --> OTEL
  W3 --> OTEL
  W4 --> OTEL

  PROM[Prometheus + Grafana]
  V -->|Queue depth| PROM
  DCGM -->|GPU health metrics| PROM
  OTEL -->|Traces| PROM
```

## Notes

- This is the production target architecture for Open-ASR and intentionally removes local audio spooling from the ingestion path.
- The API process manages control-plane metadata only; binary data plane traffic goes directly client -> S3.
- Queue depth is the primary autoscaling input for GPU workers.
