# ── Stage 1: Proto compilation ────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS proto

RUN pip install --no-cache-dir grpcio-tools==1.72.0 protobuf>=5.29.0

WORKDIR /build
COPY proto/scribe.proto ./proto/
RUN mkdir -p out && python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./out \
    --grpc_python_out=./out \
    proto/scribe.proto

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY worker/requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy generated proto stubs
COPY --from=proto /build/out/scribe_pb2.py /build/out/scribe_pb2_grpc.py ./

# Copy application code
COPY worker/scribe_server.py worker/batch_consumer.py worker/embedder.py ./

EXPOSE 50051

# Default: run the gRPC server.
# Override with CMD ["python3", "batch_consumer.py"] for the batch worker.
CMD ["python3", "scribe_server.py"]
