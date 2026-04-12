# ── Scribe Normalizer: Rust Audio Preprocessing Sidecar ──────────────────────
# Multi-stage: Rust compile → minimal runtime
FROM rust:1.86-bookworm AS builder

# Install audio codec dev libraries for Symphonia
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml ./
COPY src/ ./src/

RUN cargo build --release

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/scribe-normalizer /usr/local/bin/
EXPOSE 9090
ENV LISTEN_ADDR=0.0.0.0:9090
ENTRYPOINT ["scribe-normalizer"]
