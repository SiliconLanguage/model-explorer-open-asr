# ── Scribe Gateway: Go/Gin Ingestion Service ────────────────────────────────
# Multi-stage build: protoc code-gen → Go compile → distroless runtime
FROM golang:1.24-bookworm AS builder

# Install protoc and Go plugins
RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

WORKDIR /build

# Copy proto definition and generate Go code
# Build context is scribe/ (set by docker-compose)
COPY proto/scribe.proto ./proto/
RUN mkdir -p pb && protoc \
    --go_out=pb --go_opt=paths=source_relative \
    --go-grpc_out=pb --go-grpc_opt=paths=source_relative \
    -I proto proto/scribe.proto

# Copy Go source and build
COPY gateway/go.mod ./
COPY gateway/main.go .
RUN go mod tidy && go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /scribe-gateway .

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /scribe-gateway /scribe-gateway
EXPOSE 8080
ENTRYPOINT ["/scribe-gateway"]
