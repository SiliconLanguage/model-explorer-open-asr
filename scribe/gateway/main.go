// Scribe v1 — Go/Gin Ingestion Gateway
//
// This service acts as the high-concurrency "Traffic Controller" for the
// Zoom Scribe-compatible API.  It directs synchronous traffic (Fast Mode)
// straight to the Python inference worker via gRPC, and queues asynchronous
// traffic (Batch Mode) into a Valkey stream.
//
// Environment variables:
//   WORKER_ENDPOINT   — gRPC address of the Python worker (default: worker:50051)
//   VALKEY_ENDPOINT   — Valkey/Redis address (default: valkey:6379)
//   LISTEN_ADDR       — HTTP listen address (default: :8080)
//   NORMALIZER_ENDPOINT — Rust audio normalizer sidecar (default: normalizer:9090)

package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/SiliconLanguage/model-explorer-open-asr/scribe/gateway/pb"
)

var (
	valkeyClient *redis.Client
	grpcClient   pb.ScribeEngineClient
	grpcConn     *grpc.ClientConn
	normEndpoint string
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	// ── Valkey ──────────────────────────────────────────────────────────────
	valkeyAddr := envOr("VALKEY_ENDPOINT", "valkey:6379")
	valkeyClient = redis.NewClient(&redis.Options{Addr: valkeyAddr})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := valkeyClient.Ping(ctx).Err(); err != nil {
		log.Printf("WARN: Valkey not reachable at %s: %v (batch mode disabled)", valkeyAddr, err)
	} else {
		log.Printf("INFO: Connected to Valkey at %s", valkeyAddr)
	}

	// ── gRPC ────────────────────────────────────────────────────────────────
	workerAddr := envOr("WORKER_ENDPOINT", "worker:50051")
	var err error
	grpcConn, err = grpc.NewClient(
		workerAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64*1024*1024)),
	)
	if err != nil {
		log.Fatalf("FATAL: Cannot create gRPC client for %s: %v", workerAddr, err)
	}
	defer grpcConn.Close()
	grpcClient = pb.NewScribeEngineClient(grpcConn)
	log.Printf("INFO: gRPC client targeting %s", workerAddr)

	// ── Rust normalizer sidecar ─────────────────────────────────────────────
	normEndpoint = envOr("NORMALIZER_ENDPOINT", "http://normalizer:9090")
	log.Printf("INFO: Audio normalizer sidecar at %s", normEndpoint)

	// ── Gin HTTP server ─────────────────────────────────────────────────────
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	// Health check
	r.GET("/health", handleHealth)

	// Zoom Scribe v1 compatible API
	v1 := r.Group("/aiservices/scribe")
	{
		v1.POST("/fast", handleFastMode)
		v1.POST("/jobs", handleBatchMode)
		v1.GET("/jobs/:job_id", handleJobStatus)
	}

	listenAddr := envOr("LISTEN_ADDR", ":8080")
	log.Printf("INFO: Scribe Gateway listening on %s", listenAddr)
	if err := r.Run(listenAddr); err != nil {
		log.Fatalf("FATAL: Server failed: %v", err)
	}
}

// ── Health ──────────────────────────────────────────────────────────────────

func handleHealth(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	status := gin.H{
		"service": "scribe-gateway",
		"version": "v1",
		"grpc":    "unknown",
		"valkey":  "unknown",
	}

	if err := valkeyClient.Ping(ctx).Err(); err == nil {
		status["valkey"] = "ok"
	} else {
		status["valkey"] = fmt.Sprintf("error: %v", err)
	}

	// gRPC readiness is best-effort; we just report connectivity state
	state := grpcConn.GetState().String()
	status["grpc"] = state

	c.JSON(http.StatusOK, status)
}

// ── Fast Mode: synchronous gRPC streaming ───────────────────────────────────

func handleFastMode(c *gin.Context) {
	file, header, err := c.Request.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing 'audio' file field"})
		return
	}
	defer file.Close()

	model := c.DefaultPostForm("model", "large-v3")
	language := c.DefaultPostForm("language", "")
	requestID := uuid.New().String()

	log.Printf("INFO: [%s] Fast transcribe: model=%s lang=%s file=%s",
		requestID, model, language, header.Filename)

	// ── Optional: normalize via Rust sidecar ────────────────────────────────
	var audioReader io.Reader = file
	normalizedData, normErr := normalizeAudio(file, header.Filename)
	if normErr != nil {
		log.Printf("WARN: [%s] Normalizer unavailable, sending raw: %v", requestID, normErr)
		// Reset file position and send raw
		if seeker, ok := file.(io.Seeker); ok {
			seeker.Seek(0, io.SeekStart)
		}
	} else {
		audioReader = normalizedData
	}

	// ── Stream to Python worker via gRPC ────────────────────────────────────
	ctx, cancel := context.WithTimeout(c.Request.Context(), 120*time.Second)
	defer cancel()

	stream, err := grpcClient.FastTranscribe(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":      "inference worker unavailable",
			"request_id": requestID,
			"detail":     err.Error(),
		})
		return
	}

	// Stream audio in 64KB chunks
	buf := make([]byte, 64*1024)
	firstChunk := true
	for {
		n, readErr := audioReader.Read(buf)
		if n > 0 {
			chunk := &pb.AudioChunk{
				Data:     buf[:n],
				Model:    model,
				Language: language,
			}
			if firstChunk {
				chunk.SampleRate = 16000
				firstChunk = false
			}
			if sendErr := stream.Send(chunk); sendErr != nil {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":      "stream send failed",
					"request_id": requestID,
					"detail":     sendErr.Error(),
				})
				return
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":      "audio read error",
				"request_id": requestID,
				"detail":     readErr.Error(),
			})
			return
		}
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "transcription failed",
			"request_id": requestID,
			"detail":     err.Error(),
		})
		return
	}

	// Map gRPC response to Zoom Scribe v1 JSON
	segments := make([]gin.H, 0)
	if resp.Result != nil {
		for _, seg := range resp.Result.Segments {
			segments = append(segments, gin.H{
				"start":      seg.Start,
				"end":        seg.End,
				"text":       seg.Text,
				"speaker":    seg.Speaker,
				"confidence": seg.Confidence,
			})
		}
	}

	transcript := ""
	if resp.Result != nil {
		transcript = resp.Result.TextDisplay
	}

	c.JSON(http.StatusOK, gin.H{
		"request_id":   requestID,
		"status":       "COMPLETED",
		"duration_sec":  resp.DurationSec,
		"transcript":   transcript,
		"segments":     segments,
	})
}

// ── Batch Mode: Valkey Streams ──────────────────────────────────────────────

type batchRequest struct {
	Input struct {
		URI string `json:"uri" binding:"required"`
	} `json:"input" binding:"required"`
	Model    string `json:"model"`
	Language string `json:"language"`
	Callback string `json:"callback_url"`
}

func handleBatchMode(c *gin.Context) {
	var req batchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	jobID := uuid.New().String()
	model := req.Model
	if model == "" {
		model = "large-v3"
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	// Push job into Valkey Stream
	_, err := valkeyClient.XAdd(ctx, &redis.XAddArgs{
		Stream: "scribe_jobs",
		Values: map[string]interface{}{
			"job_id":       jobID,
			"input_uri":    req.Input.URI,
			"model":        model,
			"language":     req.Language,
			"callback_url": req.Callback,
			"status":       "QUEUED",
			"created_at":   time.Now().UTC().Format(time.RFC3339),
		},
	}).Result()

	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":  "failed to queue job",
			"detail": err.Error(),
		})
		return
	}

	// Also store job metadata as a hash for status lookups
	valkeyClient.HSet(ctx, "scribe:job:"+jobID, map[string]interface{}{
		"status":    "QUEUED",
		"input_uri": req.Input.URI,
		"model":     model,
		"language":  req.Language,
	})

	log.Printf("INFO: [%s] Batch job queued: model=%s uri=%s", jobID, model, req.Input.URI)

	c.JSON(http.StatusAccepted, gin.H{
		"job_id": jobID,
		"status": "QUEUED",
	})
}

// ── Job Status ──────────────────────────────────────────────────────────────

func handleJobStatus(c *gin.Context) {
	jobID := c.Param("job_id")

	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
	defer cancel()

	result, err := valkeyClient.HGetAll(ctx, "scribe:job:"+jobID).Result()
	if err != nil || len(result) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
		return
	}

	response := gin.H{
		"job_id": jobID,
		"status": result["status"],
	}

	// If completed, include the transcript
	if result["status"] == "COMPLETED" {
		transcript, _ := valkeyClient.Get(ctx, "scribe:result:"+jobID).Result()
		response["transcript"] = transcript
	}

	c.JSON(http.StatusOK, response)
}

// ── Audio normalizer helper ─────────────────────────────────────────────────

func normalizeAudio(reader io.Reader, filename string) (io.Reader, error) {
	// POST raw audio to Rust normalizer sidecar, get back 16kHz mono PCM
	resp, err := http.Post(
		normEndpoint+"/normalize",
		"application/octet-stream",
		reader,
	)
	if err != nil {
		return nil, fmt.Errorf("normalizer request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("normalizer returned %d", resp.StatusCode)
	}
	return resp.Body, nil
}
