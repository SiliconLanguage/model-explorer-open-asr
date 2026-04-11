/**
 * MetricsDashboard.jsx
 * Displays performance metrics after each transcription run.
 *
 * The first metric card is engine-aware:
 *   WebGPU        → EST. VRAM   (static estimate from WEBGPU_VRAM_MAP)
 *   vLLM          → TTFT        (time-to-first-token from streaming)
 *   HF-GPU/HF-CPU → LATENCY    (total backend execution time; HF pipelines
 *                                do not stream, so TTFT is meaningless)
 *
 * The remaining cards (ITL, RTFx) are always shown.
 */

/** Estimated VRAM footprint (ONNX FP16/q4f16) per short model key. */
const WEBGPU_VRAM_MAP = {
  'whisper-tiny': '~75 MB',
  'whisper-base': '~150 MB',
  'cohere-transcribe-03-2026': '~3.8 GB',
  'qwen3-asr-1.7b': '~3.4 GB',
};

/** Badge config for each engine type. */
const ENGINE_BADGE = {
  webgpu:  { label: 'WebGPU',  color: '#059669' },
  vllm:    { label: 'vLLM',    color: '#4f46e5' },
  'hf-gpu': { label: 'HF-GPU', color: '#2563eb' },
  'hf-cpu': { label: 'HF-CPU', color: '#0f766e' },
};

function MetricCard({ label, value, unit, description, highlight }) {
  return (
    <div className={`metric-card ${highlight ? 'metric-card--highlight' : ''}`} title={description}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">
        {value != null ? value : '—'}
        {value != null && <span className="metric-unit">{unit}</span>}
      </span>
      <span className="metric-desc">{description}</span>
    </div>
  );
}

/**
 * Render the engine-specific first metric card.
 *   - WebGPU  → EST. VRAM (static map lookup)
 *   - vLLM    → TTFT      (real streaming TTFT)
 *   - HF-*    → LATENCY   (total backend elapsed time; backend returns it as
 *                           ttft_ms but for non-streaming engines it is total latency)
 */
function FirstMetricCard({ engine, model, ttft_ms }) {
  if (engine === 'webgpu') {
    const vram = WEBGPU_VRAM_MAP[model] || 'Unknown';
    return (
      <MetricCard
        label="EST. VRAM"
        value={vram}
        unit=""
        description="Estimated Video RAM allocation for the active WebGPU model weights and KV cache"
      />
    );
  }

  if (engine === 'vllm') {
    return (
      <MetricCard
        label="TTFT"
        value={ttft_ms != null ? ttft_ms.toFixed(1) : null}
        unit=" ms"
        description="Time-to-First-Token: latency from request submission to the first generated token"
        highlight={ttft_ms != null && ttft_ms < 200}
      />
    );
  }

  // HF-GPU / HF-CPU — the backend returns total elapsed time as ttft_ms
  return (
    <MetricCard
      label="LATENCY"
      value={ttft_ms != null ? ttft_ms.toFixed(1) : null}
      unit=" ms"
      description="Total backend execution time (Hugging Face pipelines do not stream TTFT)"
    />
  );
}

export default function MetricsDashboard({ metrics, engine, model }) {
  const { ttft_ms, itl_ms, rtfx } = metrics ?? {};
  const badge = ENGINE_BADGE[engine] ?? { label: engine ?? '', color: '#6b7280' };

  return (
    <div className="metrics-dashboard">
      <h3 className="metrics-title">
        Performance Metrics
        {engine && (
          <span
            className="metrics-mode-badge"
            style={{ backgroundColor: badge.color }}
          >
            {badge.label}
          </span>
        )}
      </h3>
      <div className="metrics-grid">
        <FirstMetricCard engine={engine} model={model} ttft_ms={ttft_ms} />
        <MetricCard
          label="ITL"
          value={itl_ms != null ? itl_ms.toFixed(1) : null}
          unit=" ms"
          description="Inter-Token Latency: mean time between successive tokens during the decode phase"
        />
        <MetricCard
          label="RTFx"
          value={rtfx != null ? rtfx.toFixed(2) : null}
          unit="×"
          description="Real-Time Factor: audio duration ÷ processing time (>1.0 = faster than real-time)"
          highlight={rtfx != null && rtfx > 1}
        />
      </div>
    </div>
  );
}
