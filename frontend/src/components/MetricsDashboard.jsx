/**
 * MetricsDashboard.jsx
 * Displays the three core performance metrics after each transcription run:
 *
 *   TTFT   Time-to-First-Token   – latency from request to first generated token (ms)
 *   ITL    Inter-Token Latency   – mean time between tokens during the decode phase (ms)
 *   RTFx   Real-Time Factor      – audio_duration / processing_time  (higher = faster)
 */

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

export default function MetricsDashboard({ metrics, mode }) {
  const { ttft_ms, itl_ms, rtfx } = metrics ?? {};

  return (
    <div className="metrics-dashboard">
      <h3 className="metrics-title">
        Performance Metrics
        {mode && (
          <span
            className="metrics-mode-badge"
            style={{ backgroundColor: mode === 'webgpu' ? '#059669' : '#4f46e5' }}
          >
            {mode === 'webgpu' ? 'WebGPU' : 'vLLM'}
          </span>
        )}
      </h3>
      <div className="metrics-grid">
        <MetricCard
          label="TTFT"
          value={ttft_ms != null ? ttft_ms.toFixed(1) : null}
          unit=" ms"
          description="Time-to-First-Token: latency from request submission to the first generated token"
          highlight={ttft_ms != null && ttft_ms < 200}
        />
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
