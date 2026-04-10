/**
 * ModelSelector.jsx
 * Dropdown for choosing the ASR model.
 * Models containing "WebGPU" are tagged as "Client-Side (WebGPU)";
 * others are tagged as "Server-Side (vLLM)".
 */

const MODELS = [
  { id: 'Cohere-transcribe-03-2026',          label: 'Cohere Transcribe 03-2026',          mode: 'server' },
  { id: 'Qwen3-ASR-1.7B',                     label: 'Qwen3-ASR 1.7B',                     mode: 'server' },
  { id: 'ibm-granite/granite-4.0-1b-speech',  label: 'IBM Granite 4.0 1B Speech',          mode: 'server' },
  { id: 'Cohere-Transcribe-WebGPU',           label: 'Cohere Transcribe (WebGPU)',          mode: 'webgpu' },
  { id: 'Whisper-Large-v3-Turbo-WebGPU',      label: 'Whisper Large v3 Turbo (WebGPU)',    mode: 'webgpu' },
];

const BADGE = {
  server: { label: 'vLLM · Server-Side', color: '#4f46e5' },
  webgpu: { label: 'WebGPU · Client-Side', color: '#059669' },
};

export default function ModelSelector({ value, onChange, disabled }) {
  const selected = MODELS.find((m) => m.id === value) ?? MODELS[0];
  const badge = BADGE[selected.mode];

  return (
    <div className="model-selector">
      <label htmlFor="model-select" className="label">
        Model
      </label>
      <div className="select-row">
        <select
          id="model-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="select"
        >
          {MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <span
          className="badge"
          style={{ backgroundColor: badge.color }}
          title={selected.mode === 'webgpu' ? 'Runs entirely in the browser – no audio leaves your device' : 'Routes audio to the FastAPI/vLLM backend'}
        >
          {badge.label}
        </span>
      </div>
    </div>
  );
}

export { MODELS };
