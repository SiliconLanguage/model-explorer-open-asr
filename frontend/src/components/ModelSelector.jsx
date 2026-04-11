/**
 * ModelSelector.jsx
 * Dropdown for choosing the ASR model.
 * WebGPU models (mode: 'webgpu') are tagged "Client-Side (WebGPU)";
 * others are tagged by their server-side engine.
 */

const MODELS = [
  { value: 'whisper-base-hf-gpu', id: 'openai/whisper-base', label: 'Whisper Base (Server: HF-GPU Fast)', mode: 'server', engine: 'hf-gpu' },
  { value: 'whisper-base-hf-cpu', id: 'openai/whisper-base', label: 'Whisper Base (Server: HF-CPU Safe)', mode: 'server', engine: 'hf-cpu' },
  { value: 'cohere-hf-gpu', id: 'CohereLabs/cohere-transcribe-03-2026', label: 'Cohere Transcribe (HF-GPU Safe)', mode: 'server', engine: 'hf-gpu' },
  { value: 'cohere-hf-cpu', id: 'CohereLabs/cohere-transcribe-03-2026', label: 'Cohere Transcribe (HF-CPU Safe)', mode: 'server', engine: 'hf-cpu' },
  { value: 'Qwen3-ASR-1.7B', id: 'Qwen3-ASR-1.7B', label: 'Qwen3-ASR 1.7B', mode: 'server', engine: 'vllm' },
  { value: 'ibm-granite/granite-4.0-1b-speech', id: 'ibm-granite/granite-4.0-1b-speech', label: 'IBM Granite 4.0 1B Speech', mode: 'server', engine: 'vllm' },
  { value: 'whisper-tiny-webgpu', id: 'Xenova/whisper-tiny', label: 'Whisper Tiny (WebGPU Safe)', mode: 'webgpu' },
  { value: 'whisper-base-webgpu', id: 'Xenova/whisper-base', label: 'Whisper Base (WebGPU)', mode: 'webgpu' },
];

const BADGE = {
  'vllm-server': { label: 'vLLM · Server-Side', color: '#4f46e5' },
  'hf-gpu-server': { label: 'HF GPU · Server-Side', color: '#2563eb' },
  'hf-cpu-server': { label: 'HF CPU · Server-Side', color: '#0f766e' },
  webgpu: { label: 'WebGPU · Client-Side', color: '#059669' },
};

function modelBadge(model) {
  if (model.mode === 'webgpu') return BADGE.webgpu;
  if (model.engine === 'hf-gpu') return BADGE['hf-gpu-server'];
  if (model.engine === 'hf-cpu') return BADGE['hf-cpu-server'];
  return BADGE['vllm-server'];
}

function modelTitle(model) {
  if (model.mode === 'webgpu') {
    return 'Runs entirely in the browser - no audio leaves your device';
  }
  if (model.engine === 'hf-gpu') {
    return 'Routes audio to backend transformers on GPU';
  }
  if (model.engine === 'hf-cpu') {
    return 'Routes audio to backend transformers on CPU';
  }
  return 'Routes audio to FastAPI/vLLM backend';
}

export function getModelByValue(value) {
  return MODELS.find((m) => m.value === value) ?? MODELS[0];
}

export default function ModelSelector({ value, onChange, disabled }) {
  const selected = getModelByValue(value);
  const badge = modelBadge(selected);

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
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
        <span
          className="badge"
          style={{ backgroundColor: badge.color }}
          title={modelTitle(selected)}
        >
          {badge.label}
        </span>
      </div>
    </div>
  );
}

export { MODELS };
