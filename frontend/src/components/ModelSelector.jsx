/**
 * ModelSelector.jsx
 * Dropdown for choosing the ASR model.
 * WebGPU models (mode: 'webgpu') are tagged "Client-Side (WebGPU)";
 * others are tagged by their server-side engine.
 */

const MODELS = [
  { value: 'whisper-base-hf-gpu', id: 'openai/whisper-base', label: 'Whisper Base (Server GPU)', mode: 'server', engine: 'hf-gpu' },
  { value: 'whisper-base-hf-cpu', id: 'openai/whisper-base', label: 'Whisper Base (Server CPU)', mode: 'server', engine: 'hf-cpu' },
  { value: 'whisper-base-faster', id: 'openai/whisper-base', label: 'Whisper Base (faster-whisper CTranslate2)', mode: 'server', engine: 'faster_whisper' },
  { value: 'cohere-hf-gpu', id: 'CohereLabs/cohere-transcribe-03-2026', label: 'cohere-transcribe-03-2026 (Server GPU)', mode: 'server', engine: 'hf-gpu' },
  { value: 'cohere-hf-cpu', id: 'CohereLabs/cohere-transcribe-03-2026', label: 'cohere-transcribe-03-2026 (Server CPU)', mode: 'server', engine: 'hf-cpu' },
  { value: 'qwen3-asr-hf-gpu', id: 'Qwen3-ASR-1.7B', label: 'Qwen3-ASR 1.7B (Server GPU)', mode: 'server', engine: 'hf-gpu' },
  { value: 'granite-hf-gpu', id: 'ibm-granite/granite-4.0-1b-speech', label: 'Granite 4.0 1B Speech (Server GPU)', mode: 'server', engine: 'hf-gpu' },
  { value: 'vibevoice-asr', id: 'microsoft/VibeVoice-ASR-HF', label: 'VibeVoice-ASR 9B (Batch Worker)', mode: 'server', engine: 'vibevoice' },
  { value: 'whisper-tiny-webgpu', id: 'Xenova/whisper-tiny', label: 'Whisper Tiny (WebGPU Safe)', mode: 'webgpu' },
  { value: 'whisper-base-webgpu', id: 'Xenova/whisper-base', label: 'Whisper Base (WebGPU - Unstable)', mode: 'webgpu' },
  { value: 'cohere-webgpu', id: 'onnx-community/cohere-transcribe-03-2026-ONNX', label: 'cohere-transcribe-03-2026 (WebGPU)', mode: 'webgpu' },
];

const BADGE = {
  'vllm-server': { label: 'Server Stream · Legacy', color: '#4f46e5' },
  'hf-gpu-server': { label: 'Server GPU · API', color: '#2563eb' },
  'hf-cpu-server': { label: 'Server CPU · API', color: '#0f766e' },
  'faster-whisper-server': { label: 'faster-whisper · Server-Side', color: '#7c3aed' },
  'vibevoice-server': { label: 'VibeVoice · Batch Worker', color: '#dc2626' },
  webgpu: { label: 'WebGPU · Client-Side', color: '#059669' },
};

function modelBadge(model) {
  if (model.mode === 'webgpu') return BADGE.webgpu;
  if (model.engine === 'hf-gpu') return BADGE['hf-gpu-server'];
  if (model.engine === 'hf-cpu') return BADGE['hf-cpu-server'];
  if (model.engine === 'faster_whisper') return BADGE['faster-whisper-server'];
  if (model.engine === 'vibevoice') return BADGE['vibevoice-server'];
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
  if (model.engine === 'faster_whisper') {
    return 'Routes audio to faster-whisper CTranslate2 engine';
  }
  if (model.engine === 'vibevoice') {
    return 'Microsoft VibeVoice-ASR 9B — speaker diarization + timestamps via batch worker';
  }
  return 'Routes audio to backend streaming engine';
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
