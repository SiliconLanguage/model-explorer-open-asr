/**
 * SampleClips.jsx
 * "Examples" UI — lets the user click a language button to instantly load
 * a sample audio clip, replicating the Hugging Face demo UX.
 * Active selection "sticks" with highlighted styling.
 */

import { useState } from 'react';

const SAMPLES = [
  { name: 'english', label: '🇺🇸 English', file: '/samples/english.wav' },
  { name: 'chinese', label: '🇨🇳 Chinese', file: '/samples/chinese.wav' },
  { name: 'spanish', label: '🇪🇸 Spanish', file: '/samples/spanish.wav' },
];

export default function SampleClips({ onSampleSelect, disabled }) {
  const [loading, setLoading] = useState(null);
  const [selected, setSelected] = useState(null);

  async function handleClick(sample) {
    setLoading(sample.name);
    try {
      const response = await fetch(sample.file);
      if (!response.ok) throw new Error(`Failed to fetch ${sample.file}`);
      const blob = await response.blob();
      const file = new File([blob], sample.file.split('/').pop(), { type: 'audio/wav' });
      setSelected(sample.name);
      onSampleSelect(file);
    } catch (err) {
      console.error('Sample clip load failed:', err);
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="sample-clips">
      <p className="sample-clips-hint">Try one of the sample clips.</p>
      <div className="sample-clips-row">
        {SAMPLES.map((s) => (
          <button
            key={s.name}
            className={`btn ${selected === s.name ? 'btn-sample-active' : 'btn-secondary'}`}
            onClick={() => handleClick(s)}
            disabled={disabled || loading !== null}
          >
            {loading === s.name ? '…' : s.label}
          </button>
        ))}
      </div>
    </div>
  );
}
