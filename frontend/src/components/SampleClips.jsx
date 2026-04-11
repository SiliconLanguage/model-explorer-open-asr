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
  { name: 'french', label: '🇫🇷 French', file: '/samples/french.wav' },
  { name: 'german', label: '🇩🇪 German', file: '/samples/german.wav' },
  { name: 'italian', label: '🇮🇹 Italian', file: '/samples/italian.wav' },
  { name: 'japanese', label: '🇯🇵 Japanese', file: '/samples/japanese.wav' },
  { name: 'hindi', label: '🇮🇳 Hindi', file: '/samples/hindi.wav' },
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
      // Detect actual audio format from magic bytes (servers may return
      // application/octet-stream for .wav files, and some samples may be
      // OGG-encoded despite the .wav extension).
      const header = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
      const magic = String.fromCharCode(...header);
      let mimeType = 'audio/wav';
      if (magic === 'OggS') mimeType = 'audio/ogg';
      else if (magic === 'fLaC') mimeType = 'audio/flac';
      else if (magic.startsWith('\xff\xfb') || magic.startsWith('ID3')) mimeType = 'audio/mpeg';
      const file = new File([blob], sample.file.split('/').pop(), { type: mimeType });
      setSelected(sample.name);
      onSampleSelect(file, sample.name);
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
