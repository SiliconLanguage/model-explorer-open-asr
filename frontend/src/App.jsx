/**
 * App.jsx
 * Root component for the Open-ASR Model Explorer.
 *
 * Orchestrates:
 *   - Model selection (strategy-based routing)
 *   - Audio capture / upload
 *   - Transcription dispatch via inferenceRouter
 *   - Live transcript streaming (LocalAgreement-2 for WebGPU)
 *   - Metrics dashboard
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import ModelSelector, { getModelByValue } from './components/ModelSelector';
import AudioRecorder from './components/AudioRecorder';
import SampleClips from './components/SampleClips';
import MetricsDashboard from './components/MetricsDashboard';
import TranscriptDisplay from './components/TranscriptDisplay';
import { route } from './services/inferenceRouter';

export default function App() {
  const [modelValue, setModelValue] = useState('cohere-hf-gpu');
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [status, setStatus] = useState('idle'); // idle | loading | streaming | done | error
  const [errorMsg, setErrorMsg] = useState(null);

  // Transcript state
  const [stableText, setStableText] = useState('');
  const [unstableText, setUnstableText] = useState('');

  // Metrics state
  const [metrics, setMetrics] = useState(null);
  const [transcribeMode, setTranscribeMode] = useState(null);

  const abortRef = useRef(false);
  const selectedModel = getModelByValue(modelValue);
  // Short model key for VRAM map lookup (strip org prefix, -ONNX suffix, lowercase)
  const modelKey = selectedModel?.id
    ? selectedModel.id.split('/').pop().replace(/-ONNX$/i, '').toLowerCase()
    : '';

  const isRunning = status === 'loading' || status === 'streaming';

  // ── Audio URL lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    if (!audioFile) { setAudioUrl(null); return; }
    const url = URL.createObjectURL(audioFile);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioFile]);

  // ── Progress callback from inferenceRouter ─────────────────────────────────
  const handleProgress = useCallback((data) => {
    if (abortRef.current) return;
    setStatus('streaming');

    if ('stable' in data) {
      // WebGPU LocalAgreement-2 update
      setStableText(data.stable ?? '');
      setUnstableText(data.unstable ?? '');
    } else if ('token' in data) {
      // Server-side SSE token
      setStableText((prev) => prev + (data.token ?? ''));
    }

    if (data.ttft_ms != null) {
      setMetrics((prev) => ({ ...prev, ttft_ms: data.ttft_ms }));
    }
  }, []);

  // ── Submit handler ─────────────────────────────────────────────────────────
  async function handleSubmit() {
    if (!audioFile) {
      alert('Please provide audio first (upload a file or use the recorder).');
      return;
    }

    abortRef.current = false;
    setStatus('loading');
    setErrorMsg(null);
    setStableText('');
    setUnstableText('');
    setMetrics(null);
    setTranscribeMode(null);

    try {
      const result = await route(selectedModel.id, audioFile, handleProgress, selectedModel.engine, selectedModel.mode, selectedLanguage);

      if (abortRef.current) return;

      setStableText(result.transcript);
      setUnstableText('');
      setMetrics({
        ttft_ms: result.ttft_ms,
        itl_ms: result.itl_ms,
        rtfx: result.rtfx,
      });
      setTranscribeMode(result.mode);
      setStatus('done');
    } catch (err) {
      if (!abortRef.current) {
        setErrorMsg(String(err));
        setStatus('error');
      }
    }
  }

  function handleStop() {
    abortRef.current = true;
    setStatus('idle');
  }

  function handleClear() {
    setStableText('');
    setUnstableText('');
    setMetrics(null);
    setTranscribeMode(null);
    setStatus('idle');
    setErrorMsg(null);
  }

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-title-row">
          <span className="header-icon">🎙</span>
          <h1 className="header-title">Open-ASR Model Explorer</h1>
        </div>
        <p className="header-subtitle">
          Hybrid inference testbed · vLLM server-side &amp; WebGPU client-side
        </p>
      </header>

      <main className="app-main">
        <div className="card">
          {/* ── Model selection ────────────────────────────────────────── */}
          <ModelSelector value={modelValue} onChange={setModelValue} disabled={isRunning} />

          {/* ── Audio input ────────────────────────────────────────────── */}
          <section className="section">
            <h2 className="section-title">Audio Input</h2>
            <AudioRecorder onAudioReady={setAudioFile} disabled={isRunning} externalAudioUrl={audioUrl} />
            <SampleClips onSampleSelect={(file, lang) => { setAudioFile(file); if (lang) setSelectedLanguage(lang); }} disabled={isRunning} />
          </section>

          {/* ── Action buttons ─────────────────────────────────────────── */}
          <div className="action-row">
            <button
              className="btn btn-primary btn-large"
              onClick={handleSubmit}
              disabled={isRunning || !audioFile}
            >
              {status === 'loading' ? (
                <>
                  <span className="spinner-inline" /> Loading model…
                </>
              ) : status === 'streaming' ? (
                <>
                  <span className="spinner-inline" /> Transcribing…
                </>
              ) : (
                '▶ Transcribe'
              )}
            </button>

            {isRunning && (
              <button className="btn btn-secondary" onClick={handleStop}>
                ⏹ Stop
              </button>
            )}

            {(status === 'done' || status === 'error') && (
              <button className="btn btn-ghost" onClick={handleClear}>
                🗑 Clear
              </button>
            )}
          </div>
        </div>

        {/* ── Error banner ─────────────────────────────────────────────── */}
        {status === 'error' && errorMsg && (
          <div className="error-banner" role="alert">
            <strong>Error:</strong> {errorMsg}
          </div>
        )}

        {/* ── Transcript output ──────────────────────────────────────── */}
        {(stableText || unstableText || isRunning) && (
          <div className="card">
            <TranscriptDisplay
              stable={stableText}
              unstable={unstableText}
              loading={status === 'loading'}
              mode={selectedModel.mode === 'webgpu' ? 'webgpu' : 'server'}
            />
          </div>
        )}

        {/* ── Metrics dashboard ─────────────────────────────────────── */}
        {(metrics || status === 'done') && (
          <div className="card">
            <MetricsDashboard
              metrics={metrics}
              engine={selectedModel?.mode === 'webgpu' ? 'webgpu' : (selectedModel?.engine ?? 'server')}
              model={modelKey}
            />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Open-ASR Model Explorer ·{' '}
          <a
            href="https://huggingface.co/spaces"
            target="_blank"
            rel="noopener noreferrer"
          >
            🤗 Hugging Face Spaces
          </a>
        </p>
      </footer>
    </div>
  );
}
