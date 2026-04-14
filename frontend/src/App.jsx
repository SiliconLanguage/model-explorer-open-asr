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
import JobsList from './components/JobsList';
import StagedFiles from './components/StagedFiles';
import { route, submitBatchConcurrent, pollJob, fetchAllJobs, deleteJob, deleteAllJobs, resubmitJob } from './services/inferenceRouter';

// ── Session ID: scopes jobs to this browser tab ──────────────────────────────
function getSessionId() {
  let id = sessionStorage.getItem('asr_session_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('asr_session_id', id);
  }
  return id;
}
const SESSION_ID = getSessionId();

export default function App() {
  const [modelValue, setModelValue] = useState('whisper-base-faster');
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [sampleLoading, setSampleLoading] = useState(false);
  const [status, setStatus] = useState('idle'); // idle | loading | streaming | done | error
  const [errorMsg, setErrorMsg] = useState(null);

  // Transcript state
  const [stableText, setStableText] = useState('');
  const [unstableText, setUnstableText] = useState('');

  // Metrics state
  const [metrics, setMetrics] = useState(null);
  const [transcribeMode, setTranscribeMode] = useState(null);

  // Jobs store: { jobId: { status, original_filename, transcript, ... } }
  const [jobs, setJobs] = useState({});
  const [selectedJobId, setSelectedJobId] = useState(null);

  // Staged files: files waiting to be batch-submitted
  const [stagedFiles, setStagedFiles] = useState([]);
  const [selectedStagedIndex, setSelectedStagedIndex] = useState(null);

  const abortRef = useRef(false);
  const selectedModel = getModelByValue(modelValue);
  // Short model key for VRAM map lookup (strip org prefix, -ONNX suffix, lowercase)
  const modelKey = selectedModel?.id
    ? selectedModel.id.split('/').pop().replace(/-ONNX$/i, '').toLowerCase()
    : '';

  const isRunning = status === 'loading' || status === 'streaming';
  const hasActiveJobs = Object.values(jobs).some(
    (j) => j.status === 'queued' || j.status === 'processing',
  );
  const isBusy = isRunning || sampleLoading || hasActiveJobs;

  // ── Load persisted jobs on mount + auto-refresh ─────────────────────────────
  const refreshTimerRef = useRef(null);

  const refreshJobs = useCallback(() => {
    fetchAllJobs(SESSION_ID)
      .then((existingJobs) => {
        if (existingJobs && Object.keys(existingJobs).length > 0) {
          setJobs((prev) => {
            const merged = { ...prev };
            for (const [id, job] of Object.entries(existingJobs)) {
              merged[id] = { ...merged[id], ...job };
            }
            return merged;
          });
        }
      })
      .catch(() => { /* Valkey may be unavailable */ });
  }, []);

  useEffect(() => {
    refreshJobs();
    // Resume polling for non-terminal jobs after refresh
    fetchAllJobs(SESSION_ID).then((existingJobs) => {
      if (!existingJobs) return;
      for (const [id, job] of Object.entries(existingJobs)) {
        if (job.status === 'queued' || job.status === 'processing') {
          startPolling(id);
        }
      }
    }).catch(() => {});
  }, []);

  // Auto-refresh when any job is still in-flight
  useEffect(() => {
    const hasActiveJobs = Object.values(jobs).some(
      (j) => j.status === 'queued' || j.status === 'processing'
    );
    if (hasActiveJobs) {
      refreshTimerRef.current = setInterval(refreshJobs, 3000);
    } else {
      clearInterval(refreshTimerRef.current);
    }
    return () => clearInterval(refreshTimerRef.current);
  }, [jobs, refreshJobs]);

  // ── Audio URL lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    if (!audioFile) { setAudioUrl(null); return; }
    const url = URL.createObjectURL(audioFile);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioFile]);

  // ── Job helpers ────────────────────────────────────────────────────────────
  function updateJob(jobId, data) {
    setJobs((prev) => ({ ...prev, [jobId]: { ...prev[jobId], ...data } }));
  }

  function startPolling(jobId) {
    pollJob(jobId, (data) => updateJob(jobId, data)).catch((err) => {
      updateJob(jobId, { status: 'failed', error: String(err) });
    });
  }

  // ── Select a job and display its transcript ────────────────────────────────
  function handleJobSelect(jobId) {
    setSelectedJobId(jobId);
    const job = jobs[jobId];
    if (job) {
      setStableText(job.transcript || '');
      setUnstableText(
        job.status === 'queued' ? 'Queued…'
          : job.status === 'processing' ? 'Processing…'
          : ''
      );
      if (job.status === 'completed') {
        setStatus('done');
        setMetrics({
          ttft_ms: job.ttft_ms ? parseFloat(job.ttft_ms) : null,
          itl_ms: job.itl_ms ? parseFloat(job.itl_ms) : null,
          rtfx: job.duration_s && job.processing_time_s
            ? parseFloat(job.duration_s) / parseFloat(job.processing_time_s)
            : null,
        });
      } else if (job.status === 'failed') {
        setStatus('error');
        setErrorMsg(job.error || 'Job failed');
      } else {
        setStatus('streaming');
      }
    }
  }

  // Keep display in sync when the selected job updates in state
  useEffect(() => {
    if (!selectedJobId || !jobs[selectedJobId]) return;
    const job = jobs[selectedJobId];
    if (job.status === 'completed') {
      setStableText(job.transcript || '');
      setUnstableText('');
      setStatus('done');
      setMetrics({
        ttft_ms: job.ttft_ms ? parseFloat(job.ttft_ms) : null,
        itl_ms: job.itl_ms ? parseFloat(job.itl_ms) : null,
        rtfx: job.duration_s && job.processing_time_s
          ? parseFloat(job.duration_s) / parseFloat(job.processing_time_s)
          : null,
      });
    } else if (job.status === 'failed') {
      setStableText('');
      setUnstableText('');
      setStatus('error');
      setErrorMsg(job.error || 'Job failed');
    } else if (job.status === 'processing') {
      setUnstableText('Processing…');
    }
  }, [jobs, selectedJobId]);

  // ── Staged files management ─────────────────────────────────────────────
  function handleFilesStaged(newFiles) {
    setStagedFiles((prev) => [...prev, ...newFiles]);
    // Auto-select the first new file if nothing selected
    setSelectedStagedIndex((prev) => prev ?? 0);
  }

  function handleSelectStaged(index) {
    setSelectedStagedIndex(index);
  }

  function handleRemoveStaged(index) {
    setStagedFiles((prev) => prev.filter((_, i) => i !== index));
    setSelectedStagedIndex((prev) => {
      if (prev === null) return null;
      if (prev === index) return null;
      if (prev > index) return prev - 1;
      return prev;
    });
  }

  function handleClearStaged() {
    setStagedFiles([]);
    setSelectedStagedIndex(null);
  }

  async function handleSubmitStaged() {
    if (stagedFiles.length === 0) return;
    const filesToSubmit = [...stagedFiles];
    const success = await handleBatchUpload(filesToSubmit);
    // Clear staged files only after full success; on partial/total failure
    // the staging UI stays visible so the user can retry.
    if (success) {
      setStagedFiles([]);
      setSelectedStagedIndex(null);
    }
  }

  // ── Multi-file upload handler (concurrency pool) ───────────────────────────
  // Upload progress: { queued, total, failed }
  const [uploadProgress, setUploadProgress] = useState(null);

  async function handleBatchUpload(files) {
    if (!files || files.length === 0) return false;

    setStatus('loading');
    setErrorMsg(null);
    setUploadProgress({ queued: 0, total: files.length, failed: 0 });

    const { created, errors } = await submitBatchConcurrent(
      files,
      selectedModel.id,
      selectedModel.engine,
      selectedLanguage,
      SESSION_ID,
      (progress) => setUploadProgress({ ...progress }),
    );

    setUploadProgress(null);

    // Register jobs in local state; the 3s refreshJobs timer handles polling.
    // Avoid calling startPolling per-job — at 2000+ jobs that creates a poll storm.
    for (const job of created) {
      updateJob(job.id, {
        original_filename: job.filename,
        status: 'queued',
        created_at_iso: new Date().toISOString().replace('T', ' ').slice(0, 19),
        created_at: String(Date.now() / 1000),
      });
    }
    // Kick an immediate refresh so the UI shows the new jobs right away
    refreshJobs();

    if (errors.length > 0) {
      setErrorMsg(
        created.length > 0
          ? `${created.length} files queued, ${errors.length} failed: ${errors[0].error}`
          : `Upload failed: ${errors[0].error}`,
      );
      setStatus('error');
      return false;
    }

    // Auto-select the first new job
    if (created.length > 0) {
      setSelectedJobId(created[0].id);
      setStableText('');
      setUnstableText('Queued…');
      setStatus('streaming');
      setErrorMsg(null);
    }
    return true;
  }

  // ── Delete job handler ─────────────────────────────────────────────────────
  async function handleDeleteJob(jobId) {
    try {
      await deleteJob(jobId);
      setJobs((prev) => {
        const next = { ...prev };
        delete next[jobId];
        return next;
      });
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
        setStableText('');
        setUnstableText('');
        setStatus('idle');
      }
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  // ── Re-submit job handler ──────────────────────────────────────────────────
  async function handleResubmitJob(jobId) {
    try {
      const { job_id: newId } = await resubmitJob(jobId);
      const oldJob = jobs[jobId];
      updateJob(newId, {
        original_filename: oldJob?.original_filename || 're-submit',
        status: 'queued',
        created_at_iso: new Date().toISOString().replace('T', ' ').slice(0, 19),
        created_at: String(Date.now() / 1000),
      });
      startPolling(newId);
      setSelectedJobId(newId);
      setStableText('');
      setUnstableText('Queued…');
      setStatus('streaming');
      setErrorMsg(null);
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

  // ── Delete all jobs handler ────────────────────────────────────────────────
  async function handleDeleteAllJobs() {
    try {
      await deleteAllJobs(SESSION_ID);
      setJobs({});
      setSelectedJobId(null);
      setStableText('');
      setUnstableText('');
      setStatus('idle');
    } catch (err) {
      setErrorMsg(String(err));
    }
  }

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
    } else if ('status' in data) {
      // Async queue status (queued / processing)
      setUnstableText(data.status === 'queued' ? 'Queued…' : 'Processing…');
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

  // ── Audio player URL for selected job ────────────────────────────────────
  const selectedJob = selectedJobId ? jobs[selectedJobId] : null;
  const jobAudioUrl = selectedJob?.audio_file
    ? `/api/audio/${encodeURIComponent(selectedJob.audio_file)}`
    : null;

  // ── Audio preview for selected staged file ──────────────────────────────
  const [stagedPreviewUrl, setStagedPreviewUrl] = useState(null);
  useEffect(() => {
    if (selectedStagedIndex !== null && stagedFiles[selectedStagedIndex]) {
      const url = URL.createObjectURL(stagedFiles[selectedStagedIndex]);
      setStagedPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setStagedPreviewUrl(null);
  }, [selectedStagedIndex, stagedFiles]);

  return (
    <div className="app">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-title-row">
          <span className="header-icon">🎙</span>
          <h1 className="header-title">SiliconLanguage Data Plane</h1>
        </div>
        <p className="header-subtitle">
          High-Throughput Asynchronous ASR Pipeline · FastAPI &amp; Valkey Event-Driven Inference
        </p>
      </header>

      <div className="app-layout">
        {/* ── Sidebar: Staged files + Jobs list ─────────────────────── */}
        <aside className="app-sidebar">
          <StagedFiles
            files={stagedFiles}
            selectedIndex={selectedStagedIndex}
            onSelect={handleSelectStaged}
            onRemove={handleRemoveStaged}
            onSubmit={handleSubmitStaged}
            onClear={handleClearStaged}
            disabled={isRunning}
            previewUrl={stagedPreviewUrl}
            uploadProgress={uploadProgress}
          />
          <JobsList jobs={jobs} selectedJobId={selectedJobId} onSelect={handleJobSelect} onDelete={handleDeleteJob} onDeleteAll={handleDeleteAllJobs} onResubmit={handleResubmitJob} />
        </aside>

        {/* ── Main content ────────────────────────────────────────────── */}
        <main className="app-main">
          <div className="card">
            {/* ── Model selection ──────────────────────────────────────── */}
            <ModelSelector value={modelValue} onChange={setModelValue} disabled={isRunning} />

            {/* ── Audio input ──────────────────────────────────────────── */}
            <section className="section">
              <h2 className="section-title">Audio Input</h2>
              <AudioRecorder
                onAudioReady={(file) => { setAudioFile(file); setSelectedLanguage('english'); }}
                onFilesStaged={handleFilesStaged}
                disabled={isBusy}
                externalAudioUrl={audioUrl}
              />
              <SampleClips onSampleSelect={(file, lang) => { setAudioFile(file); if (lang) setSelectedLanguage(lang); }} onLoadingChange={setSampleLoading} disabled={isBusy} audioFile={audioFile} />
            </section>

            {/* ── Action buttons ───────────────────────────────────────── */}
            <div className="action-row">
              <button
                className="btn btn-primary btn-large"
                onClick={handleSubmit}
                disabled={isBusy || !audioFile}
              >
                {status === 'loading' ? (
                  <>
                    <span className="spinner-inline" /> Loading model…
                  </>
                ) : status === 'streaming' ? (
                  <>
                    <span className="spinner-inline" /> Transcribing…
                  </>
                ) : hasActiveJobs ? (
                  '⏳ Jobs in progress…'
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

          {/* ── Error banner ───────────────────────────────────────────── */}
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


          {/* ── Audio player for selected job ──────────────────────────── */}
          {jobAudioUrl && selectedJob?.status === 'completed' && (
            <div className="card">
              <h3 className="section-title">Playback</h3>
              <audio className="audio-player" controls src={jobAudioUrl} key={jobAudioUrl} />
            </div>
          )}

          {/* ── Metrics dashboard ──────────────────────────────────────── */}
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
      </div>

      <footer className="app-footer">
        <div className="footer-inner">
          <a
            href="https://siliconlanguage.com"
            target="_blank"
            rel="noopener noreferrer"
            className="footer-logo-link"
          >
            <img src="/logo-lockup.png" alt="Silicon Language" className="footer-logo" />
          </a>
          <p>
            SiliconLanguage Data Plane · Decoupled Queue-Driven Processing
          </p>
        </div>
      </footer>
    </div>
  );
}
