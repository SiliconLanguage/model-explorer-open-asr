/**
 * inferenceRouter.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Strategy-based routing:
 *   - Model name contains "WebGPU" → client-side via WebWorker (transformers.js)
 *   - All other models             → server-side via FastAPI / vLLM
 *
 * Exports:
 *   route(model, audioBuffer, onProgress) → Promise<TranscribeResult>
 *
 * TranscribeResult = { transcript, ttft_ms, itl_ms, rtfx, mode }
 */

// ── WebGPU / WebWorker client ─────────────────────────────────────────────────

let workerInstance = null;
let workerReadyResolve = null;
let workerReadyReject = null;  // companion reject for the load promise
let workerLoadedModelId = null; // mirrors worker's loadedModelId for fast cache hit
const pendingCallbacks = new Map(); // id → { resolve, reject, onProgress }

function getWorker() {
  if (workerInstance) return workerInstance;

  // Vite's ?worker syntax creates an ES-module Worker automatically
  workerInstance = new Worker(
    new URL('../workers/webgpu.worker.js', import.meta.url),
    { type: 'module' }
  );

  workerInstance.addEventListener('message', (event) => {
    const { type, id, stable, unstable, transcript, ttft_ms, itl_ms, rtfx, message } = event.data;

    // ── Load lifecycle messages ─────────────────────────────────────────────
    if (type === 'ready') {
      workerLoadedModelId = event.data.modelId ?? workerLoadedModelId;
      if (workerReadyResolve) {
        const res = workerReadyResolve;
        workerReadyResolve = null;
        workerReadyReject = null;
        res();
      }
      return;
    }

    // A null id means the error came from loadModel(), not a transcription.
    if (type === 'error' && id == null) {
      const rej = workerReadyReject;
      workerReadyResolve = null;
      workerReadyReject = null;
      rej?.(new Error(message ?? 'Model failed to load'));
      return;
    }

    // ── Transcription lifecycle messages ────────────────────────────────────
    const cb = pendingCallbacks.get(id);
    if (!cb) return;

    switch (type) {
      case 'progress':
        cb.onProgress?.({ stable, unstable });
        break;
      case 'result':
        pendingCallbacks.delete(id);
        cb.resolve({ transcript, ttft_ms, itl_ms, rtfx, mode: 'webgpu' });
        break;
      case 'error':
        pendingCallbacks.delete(id);
        cb.reject(new Error(message));
        break;
    }
  });

  return workerInstance;
}

async function ensureModelLoaded(modelId) {
  // Fast path: model already cached in worker — skip the message round-trip
  if (workerLoadedModelId === modelId) return;
  const worker = getWorker();
  return new Promise((resolve, reject) => {
    workerReadyResolve = resolve;
    workerReadyReject = reject;
    worker.postMessage({ type: 'load', model: modelId });
  });
}

async function routeWebGPU(modelId, audioBuffer, onProgress, language) {
  await ensureModelLoaded(modelId);
  const worker = getWorker();
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    pendingCallbacks.set(id, { resolve, reject, onProgress });
    worker.postMessage({ type: 'transcribe', audio: audioBuffer, id, language }, [audioBuffer.buffer]);
  });
}

// ── Server-side / vLLM client ─────────────────────────────────────────────────

async function routeServer(modelId, audioFile, onProgress, engine, language) {
  const form = new FormData();
  form.append('audio', audioFile);
  form.append('model', modelId);
  if (engine) {
    form.append('engine', engine);
  }
  if (language) {
    form.append('language', language);
  }

  // Use streaming endpoint for token-by-token delivery
  const response = await fetch('/api/transcribe/stream', {
    method: 'POST',
    body: form,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail ?? `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult = null;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse Server-Sent Events
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';

      for (const event of events) {
        const dataLine = event.split('\n').find((l) => l.startsWith('data: '));
        if (!dataLine) continue;
        try {
          const payload = JSON.parse(dataLine.slice(6));
          if (payload.done) {
            finalResult = payload;
          } else if (payload.token) {
            onProgress?.({ token: payload.token, ttft_ms: payload.ttft_ms });
          } else if (payload.ttft_ms != null) {
            onProgress?.({ ttft_ms: payload.ttft_ms });
          }
        } catch {
          // ignore malformed SSE frames
        }
      }
    }
  } finally {
    // Always release the reader so the underlying HTTP connection can be
    // returned to the pool.  This prevents stream leaks when an exception
    // is thrown mid-stream (e.g. network error or component unmount).
    reader.cancel();
  }

  if (!finalResult) throw new Error('Stream ended without a final result.');

  return {
    transcript: finalResult.transcript,
    ttft_ms: finalResult.ttft_ms,
    itl_ms: finalResult.itl_ms,
    rtfx: finalResult.rtfx,
    mode: 'server',
  };
}

// ── Server-side async (Valkey queue + polling) ────────────────────────────────

const POLL_INTERVAL_MS = 2000;

async function routeAsync(modelId, audioFile, onProgress, engine, language) {
  const form = new FormData();
  form.append('audio', audioFile);
  form.append('model', modelId);
  if (engine) form.append('engine', engine);
  if (language) form.append('language', language);
  // Include session_id so this job appears in the session-scoped list
  const sid = typeof sessionStorage !== 'undefined' ? sessionStorage.getItem('asr_session_id') : '';
  if (sid) form.append('session_id', sid);

  const resp = await fetch('/api/transcribe/async', { method: 'POST', body: form });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `HTTP ${resp.status}`);
  }

  const { job_id } = await resp.json();
  console.info(`[inference] async job ${job_id} accepted (model=${modelId})`);
  onProgress?.({ status: 'queued' });

  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const statusResp = await fetch(`/api/jobs/${encodeURIComponent(job_id)}`);
        if (!statusResp.ok) {
          reject(new Error(`Job status check failed: HTTP ${statusResp.status}`));
          return;
        }
        const data = await statusResp.json();

        if (data.status === 'completed') {
          console.info(`[inference] job ${job_id} completed`);
          resolve({
            transcript: data.transcript || '',
            ttft_ms: data.ttft_ms ? parseFloat(data.ttft_ms) : null,
            itl_ms: data.itl_ms ? parseFloat(data.itl_ms) : null,
            rtfx: data.rtfx ? parseFloat(data.rtfx) : null,
            mode: 'server-async',
          });
        } else if (data.status === 'failed') {
          console.error(`[inference] job ${job_id} failed:`, data.error);
          reject(new Error(data.error || 'Transcription failed'));
        } else {
          onProgress?.({ status: data.status });
          setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        reject(err);
      }
    };
    poll();
  });
}

// ── Public router ─────────────────────────────────────────────────────────────

/**
 * Route transcription to the appropriate execution mode.
 *
 * @param {string}   modelId      Selected model name
 * @param {File|Float32Array} audio  Audio source
 * @param {function} [onProgress] Called with partial results during streaming
 * @returns {Promise<{transcript:string, ttft_ms:number, itl_ms:number, rtfx:number, mode:string}>}
 */
export async function route(modelId, audio, onProgress, engine = undefined, mode = 'server', language = undefined) {
  if (mode === 'webgpu') {
    // Ensure we have a Float32Array of 16 kHz samples
    const audioBuffer = audio instanceof Float32Array ? audio : await fileToFloat32(audio);
    return routeWebGPU(modelId, audioBuffer, onProgress, language);
  }

  // Server-side: try async queue first, fall back to sync SSE if unavailable
  try {
    return await routeAsync(modelId, audio, onProgress, engine, language);
  } catch (err) {
    if (err.message?.includes('Queue unavailable') || err.message?.includes('503')) {
      return routeServer(modelId, audio, onProgress, engine, language);
    }
    throw err;
  }
}

// ── Helper: decode File → Float32Array at 16 kHz ──────────────────────────────

async function fileToFloat32(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);
  return decoded.getChannelData(0); // mono, 16 kHz
}

// ── Batch upload + job polling (multi-file) ───────────────────────────────────

import pLimit from 'p-limit';

/**
 * Submit a small group of files as a batch for async transcription.
 * Returns an array of { id, filename } for each queued job.
 */
export async function submitBatch(files, modelId, engine, language, sessionId) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  form.append('model', modelId);
  if (engine) form.append('engine', engine);
  if (language) form.append('language', language);
  if (sessionId) form.append('session_id', sessionId);

  const resp = await fetch('/api/transcribe/batch', { method: 'POST', body: form });
  if (!resp.ok) {
    if (resp.status === 413) {
      console.error(`[batch] upload too large: ${files.length} files`);
      throw new Error(`Upload too large (${files.length} files). Try uploading in smaller batches.`);
    }
    const err = await resp.json().catch(() => ({ detail: resp.statusText || `HTTP ${resp.status}` }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  const { jobs } = await resp.json();
  console.info(`[batch] ${jobs.length} jobs queued`);
  return jobs; // [{ id, filename }, ...]
}

/**
 * Upload many files using a concurrency-limited pool.
 * Files are grouped into small batches (MICRO_BATCH) and up to CONCURRENCY
 * requests fly in parallel.  Calls onProgress({ queued, total, failed })
 * after each micro-batch completes or fails.
 *
 * Returns { created: [...jobs], errors: [...{ file, error }] }.
 */
const CONCURRENCY = 6;   // parallel HTTP requests (≤ browser per-origin limit)
const MICRO_BATCH = 5;   // files per request (keeps each body small)
const INTER_BATCH_DELAY_MS = 50; // let the event loop + backend breathe
const MAX_RETRIES = 2;   // retry transient network failures

async function submitBatchWithRetry(batch, modelId, engine, language, sessionId) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await submitBatch(batch, modelId, engine, language, sessionId);
    } catch (err) {
      lastErr = err;
      // Don't retry 413 (payload too large) or 422 (validation)
      if (err.message?.includes('413') || err.message?.includes('422')) throw err;
      if (attempt < MAX_RETRIES) {
        console.warn(`[batch] retry ${attempt + 1}/${MAX_RETRIES} after: ${err.message}`);
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

export async function submitBatchConcurrent(
  files, modelId, engine, language, sessionId, onProgress,
) {
  const limit = pLimit(CONCURRENCY);

  // Split files into micro-batches
  const microBatches = [];
  for (let i = 0; i < files.length; i += MICRO_BATCH) {
    microBatches.push(files.slice(i, i + MICRO_BATCH));
  }

  let queued = 0;
  let failed = 0;
  const allCreated = [];
  const allErrors = [];

  const tasks = microBatches.map((batch) =>
    limit(async () => {
      // Small delay between slots to avoid overwhelming the connection pool
      await new Promise((r) => setTimeout(r, INTER_BATCH_DELAY_MS));
      try {
        const jobs = await submitBatchWithRetry(batch, modelId, engine, language, sessionId);
        queued += jobs.length;
        allCreated.push(...jobs);
      } catch (err) {
        failed += batch.length;
        allErrors.push(...batch.map((f) => ({ file: f.name, error: err })));
      }
      onProgress?.({ queued, total: files.length, failed });
    }),
  );

  await Promise.all(tasks);
  console.info(`[batch-concurrent] done: ${allCreated.length} queued, ${allErrors.length} errors out of ${files.length} files`);
  return { created: allCreated, errors: allErrors };
}

/**
 * Poll a single job until it reaches a terminal state.
 * Calls onUpdate(jobData) on every poll tick.
 * Returns the final job data on completion, rejects on failure.
 */
export function pollJob(jobId, onUpdate) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
        if (!resp.ok) { reject(new Error(`HTTP ${resp.status}`)); return; }
        const data = await resp.json();
        onUpdate?.(data);
        if (data.status === 'completed') {
          console.info(`[poll] job ${jobId} completed`);
          resolve(data);
        }
        else if (data.status === 'failed') {
          console.error(`[poll] job ${jobId} failed:`, data.error);
          reject(new Error(data.error || 'Job failed'));
        }
        else setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) { reject(err); }
    };
    tick();
  });
}

/**
 * Fetch all existing jobs from Valkey, scoped by session.
 */
export async function fetchAllJobs(sessionId) {
  const params = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  const resp = await fetch(`/api/jobs${params}`);
  if (!resp.ok) return {};
  const { jobs } = await resp.json();
  return jobs; // { jobId: { status, transcript, ... }, ... }
}

/**
 * Delete a job and its audio file.
 */
export async function deleteJob(jobId) {
  const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * Delete all jobs, optionally scoped by session.
 */
export async function deleteAllJobs(sessionId) {
  const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  const resp = await fetch(`/api/jobs${qs}`, { method: 'DELETE' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * Re-submit a job using its existing audio file.
 * Returns { job_id, status }.
 */
export async function resubmitJob(jobId) {
  const resp = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/resubmit`, { method: 'POST' });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail ?? `HTTP ${resp.status}`);
  }
  return resp.json();
}
