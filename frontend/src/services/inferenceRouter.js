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

async function routeWebGPU(modelId, audioBuffer, onProgress) {
  await ensureModelLoaded(modelId);
  const worker = getWorker();
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    pendingCallbacks.set(id, { resolve, reject, onProgress });
    worker.postMessage({ type: 'transcribe', audio: audioBuffer, id }, [audioBuffer.buffer]);
  });
}

// ── Server-side / vLLM client ─────────────────────────────────────────────────

async function routeServer(modelId, audioFile, onProgress, engine) {
  const form = new FormData();
  form.append('audio', audioFile);
  form.append('model', modelId);
  if (engine) {
    form.append('engine', engine);
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

// ── Public router ─────────────────────────────────────────────────────────────

/**
 * Route transcription to the appropriate execution mode.
 *
 * @param {string}   modelId      Selected model name
 * @param {File|Float32Array} audio  Audio source
 * @param {function} [onProgress] Called with partial results during streaming
 * @returns {Promise<{transcript:string, ttft_ms:number, itl_ms:number, rtfx:number, mode:string}>}
 */
export async function route(modelId, audio, onProgress, engine = undefined, mode = 'server') {
  if (mode === 'webgpu') {
    // Ensure we have a Float32Array of 16 kHz samples
    const audioBuffer = audio instanceof Float32Array ? audio : await fileToFloat32(audio);
    return routeWebGPU(modelId, audioBuffer, onProgress);
  }
  // Server-side: pass the raw File for multipart upload
  return routeServer(modelId, audio, onProgress, engine);
}

// ── Helper: decode File → Float32Array at 16 kHz ──────────────────────────────

async function fileToFloat32(file) {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);
  return decoded.getChannelData(0); // mono, 16 kHz
}
