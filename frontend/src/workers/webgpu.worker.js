/**
 * webgpu.worker.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Runs entirely inside a WebWorker context.
 * Responsibilities:
 *   1. Load a transformers.js Whisper pipeline (WebGPU backend).
 *   2. Cache ONNX model weights in the browser via the Cache API for fast
 *      subsequent cold-starts.
 *   3. Implement a "LocalAgreement-2" style streaming stabilisation algorithm
 *      that withholds partial hypotheses until at least two consecutive
 *      decoding passes agree on a prefix, preventing jittery live output.
 *
 * Message protocol (worker ← main thread):
 *   { type: 'load',       model: string }
 *   { type: 'transcribe', audio: Float32Array, id: string }
 *   { type: 'abort',      id: string }
 *
 * Message protocol (worker → main thread):
 *   { type: 'ready' }
 *   { type: 'progress',   id, text: string, stable: bool }
 *   { type: 'result',     id, transcript: string, ttft_ms, itl_ms, rtfx }
 *   { type: 'error',      id, message: string }
 */

import { pipeline, env } from '@xenova/transformers';

// ── Cache configuration ───────────────────────────────────────────────────────
// Point transformers.js at the Cache Storage API so ONNX weights survive
// page reloads without re-downloading from the CDN.
env.useBrowserCache = true;
env.allowLocalModels = false;

// ── State ─────────────────────────────────────────────────────────────────────
/** @type {import('@xenova/transformers').Pipeline | null} */
let asr = null;
let loadedModelId = null;
const abortedIds = new Set();

// ── LocalAgreement-2 helpers ──────────────────────────────────────────────────
/**
 * Returns the longest common prefix (token-level) shared by *a* and *b*.
 * @param {string[]} a
 * @param {string[]} b
 * @returns {string[]}
 */
function longestCommonPrefix(a, b) {
  const len = Math.min(a.length, b.length);
  let i = 0;
  while (i < len && a[i] === b[i]) i++;
  return a.slice(0, i);
}

/**
 * LocalAgreement-2 stabiliser.
 *
 * Keeps a sliding window of the last two hypotheses.  Only the prefix
 * agreed upon by *both* is emitted as "stable" text.  The remainder is
 * surfaced as an unstable suffix so the UI can render it in a dimmed style.
 */
class LocalAgreement2 {
  constructor() {
    this.prev = /** @type {string[]} */ ([]);
    this.stableTokens = /** @type {string[]} */ ([]);
  }

  /**
   * Feed in the latest hypothesis tokens.
   * @param {string[]} current  Current hypothesis split into tokens
   * @returns {{ stable: string, unstable: string }}
   */
  update(current) {
    const agreed = longestCommonPrefix(this.prev, current);
    // Extend the stable set with newly confirmed tokens
    if (agreed.length > this.stableTokens.length) {
      this.stableTokens = agreed;
    }
    this.prev = current;
    const stable = this.stableTokens.join('');
    const unstable = current.slice(this.stableTokens.length).join('');
    return { stable, unstable };
  }

  reset() {
    this.prev = [];
    this.stableTokens = [];
  }
}

// ── Model loading ─────────────────────────────────────────────────────────────
async function loadModel(modelId) {
  if (asr && loadedModelId === modelId) {
    self.postMessage({ type: 'ready' });
    return;
  }
  try {
    self.postMessage({ type: 'loading', modelId });
    asr = await pipeline('automatic-speech-recognition', modelId, {
      // Request WebGPU execution provider; falls back to WASM if unavailable
      device: 'webgpu',
      dtype: 'fp16',
    });
    loadedModelId = modelId;
    self.postMessage({ type: 'ready' });
  } catch (err) {
    self.postMessage({ type: 'error', id: null, message: String(err) });
  }
}

// ── Transcription ─────────────────────────────────────────────────────────────
async function transcribe(audio, id) {
  if (!asr) {
    self.postMessage({ type: 'error', id, message: 'Model not loaded. Send a load message first.' });
    return;
  }

  const la2 = new LocalAgreement2();
  const t0 = performance.now();
  let firstTokenTime = null;
  const tokenTimestamps = [];

  try {
    // transformers.js callback_function fires after each generated token
    const result = await asr(audio, {
      return_timestamps: true,
      chunk_length_s: 30,
      stride_length_s: 5,
      callback_function: (beams) => {
        if (abortedIds.has(id)) throw new Error('ABORTED');

        const now = performance.now();
        if (firstTokenTime === null) firstTokenTime = now;
        tokenTimestamps.push(now);

        // Extract hypothesis text from the first beam
        const hypothesis = beams[0]?.output_token_ids ?? [];
        const decoded = asr.tokenizer
          ? asr.tokenizer.decode(hypothesis, { skip_special_tokens: true })
          : (beams[0]?.text ?? '');

        // Split on whitespace boundaries, preserving leading space within each word
        // to allow correct reconstruction. Use non-capturing split to avoid empty strings.
        const tokens = decoded.match(/\S+\s*/g) ?? [];
        const { stable, unstable } = la2.update(tokens);

        self.postMessage({ type: 'progress', id, stable, unstable });
      },
    });

    if (abortedIds.has(id)) {
      abortedIds.delete(id);
      return;
    }

    const t1 = performance.now();
    const totalMs = t1 - t0;
    const audioDurationS = audio.length / 16000;
    const ttftMs = firstTokenTime !== null ? firstTokenTime - t0 : 0;
    let itlMs = 0;
    if (tokenTimestamps.length > 1) {
      const gaps = tokenTimestamps
        .slice(1)
        .map((ts, i) => ts - tokenTimestamps[i]);
      itlMs = gaps.reduce((s, g) => s + g, 0) / gaps.length;
    }
    const rtfx = audioDurationS / (totalMs / 1000);

    self.postMessage({
      type: 'result',
      id,
      transcript: typeof result.text === 'string' ? result.text : result.text ?? '',
      ttft_ms: Math.round(ttftMs * 100) / 100,
      itl_ms: Math.round(itlMs * 100) / 100,
      rtfx: Math.round(rtfx * 10000) / 10000,
    });
  } catch (err) {
    if (String(err).includes('ABORTED')) {
      abortedIds.delete(id);
    } else {
      self.postMessage({ type: 'error', id, message: String(err) });
    }
  }
}

// ── Message handler ───────────────────────────────────────────────────────────
self.addEventListener('message', async (event) => {
  const { type, model, audio, id } = event.data;
  switch (type) {
    case 'load':
      await loadModel(model);
      break;
    case 'transcribe':
      await transcribe(audio, id);
      break;
    case 'abort':
      abortedIds.add(id);
      break;
    default:
      self.postMessage({ type: 'error', id: null, message: `Unknown message type: ${type}` });
  }
});
