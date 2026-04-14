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

import { pipeline, env } from '@huggingface/transformers';

// ── Cache configuration ───────────────────────────────────────────────────────
// Point transformers.js at the Cache Storage API so ONNX weights survive
// page reloads without re-downloading from the CDN.
// useBrowserCache relies on the Cache Storage API which is only available in
// secure contexts (HTTPS or localhost); it is a no-op if the API is absent.
env.useBrowserCache = typeof caches !== 'undefined';
env.allowLocalModels = false;

// ── State ─────────────────────────────────────────────────────────────────────
/** @type {import('@huggingface/transformers').Pipeline | null} */
let asr = null;
let loadedModelId = null;
const abortedIds = new Set();

const LANGUAGE_ISO_MAP = {
  english: 'en',
  chinese: 'zh',
  spanish: 'es',
  french: 'fr',
  german: 'de',
  italian: 'it',
  japanese: 'ja',
  hindi: 'hi',
};

// ── LocalAgreement-2 helpers ──────────────────────────────────────────────────
/**
 * Returns the longest common prefix (token-level) shared by *a* and *b*.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number[]}
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
    this.prev = /** @type {number[]} */ ([]);
    this.stableTokenIds = /** @type {number[]} */ ([]);
  }

  /**
   * Feed in the latest hypothesis token IDs.
   * @param {number[]} current  Current hypothesis token IDs
   * @returns {{ stableTokenIds: number[], unstableTokenIds: number[] }}
   */
  update(current) {
    const agreed = longestCommonPrefix(this.prev, current);
    // Extend the stable set with newly confirmed tokens, but never claim more
    // stable tokens than actually exist in the current hypothesis.  Capping at
    // current.length prevents stale over-commitment when the model produces a
    // shorter hypothesis (e.g. at the start of a new audio chunk), which would
    // otherwise show confirmed text that is no longer in the active hypothesis.
    const newStableLen = Math.min(
      Math.max(agreed.length, this.stableTokenIds.length),
      current.length,
    );
    this.stableTokenIds = current.slice(0, newStableLen);
    this.prev = current;
    const stableTokenIds = this.stableTokenIds;
    const unstableTokenIds = current.slice(this.stableTokenIds.length);
    return { stableTokenIds, unstableTokenIds };
  }

  reset() {
    this.prev = [];
    this.stableTokenIds = [];
  }
}

// ── Model loading ─────────────────────────────────────────────────────────────
async function loadModel(modelId) {
  if (asr && loadedModelId === modelId) {
    self.postMessage({ type: 'ready', modelId });
    return;
  }
  try {
    // Dispose of the previous pipeline's ONNX sessions before loading a new
    // model.  Without this, ONNX Runtime tries to .destroy() GPU buffers that
    // were already garbage-collected, throwing "Cannot read properties of
    // undefined (reading 'destroy')".
    if (asr) {
      try { await asr.dispose(); } catch { /* best-effort cleanup */ }
      asr = null;
      loadedModelId = null;
    }

    self.postMessage({ type: 'loading', modelId });
    // Dtype selection:
    // - Cohere 2B: q4 (int4 MatMulNBits).  fp16 has 2.5 GB shards that
    //   exceed browser memory; q4f16 uses GatherBlockQuantized which needs
    //   shader-f16 and crashes on most GPUs.
    // - Whisper:    fp16 (small models, ~60-120 MB, widely supported).
    const isCohere = modelId.toLowerCase().includes('cohere');
    const gpuDtype = isCohere ? 'q4' : 'fp16';
    try {
      asr = await pipeline('automatic-speech-recognition', modelId, {
        device: 'webgpu',
        dtype: gpuDtype,
      });
    } catch {
      // WebGPU unavailable (non-secure context / HTTP, old browser, no
      // GPU, or unsupported ONNX ops) — fall back to WASM.
      self.postMessage({ type: 'loading', modelId, fallback: 'wasm' });
      asr = await pipeline('automatic-speech-recognition', modelId, {
        device: 'wasm',
        dtype: isCohere ? 'q4' : 'fp32',
      });
    }
    loadedModelId = modelId;
    self.postMessage({ type: 'ready', modelId });
  } catch (err) {
    // Reset state so a subsequent load attempt can retry cleanly.
    asr = null;
    loadedModelId = null;
    self.postMessage({ type: 'error', id: null, message: String(err) });
  }
}

// ── Transcription ─────────────────────────────────────────────────────────────
async function transcribe(audio, id, language) {
  if (!asr) {
    self.postMessage({ type: 'error', id, message: 'Model not loaded. Send a load message first.' });
    return;
  }

  const isoLang = language ? (LANGUAGE_ISO_MAP[language.toLowerCase()] || 'en') : undefined;
  const la2 = new LocalAgreement2();
  const t0 = performance.now();
  let firstTokenTime = null;
  const tokenTimestamps = [];

  try {
    // transformers.js callback_function fires after each generated token
    // Whisper models support chunked long-form decoding; Cohere uses its own
    // encoder and these params cause early truncation.
    const isWhisper = (loadedModelId ?? '').toLowerCase().includes('whisper');

    // Non-Whisper models (e.g. Cohere ONNX) default to max_length=20 in
    // GenerationConfig, truncating transcripts to ~10 new tokens.  Force a
    // generous limit via BOTH max_new_tokens AND generation_config to ensure
    // at least one path overrides the default.
    const cohereGenerationOverrides = !isWhisper
      ? { max_new_tokens: 512, max_length: 1024, generation_config: { max_new_tokens: 512, max_length: 1024 } }
      : {};

    const result = await asr(audio, {
      task: 'transcribe',
      ...(isoLang ? { language: isoLang } : {}),
      // Whisper: chunked long-form decoding with timestamp grounding.
      // return_timestamps forces the attention mechanism to anchor to audio
      // frames, naturally cutting off silence loops without draconian n-gram
      // penalties that truncate valid speech.
      ...(isWhisper ? {
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
        repetition_penalty: 1.05,
        condition_on_prev_tokens: false,
      } : {}),
      ...cohereGenerationOverrides,
      callback_function: (beams) => {
        if (abortedIds.has(id)) throw new Error('ABORTED');

        const now = performance.now();
        if (firstTokenTime === null) firstTokenTime = now;
        tokenTimestamps.push(now);

        // Wrap in try/catch — the WhisperTextStreamer in transformers.js can
        // throw "token_ids must be a non-empty array of integers" when
        // timestamp tokens are emitted.  Swallowing the error here lets the
        // pipeline complete; the final bulk result is still accurate.
        try {
          const beamText = beams[0]?.text ?? '';
          /** @type {number[]} */
          let tokenIds = [];

          if (asr.tokenizer?.encode) {
            const encoded = asr.tokenizer.encode(beamText);
            if (Array.isArray(encoded)) {
              tokenIds = encoded.filter((v) => Number.isFinite(v));
            } else if (Array.isArray(encoded?.ids)) {
              tokenIds = encoded.ids.filter((v) => Number.isFinite(v));
            } else if (Array.isArray(encoded?.input_ids)) {
              tokenIds = encoded.input_ids.filter((v) => Number.isFinite(v));
            } else if (Array.isArray(encoded?.[0])) {
              tokenIds = encoded[0].filter((v) => Number.isFinite(v));
            }
          }

          // Fallback to beam token IDs when encode() is unavailable or returns empty.
          if (tokenIds.length === 0) {
            tokenIds = (beams[0]?.output_token_ids ?? []).filter((v) => Number.isFinite(v));
          }

          if (tokenIds.length === 0) return; // nothing to consensus on yet

          const { stableTokenIds, unstableTokenIds } = la2.update(tokenIds);
          const stable = asr.tokenizer?.decode
            ? asr.tokenizer.decode(stableTokenIds, { skip_special_tokens: true })
            : '';
          const unstable = asr.tokenizer?.decode
            ? asr.tokenizer.decode(unstableTokenIds, { skip_special_tokens: true })
            : '';

          self.postMessage({ type: 'progress', id, stable, unstable });
        } catch (cbErr) {
          // Silently skip this streaming tick; the final result is unaffected.
        }
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
  const { type, model, audio, id, language } = event.data;
  switch (type) {
    case 'load':
      await loadModel(model);
      break;
    case 'transcribe':
      await transcribe(audio, id, language);
      break;
    case 'abort':
      abortedIds.add(id);
      break;
    default:
      self.postMessage({ type: 'error', id: null, message: `Unknown message type: ${type}` });
  }
});
