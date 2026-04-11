/**
 * fetch_fleurs.js
 * Downloads real human speech samples from Hugging Face Datasets-Server API.
 *
 * Primary source: google/fleurs (CC-BY-4.0)
 * Fallback sources when FLEURS API is unavailable:
 *   - English:  MLCommons/peoples_speech (CC-BY-SA 4.0)
 *   - Chinese:  PolyAI/minds14 zh-CN (CC-BY-4.0)
 *   - Spanish:  facebook/multilingual_librispeech (CC-BY-4.0)
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const outDir = path.join(path.dirname(new URL(import.meta.url).pathname), '../public/samples');
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const FLEURS_CONFIGS = {
  english: 'en_us',
  chinese: 'cmn_hans_cn',
  spanish: 'es_419',
  french: 'fr_fr',
  german: 'de_de',
  italian: 'it_it',
  japanese: 'ja_jp',
  hindi: 'hi_in',
};

const FALLBACK_SOURCES = {
  english: { dataset: 'MLCommons/peoples_speech', config: 'clean', split: 'train' },
  chinese: { dataset: 'PolyAI/minds14', config: 'zh-CN', split: 'train' },
  spanish: { dataset: 'facebook/multilingual_librispeech', config: 'spanish', split: 'test' },
  french: { dataset: 'facebook/multilingual_librispeech', config: 'french', split: 'test' },
  german: { dataset: 'facebook/multilingual_librispeech', config: 'german', split: 'test' },
  italian: { dataset: 'facebook/multilingual_librispeech', config: 'italian', split: 'test' },
  japanese: { dataset: 'PolyAI/minds14', config: 'ja-JP', split: 'train' },
  hindi: { dataset: 'PolyAI/minds14', config: 'hi-IN', split: 'train' },
};

async function fetchFromDatasetServer(dataset, config, split) {
  const apiUrl = `https://datasets-server.huggingface.co/first-rows?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}`;
  const apiRes = await fetch(apiUrl);
  if (!apiRes.ok) throw new Error(`API returned ${apiRes.status}`);
  const data = await apiRes.json();
  if (!data.rows || data.rows.length === 0) throw new Error('No rows returned');

  let audio = data.rows[0].row.audio;
  if (Array.isArray(audio)) audio = audio[0];
  const audioUrl = audio?.src;
  if (!audioUrl) throw new Error('No audio.src in response');

  const audioRes = await fetch(audioUrl);
  if (!audioRes.ok) throw new Error(`Audio download returned ${audioRes.status}`);
  return Buffer.from(await audioRes.arrayBuffer());
}

/**
 * If the downloaded file is not WAV (e.g. OGG from FLEURS/MLS), convert it
 * to 16 kHz mono WAV using ffmpeg so the browser <audio> element can always
 * play it without MIME-type mismatches.
 */
function ensureWav(filePath) {
  const header = Buffer.alloc(4);
  const fd = fs.openSync(filePath, 'r');
  fs.readSync(fd, header, 0, 4, 0);
  fs.closeSync(fd);
  if (header.toString('ascii', 0, 4) === 'RIFF') return; // already WAV

  console.log('  Converting to WAV (16 kHz mono)...');
  const tmp = filePath + '.tmp.wav';
  execFileSync('ffmpeg', ['-y', '-i', filePath, '-ar', '16000', '-ac', '1', '-f', 'wav', tmp], { stdio: 'ignore' });
  fs.renameSync(tmp, filePath);
}

async function fetchSamples() {
  for (const [name, fleursConfig] of Object.entries(FLEURS_CONFIGS)) {
    const outPath = path.join(outDir, `${name}.wav`);
    console.log(`\n[${name}.wav] Trying google/fleurs (${fleursConfig})...`);

    try {
      const buf = await fetchFromDatasetServer('google/fleurs', fleursConfig, 'validation');
      fs.writeFileSync(outPath, buf);
      ensureWav(outPath);
      console.log(`  Done: google/fleurs (${fs.statSync(outPath).size} bytes)`);
      continue;
    } catch (err) {
      console.warn(`  FLEURS unavailable: ${err.message}`);
    }

    const fb = FALLBACK_SOURCES[name];
    console.log(`  Trying fallback: ${fb.dataset} (${fb.config})...`);
    try {
      const buf = await fetchFromDatasetServer(fb.dataset, fb.config, fb.split);
      fs.writeFileSync(outPath, buf);
      ensureWav(outPath);
      console.log(`  Done: ${fb.dataset} (${fs.statSync(outPath).size} bytes)`);
    } catch (err) {
      console.error(`  Fallback also failed: ${err.message}`);
      process.exit(1);
    }
  }

  console.log('\nAll samples saved to public/samples/');
}

fetchSamples();
