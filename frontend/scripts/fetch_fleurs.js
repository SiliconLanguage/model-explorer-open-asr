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

const outDir = path.join(path.dirname(new URL(import.meta.url).pathname), '../public/samples');
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const FLEURS_CONFIGS = {
  english: 'en_us',
  chinese: 'cmn_hans_cn',
  spanish: 'es_419',
};

const FALLBACK_SOURCES = {
  english: { dataset: 'MLCommons/peoples_speech', config: 'clean', split: 'train' },
  chinese: { dataset: 'PolyAI/minds14', config: 'zh-CN', split: 'train' },
  spanish: { dataset: 'facebook/multilingual_librispeech', config: 'spanish', split: 'test' },
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

async function fetchSamples() {
  for (const [name, fleursConfig] of Object.entries(FLEURS_CONFIGS)) {
    const outPath = path.join(outDir, `${name}.wav`);
    console.log(`\n[${name}.wav] Trying google/fleurs (${fleursConfig})...`);

    try {
      const buf = await fetchFromDatasetServer('google/fleurs', fleursConfig, 'validation');
      fs.writeFileSync(outPath, buf);
      console.log(`  Done: google/fleurs (${buf.length} bytes)`);
      continue;
    } catch (err) {
      console.warn(`  FLEURS unavailable: ${err.message}`);
    }

    const fb = FALLBACK_SOURCES[name];
    console.log(`  Trying fallback: ${fb.dataset} (${fb.config})...`);
    try {
      const buf = await fetchFromDatasetServer(fb.dataset, fb.config, fb.split);
      fs.writeFileSync(outPath, buf);
      console.log(`  Done: ${fb.dataset} (${buf.length} bytes)`);
    } catch (err) {
      console.error(`  Fallback also failed: ${err.message}`);
      process.exit(1);
    }
  }

  console.log('\nAll samples saved to public/samples/');
}

fetchSamples();
