import fs from 'node:fs';
import path from 'node:path';

const sampleRate = 16000;
const durationSeconds = 1;
const frequency = 440;
const amplitude = 0.3;
const channels = 1;
const bitsPerSample = 16;
const numSamples = sampleRate * durationSeconds;
const bytesPerSample = bitsPerSample / 8;
const dataSize = numSamples * channels * bytesPerSample;
const fileSize = 44 + dataSize;

const outPath = path.resolve(process.cwd(), 'tests/fixtures/test.wav');
const buffer = Buffer.alloc(fileSize);

let offset = 0;
buffer.write('RIFF', offset); offset += 4;
buffer.writeUInt32LE(fileSize - 8, offset); offset += 4;
buffer.write('WAVE', offset); offset += 4;

buffer.write('fmt ', offset); offset += 4;
buffer.writeUInt32LE(16, offset); offset += 4;
buffer.writeUInt16LE(1, offset); offset += 2;
buffer.writeUInt16LE(channels, offset); offset += 2;
buffer.writeUInt32LE(sampleRate, offset); offset += 4;
buffer.writeUInt32LE(sampleRate * channels * bytesPerSample, offset); offset += 4;
buffer.writeUInt16LE(channels * bytesPerSample, offset); offset += 2;
buffer.writeUInt16LE(bitsPerSample, offset); offset += 2;

buffer.write('data', offset); offset += 4;
buffer.writeUInt32LE(dataSize, offset); offset += 4;

for (let i = 0; i < numSamples; i += 1) {
  const t = i / sampleRate;
  const sample = Math.sin(2 * Math.PI * frequency * t) * amplitude;
  const int16 = Math.max(-1, Math.min(1, sample)) * 32767;
  buffer.writeInt16LE(int16, offset);
  offset += 2;
}

fs.writeFileSync(outPath, buffer);
console.log(`Generated fixture: ${outPath}`);
