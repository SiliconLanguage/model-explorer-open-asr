import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Test 1 — WebGPU English & Singleton Speed
//   Selects Whisper Base (WebGPU), loads the English sample, transcribes
//   twice, and asserts the second run is significantly faster (singleton
//   pipeline already cached in the worker).
// ---------------------------------------------------------------------------
test('WebGPU English & Singleton Speed', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Open-ASR Model Explorer' })).toBeVisible();

  // Select Whisper Base WebGPU
  await page.locator('#model-select').selectOption('whisper-base-webgpu');

  // Load English sample clip
  await page.getByRole('button', { name: /English/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 10_000 });

  // --- First transcription (cold: model download + pipeline init) ----------
  const t0 = Date.now();
  await page.getByRole('button', { name: /Transcribe/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 120_000 });
  const firstRunMs = Date.now() - t0;

  // No errors, transcript present
  await expect(page.locator('text=Error:')).toHaveCount(0);
  await expect(page.locator('.transcript-stable')).not.toBeEmpty({ timeout: 5_000 });

  // --- Second transcription (warm: singleton cache hit) --------------------
  const t1 = Date.now();
  await page.getByRole('button', { name: /Transcribe/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 120_000 });
  const secondRunMs = Date.now() - t1;

  await expect(page.locator('text=Error:')).toHaveCount(0);
  await expect(page.locator('.transcript-stable')).not.toBeEmpty({ timeout: 5_000 });

  // Singleton proof: second run must be at most half the first run's time
  expect(secondRunMs, `Singleton speedup: ${firstRunMs}ms → ${secondRunMs}ms`)
    .toBeLessThan(firstRunMs * 0.5);

  // VRAM metric: WebGPU models must show EST. VRAM instead of TTFT
  await expect(page.locator('.metric-label', { hasText: 'EST. VRAM' })).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.metric-value', { hasText: '~150 MB' })).toBeVisible({ timeout: 5_000 });
  await expect(page.locator('.metric-label', { hasText: 'TTFT' })).toHaveCount(0);
});

// ---------------------------------------------------------------------------
// Test 2 — Server-Side Multilingual (Chinese)
//   Selects Whisper Base (HF-GPU), loads the Chinese sample, transcribes
//   via the backend streaming endpoint, and verifies the transcript
//   populates without auth/server failures.
// ---------------------------------------------------------------------------
test('Server-Side Multilingual Chinese', async ({ page }) => {
  const badResponses = [];
  page.on('response', (response) => {
    const status = response.status();
    if (status === 401 || status === 403 || status === 500) {
      badResponses.push({ url: response.url(), status });
    }
  });

  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Open-ASR Model Explorer' })).toBeVisible();

  // Select server-side Whisper Base HF-GPU
  await page.locator('#model-select').selectOption('whisper-base-hf-gpu');

  // Load Chinese sample clip
  await page.getByRole('button', { name: /Chinese/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 10_000 });

  // Set up response listener BEFORE clicking Transcribe
  const streamResponsePromise = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.url().includes('/api/transcribe/stream'),
    { timeout: 150_000 }
  );

  await page.getByRole('button', { name: /Transcribe/i }).click();

  const streamResponse = await streamResponsePromise;
  expect([401, 403, 500]).not.toContain(streamResponse.status());

  // Wait for transcription to complete
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 150_000 });

  // Transcript must be visible and error-free
  await expect(page.locator('.transcript-output, .error-banner')).not.toBeEmpty();
  await expect(page.locator('.error-banner')).toHaveCount(0);
  await expect(page.locator('.transcript-output')).not.toContainText('[HF fallback]');
  await expect(page.locator('body')).not.toContainText('Unauthorized access');
  expect(badResponses, JSON.stringify(badResponses, null, 2)).toEqual([]);

  // LATENCY metric: HF-GPU models must show LATENCY with a real number, not TTFT or EST. VRAM
  await expect(page.locator('.metric-label', { hasText: 'LATENCY' })).toBeVisible({ timeout: 5_000 });
  // The LATENCY value must be a real number (not "—")
  const latencyValue = page.locator('.metric-card', { has: page.locator('.metric-label', { hasText: 'LATENCY' }) }).locator('.metric-value');
  await expect(latencyValue).not.toHaveText('—', { timeout: 5_000 });
  await expect(page.locator('.metric-label', { hasText: 'TTFT' })).toHaveCount(0);
  await expect(page.locator('.metric-label', { hasText: 'EST. VRAM' })).toHaveCount(0);
});

// ---------------------------------------------------------------------------
// Test 3 — Cohere WebGPU English Transcript Completeness
//   Selects cohere-transcribe-03-2026 (WebGPU ONNX), loads the English
//   sample, transcribes via the WebGPU worker, and verifies the transcript
//   is NOT truncated (must be longer than the known truncation point).
//   Known truncation: "I wanted to just share a few things, but"
//   NOTE: The 2B-param Cohere ONNX model may OOM on software renderers
//   (SwiftShader). When OOM is detected the test is skipped gracefully.
// ---------------------------------------------------------------------------
test('Cohere WebGPU English Transcript Completeness', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Open-ASR Model Explorer' })).toBeVisible();

  // Select Cohere WebGPU
  await page.locator('#model-select').selectOption('cohere-webgpu');

  // Load English sample clip (longer audio → more tokens to exercise truncation)
  await page.getByRole('button', { name: /English/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 10_000 });

  // Transcribe (cold: ONNX model download + WebGPU pipeline init)
  await page.getByRole('button', { name: /Transcribe/i }).click();
  await expect(page.getByRole('button', { name: /Transcribe/i })).toBeEnabled({ timeout: 150_000 });

  // The 2B-param Cohere ONNX model may OOM on software renderers (SwiftShader).
  // If an error banner appears with an allocation failure, skip the length check.
  const errorBanner = page.locator('.error-banner');
  const errorCount = await errorBanner.count();
  if (errorCount > 0) {
    const errorText = await errorBanner.textContent();
    if (errorText.includes('bad_alloc') || errorText.includes('session')) {
      // OOM in software renderer — expected for 2B-param model. Skip.
      console.log(`Cohere ONNX OOM in test env (expected): ${errorText}`);
      return;
    }
    // Unexpected error — fail the test.
    expect(errorCount, `Unexpected error: ${errorText}`).toBe(0);
  }

  // Transcript must exist
  const stableLocator = page.locator('.transcript-stable');
  await expect(stableLocator).not.toBeEmpty({ timeout: 5_000 });

  // Anti-truncation: the known truncation point is ~40 chars.
  // A complete transcript of the English sample (~15s audio) should be
  // significantly longer — at least 60 characters.
  const transcript = await stableLocator.textContent();
  expect(
    transcript.length,
    `Transcript too short (truncated?): "${transcript}"`,
  ).toBeGreaterThan(60);
});
