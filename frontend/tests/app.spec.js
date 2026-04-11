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
});
