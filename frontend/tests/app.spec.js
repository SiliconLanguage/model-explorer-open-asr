import { test, expect } from '@playwright/test';
import path from 'node:path';

test('deep transcription flow has no auth/server failures', async ({ page }) => {
  const badResponses = [];
  page.on('response', (response) => {
    const status = response.status();
    if (status === 401 || status === 403 || status === 500) {
      badResponses.push({ url: response.url(), status });
    }
  });

  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Open-ASR Model Explorer' })).toBeVisible();

  await page.locator('#model-select').selectOption('Qwen3-ASR-1.7B');

  const fixturePath = path.resolve(process.cwd(), 'tests/fixtures/test.wav');
  await page.locator('input[type="file"]').setInputFiles(fixturePath);

  await page.getByRole('button', { name: /Transcribe/i }).click();

  const streamResponse = await page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' &&
      response.url().includes('/api/transcribe/stream'),
    { timeout: 150_000 }
  );
  expect([401, 403, 500]).not.toContain(streamResponse.status());

  await expect(page.locator('.transcript-output, .error-banner')).not.toBeEmpty({ timeout: 150_000 });

  await expect(page.locator('.error-banner')).toHaveCount(0);

  await expect(page.locator('body')).not.toContainText('Unauthorized access');
  expect(badResponses, JSON.stringify(badResponses, null, 2)).toEqual([]);
});
