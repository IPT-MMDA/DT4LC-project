// Smoke: the built app loads through the harness and renders its shell.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

test('app shell renders without uncaught page errors', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  try {
    const page = await context.newPage();
    const errors = trackPageErrors(page);
    // The bundle's default API origin must be rewritten to the harness proxy.
    const stray = [];
    page.on('request', (req) => {
      if (req.url().startsWith('http://localhost:8000')) stray.push(req.url());
    });

    await page.goto(h.baseURL);
    await page.locator('#root > *').first().waitFor({ state: 'visible' });
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: outPath('smoke.png') });

    assert.deepEqual(errors, []);
    assert.deepEqual(stray, []);
  } finally {
    await context.close();
  }
});
