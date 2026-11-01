// The guest quota against the real API: two analyses an hour (E2E_GUEST_LIMIT=2,
// `npm run e2e:workspace:quota`; guest, E2E_LLM=none). The badge counts down, in the
// canc hue at one left; at none the composer says when the next one is available,
// the third analysis gets the server's refusal inline, and conversation still works.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NDWI = 'Compare water before and after the dam breach';
const quota = { skip: process.env.E2E_GUEST_LIMIT !== '2' };

let h;
before(async () => {
  if (quota.skip) return;
  h = await setup();
});
after(async () => {
  if (quota.skip) return;
  await h?.browser.close();
});

async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
}

const badge = (page) => page.getByTestId('session-badge');
const badgeSays = (page, text) => badge(page).filter({ hasText: text }).waitFor();

test('guest quota: two analyses, then the third is refused inline and the composer says when', quota, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await badgeSays(page, 'Guest. 2 of 2 analyses left this hour · kept 24 h · no account.');
    assert.equal(await page.getByTestId('low-quota').count(), 0);

    // First analysis: one left, in the canc hue.
    await send(page, NDWI);
    await page.getByTestId('analysis-message').nth(0).and(page.locator('[data-phase="done"]')).waitFor({ timeout: 60_000 });
    await badgeSays(page, 'Guest. 1 of 2 analyses left this hour');
    assert.equal(await page.getByTestId('low-quota').innerText(), '1 of 2 analyses left this hour');
    assert.equal(await page.getByTestId('quota-line').count(), 0);

    // Second, a follow-up with the sample carried: none left.
    await send(page, NDWI);
    await page.getByTestId('analysis-message').nth(1).and(page.locator('[data-phase="done"]')).waitFor({ timeout: 60_000 });
    await badgeSays(page, 'Guest. 0 of 2 analyses left this hour');
    const line = page.getByTestId('quota-line');
    await line.waitFor();
    const session = await (await page.request.get(`${h.baseURL}/v1/session`)).json();
    assert.equal(session.analyses.remaining, 0);
    const at = await page.evaluate((iso) => {
      const d = new Date(iso);
      return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    }, session.analyses.resets_at);
    assert.equal(await line.innerText(), `You've used your 2 analyses for this hour — the next one is available at ${at}.`);

    // Third: the server's 429 inline, no job; the input still takes text.
    await send(page, NDWI);
    const refusal = page.getByTestId('assistant-message').last().getByRole('alert');
    await refusal.waitFor();
    assert.match(await refusal.innerText(), /^Guests can run 2 analyses per hour on this server\. The next one is available at \d\d:\d\d UTC\.$/);
    assert.equal(await page.getByTestId('analysis-message').count(), 2);
    assert.equal((await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).jobs.length, 2);
    assert.equal(await page.getByLabel('Message', { exact: true }).isEditable(), true);
    await line.waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    await page.screenshot({ path: outPath('workspace-guest-quota.png') });

    // Conversation still works at 0.
    await send(page, 'hello');
    await page.getByTestId('user-message').filter({ hasText: 'hello' }).waitFor();
    await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
    assert.equal(await page.getByTestId('assistant-message').last().getByRole('alert').count(), 0);
    assert.equal(await page.getByTestId('analysis-message').count(), 2);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
