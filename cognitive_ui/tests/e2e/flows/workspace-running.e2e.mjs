// The running timeline against the real API, caught mid-run: design/screens/
// workspace-analysis-running.png. Needs E2E_LLM=fake (`npm run e2e:workspace:fake`): the
// fake Ollama holds the Interpretation step's LLM call back for INTERPRETATION_MS, so
// the Interpretation row is reliably running while the flow looks at it.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, resetLLMRules, setLLMRules, setup, trackPageErrors } from '../harness.mjs';

const NDWI = 'Compare water before and after the dam breach';
const INTERPRETATION_MS = 6000;
const INTERPRETATION =
  'Open water in the floodplain below the dam grew sharply between 1 and 9 June, almost all of it in the flooded river margins.';
const fake = { skip: process.env.E2E_LLM !== 'fake' };

let h;
before(async () => {
  if (fake.skip) return;
  h = await setup();
  // Only the agent-analysis step's call (its system prompt) is slow; everything else
  // gets the default at once.
  await setLLMRules({
    default: 'This is the E2E fake LLM. No rule matched this request.',
    rules: [{ match: /interpreting satellite imagery analysis results/, reply: INTERPRETATION, delay_ms: INTERPRETATION_MS }],
  });
});
after(async () => {
  if (fake.skip) return;
  await resetLLMRules();
  await h?.browser.close();
});

async function startNDWI() {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  await page.goto(`${h.baseURL}/app?sample=1`);
  await page.getByTestId('attachment-chip').waitFor();
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(NDWI);
  await input.press('Enter');
  const message = page.getByTestId('analysis-message').first();
  // The Interpretation row (the last of four) is running: the slow LLM call.
  const interpretation = message.getByTestId('step-row').filter({ hasText: 'Interpretation' });
  await interpretation.and(page.locator('[data-status="running"]')).waitFor({ timeout: 30_000 });
  return { context, page, errors, message, interpretation };
}

async function jobOf(page, message) {
  const res = await page.request.get(`${h.baseURL}/v1/jobs/${await message.getAttribute('data-job-id')}`);
  assert.equal(res.status(), 200);
  return res.json();
}

test('running: the timeline fills ✓ / spinner / queued with Cancel on the pill row, then done', fake, async () => {
  const { context, page, errors, message, interpretation } = await startNDWI();
  try {
    assert.equal(await message.getAttribute('data-phase'), 'running');
    assert.match(await message.innerText(), /I'll compute NDWI for 1 June and 9 June from the sample, then difference them/);
    assert.match(await message.getByTestId('status-pill').innerText(), /Running/);
    assert.match(await message.innerText(), /Step 4 of 4/);
    const statuses = await message.getByTestId('step-row').evaluateAll((rows) => rows.map((r) => r.dataset.status));
    assert.deepEqual(statuses, ['done', 'done', 'done', 'running']);
    assert.equal(await message.getByRole('button', { name: 'Cancel' }).count(), 1);
    assert.equal(await message.getByRole('progressbar', { name: 'Analysis progress' }).getAttribute('aria-valuenow'), '75');
    // The running row counts its seconds (the clock ticks once a second).
    await page.waitForTimeout(2500);
    assert.match(await interpretation.innerText(), /Interpretation\s+[2-4] s/);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: outPath('workspace-analysis-running.png') });

    // The slow call returns: done, with the fake's interpretation under the result slot.
    await message.and(page.locator('[data-phase="done"]')).waitFor({ timeout: INTERPRETATION_MS + 15_000 });
    await message.getByTestId('interpretation').waitFor();
    assert.equal(await message.getByTestId('interpretation').innerText(), INTERPRETATION);
    assert.match(await message.getByTestId('done-line').innerText(), /Done\s+4 steps · [5-9] s · Change Detection · NDWI/);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('cancel mid-run: Cancelled, and nothing is logged after it', fake, async () => {
  const { context, page, errors, message, interpretation } = await startNDWI();
  try {
    await message.getByRole('button', { name: 'Cancel' }).click();
    await message.and(page.locator('[data-phase="cancelled"]')).waitFor({ timeout: 10_000 });
    assert.match(await message.getByTestId('status-pill').innerText(), /Cancelled/);
    assert.match(await message.innerText(), /at step 4 of 4/);
    assert.equal(await interpretation.getAttribute('data-status'), 'cancelled');
    assert.equal(await message.getByRole('button', { name: 'Cancel' }).count(), 0);

    // The held-back LLM call returns after the cancel; the job logs nothing more.
    await page.waitForTimeout(INTERPRETATION_MS + 1000);
    const job = await jobOf(page, message);
    assert.equal(job.status, 'cancelled');
    const types = job.events.map((e) => e.type);
    assert.equal(types.at(-1), 'cancelled');
    assert.deepEqual(job.events.at(-1).data.at_step, 4);
    assert.equal(types.filter((t) => t === 'step_started').length, 4, 'no row started after the cancel');
    assert.equal(types.includes('done'), false);
    assert.equal(await message.getAttribute('data-phase'), 'cancelled');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
