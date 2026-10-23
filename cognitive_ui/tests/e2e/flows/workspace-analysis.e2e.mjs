// The analysis message against the real API (guest + E2E_LLM=none, Earth Engine not
// configured): design/screens/workspace-proposal.png, workspace-composed-done.png,
// workspace-analysis-failed-and-retry.png and workspace-model-gated-for-guest.png.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NDMI = 'Map moisture change with NDMI between the two dates';
const NDWI = 'Compare water before and after the dam breach';
const PRITHVI = 'Fill the cloud gaps in the 9 June image';
const PROPOSAL = /^I don't have a moisture-change algorithm\. I can compute NDMI = \(NIR − SWIR1\)\/\(NIR \+ SWIR1\) from B5 and B6 for both dates and difference them — run that\?$/;
const ANALYSIS_TIMEOUT = 60_000;

// About 410 km² of the Lower Dnipro floodplain, as the map store persists a drawn region.
const REGION = {
  id: 'region-e2e',
  name: 'Dnipro floodplain',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: {
      type: 'Polygon',
      coordinates: [[[32.4, 46.47], [32.66, 46.47], [32.66, 46.655], [32.4, 46.655], [32.4, 46.47]]],
    },
  },
  bbox: [32.4, 46.47, 32.66, 46.655],
  createdAt: '2026-10-23T00:00:00.000Z',
};

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function open(query, { region } = {}) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  if (region) {
    await context.addInitScript((r) => {
      localStorage.setItem('dt4lc-app-storage', JSON.stringify({ state: { drawnRegions: [r], selectedRegion: r }, version: 0 }));
    }, region);
  }
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  await page.goto(`${h.baseURL}/app?${query}`);
  if (query === 'sample=1') await page.getByTestId('attachment-chip').waitFor();
  return { context, page, errors };
}

async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
}

/** The analysis message for job N (0-based) once its phase is `phase`. */
async function analysis(page, phase, n = 0) {
  const message = page.getByTestId('analysis-message').nth(n);
  await message.and(page.locator(`[data-phase="${phase}"]`)).waitFor({ timeout: ANALYSIS_TIMEOUT });
  return message;
}

async function jobs(page) {
  const res = await page.request.get(`${h.baseURL}/v1/jobs`);
  assert.equal(res.status(), 200);
  return (await res.json()).jobs;
}

async function screenshot(page, name) {
  await page.evaluate(() => document.fonts.ready);
  // Basemap tiles come from the internet; take the picture whether or not they arrive.
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.screenshot({ path: outPath(name) });
}

test('proposal: NDMI on the sample proposes a composed plan with Run · Change · Explain', async () => {
  const { context, page, errors } = await open('sample=1');
  try {
    await send(page, NDMI);
    const message = await analysis(page, 'proposal');
    const proposal = message.getByTestId('proposal');
    assert.match((await proposal.locator('p').first().innerText()).trim(), PROPOSAL);
    assert.equal(await proposal.locator('code').innerText(), 'NDMI = (NIR − SWIR1)/(NIR + SWIR1)');
    for (const name of ['Run', 'Change', 'Explain']) {
      assert.equal(await proposal.getByRole('button', { name, exact: true }).count(), 1, name);
    }
    assert.equal(await message.getByTestId('analysis-timeline').count(), 0);
    const [job] = await jobs(page);
    assert.equal(job.status, 'proposal');
    await screenshot(page, 'workspace-proposal.png');

    // Change: an editable prompt with the formula in the composer.
    await proposal.getByRole('button', { name: 'Change' }).click();
    assert.equal(
      await page.getByLabel('Message', { exact: true }).inputValue(),
      `${NDMI}, using NDMI = (B5 - B6) / (B5 + B6) and a change threshold of 0.2`,
    );
    await page.getByLabel('Message', { exact: true }).fill('');

    // Explain: a conversation reply, with the sample still attached, and no new job.
    await proposal.getByRole('button', { name: 'Explain' }).click();
    await page.getByTestId('user-message').filter({ hasText: 'Explain this proposal' }).waitFor();
    await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
    await page.getByTestId('assistant-message').last().waitFor();
    assert.equal((await jobs(page)).length, 1, 'Explain creates no job');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('composed done: Run confirms the proposal, the same message runs it and carries the badge', async () => {
  const { context, page, errors } = await open('sample=1');
  try {
    await send(page, NDMI);
    const message = await analysis(page, 'proposal');
    await message.getByRole('button', { name: 'Run' }).click();
    await message.getByText('Accepted · running below').waitFor();
    await analysis(page, 'done');

    const line = message.getByTestId('done-line');
    assert.match(await line.innerText(), /Done\s+4 steps · \S+ s · NDMI\s+Composed by the assistant\s+Show steps/);
    assert.match(await message.innerText(), /I'll compute NDMI for 1 June and 9 June from the sample\./);
    // The composed card: band math change, the formula in its header, real areas.
    const card = message.getByRole('region', { name: 'Moisture change · ΔNDMI' });
    await card.waitFor();
    assert.equal(await card.getByTestId('card-formula').innerText(), '(B5 − B6)/(B5 + B6) · 06-01 → 06-09 · 30 m');
    assert.equal(
      (await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim(),
      'DRIER 42.0 km² WETTER +37.9 km² THRESHOLD ΔNDMI 0.2',
    );
    assert.equal(await card.getByTestId('legend').innerText(), 'drier\nwetter\nno change');
    const followups = message.getByRole('list', { name: 'Follow-ups' }).getByRole('button');
    assert.ok((await followups.count()) >= 2, 'two or three follow-ups');

    // Show steps expands the four rows.
    await line.getByRole('button', { name: 'Show steps' }).click();
    assert.equal(await message.getByTestId('step-row').count(), 4);
    await line.getByRole('button', { name: 'Hide steps' }).click();

    const [job] = await jobs(page);
    assert.equal(job.status, 'completed');
    // The design's framing: the question at the top, the proposal and the card below it.
    await page.getByTestId('user-message').first().evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await screenshot(page, 'workspace-composed-done.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('failed and retry: a region without Earth Engine fails with two fixes; the sample fix succeeds', async () => {
  const { context, page, errors } = await open('sample=0', { region: REGION });
  try {
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
    await page.getByRole('button', { name: 'From map' }).click();
    await page.getByTestId('region-chip').waitFor();
    await send(page, NDWI);

    const failed = await analysis(page, 'failed');
    assert.match(await failed.innerText(), /I'll compute NDWI for .+ from Dnipro floodplain, then difference them/);
    assert.match(await failed.getByTestId('status-pill').innerText(), /Failed/);
    assert.equal(await failed.getByTestId('failure-code').innerText(), 'EE_NOT_CONFIGURED');
    assert.match(
      await failed.getByTestId('analysis-failure').innerText(),
      /^Earth Engine isn't configured on this server, so I can't fetch imagery for this region\./,
    );
    const fixes = failed.getByRole('group', { name: 'Fixes' }).getByRole('button');
    assert.deepEqual(await fixes.allInnerTexts(), ['Use my own image', 'Try the Kakhovka sample']);
    await failed.getByRole('button', { name: 'show raw' }).click();
    await failed.getByTestId('failure-raw').waitFor();
    await failed.getByRole('button', { name: 'hide raw' }).click();

    // The sample instead of the region, the same prompt again.
    await failed.getByRole('button', { name: 'Try the Kakhovka sample' }).click();
    const retry = await analysis(page, 'done', 1);
    const retried = page.getByTestId('user-message').nth(1);
    assert.match(await retried.innerText(), new RegExp(NDWI));
    assert.equal(await retried.getByTestId('attachment-chip').count(), 1);
    assert.equal(await retried.getByTestId('region-chip').count(), 0);
    assert.match(await retry.getByTestId('done-line').innerText(), /Done\s+4 steps · .* · Change Detection · NDWI/);
    // The failed message keeps its open timeline.
    assert.equal(await failed.getAttribute('data-phase'), 'failed');

    await failed.scrollIntoViewIfNeeded();
    await screenshot(page, 'workspace-analysis-failed-and-retry.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('model gated for a guest: Prithvi on the sample is refused in one assistant sentence with its fixes', async () => {
  const { context, page, errors } = await open('sample=1');
  try {
    await send(page, PRITHVI);
    const failed = await analysis(page, 'failed');
    // The design's plain sentence: no Failed pill, timeline, code or show raw.
    assert.equal(
      await failed.getByTestId('analysis-refusal').innerText(),
      "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator.",
    );
    for (const id of ['status-pill', 'analysis-timeline', 'failure-code', 'step-row']) {
      assert.equal(await failed.getByTestId(id).count(), 0, id);
    }
    assert.equal(await failed.getByRole('button', { name: 'show raw' }).count(), 0);
    // Still the server's MODEL_GATED failure underneath.
    const record = await (await page.request.get(`${h.baseURL}/v1/jobs/${await failed.getAttribute('data-job-id')}`)).json();
    assert.equal(record.status, 'failed');
    assert.equal(record.events.at(-1).data.code, 'MODEL_GATED');
    const fixes = failed.getByRole('group', { name: 'Fixes' }).getByRole('button');
    assert.deepEqual(await fixes.allInnerTexts(), ['See what I can do', 'Use a built-in analysis instead']);
    await screenshot(page, 'workspace-model-gated-for-guest.png');

    // A `prompt` fix sends its prompt: the catalog reply.
    await fixes.first().click();
    await page.getByTestId('user-message').filter({ hasText: 'What can you do?' }).waitFor();
    await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
    assert.ok((await page.getByTestId('assistant-message').last().locator('li').count()) >= 5, 'the catalog');
    assert.equal((await jobs(page)).length, 1);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
