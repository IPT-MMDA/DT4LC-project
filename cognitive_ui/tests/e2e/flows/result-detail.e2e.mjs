// Result detail against the real API (guest + E2E_LLM=none, no Earth Engine):
// design/screens/result-detail.png (1440×900). Seeded through the UI: the sample and
// NDWI change in one conversation, a region job failing with EE_NOT_CONFIGURED in a
// second. The Library row opens /app/results/:id: the breadcrumb, the difference,
// before and after images, the class statistics with the real areas, the plan steps
// with times, the downloads with sizes and the CSV. Then Open on map (map mode with
// the layers), Re-run with changes (the composer, no job created) and Open in
// workspace (the conversation selected). The failed job's detail shows its code and
// fixes; a random id is not available.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NDWI = 'Compare water before and after the dam breach';
const ANALYSIS_TIMEOUT = 60_000;

// About 410 km² of the Lower Dnipro floodplain, as the map store persists a drawn region.
const REGION = {
  id: 'region-e2e',
  name: 'Dnipro floodplain',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[32.4, 46.47], [32.66, 46.47], [32.66, 46.655], [32.4, 46.655], [32.4, 46.47]]] },
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

async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
}

async function analysis(page, phase, n) {
  const message = page.getByTestId('analysis-message').nth(n);
  await message.and(page.locator(`[data-phase="${phase}"]`)).waitFor({ timeout: ANALYSIS_TIMEOUT });
  return message.getAttribute('data-job-id');
}

async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
}

const stored = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('dt4lc-chat-storage')).state);
const jobCount = async (page) => (await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).total;

test('result detail: from the Library row, maps, statistics, plan, downloads, CSV and the three actions', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript((r) => {
    if (!localStorage.getItem('dt4lc-app-storage')) {
      localStorage.setItem('dt4lc-app-storage', JSON.stringify({ state: { drawnRegions: [r], selectedRegion: r }, version: 0 }));
    }
  }, REGION);
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    // Seed: the sample and NDWI change; a region job in a second conversation.
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await send(page, NDWI);
    const ndwiId = await analysis(page, 'done', 0);
    await page.getByRole('complementary', { name: 'Conversations' }).getByRole('button', { name: '+ New' }).click();
    await page.getByRole('button', { name: 'From map' }).click();
    await page.getByTestId('region-chip').waitFor();
    await send(page, NDWI);
    const failedId = await analysis(page, 'failed', 0);
    const ndwiConversation = (await stored(page)).conversations.find((c) => c.messages.some((m) => m.jobId === ndwiId));

    // The Library row opens Result detail.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).click();
    const row = page.locator(`[data-testid="library-row"][data-id="${ndwiId}"]`);
    await row.getByTestId('library-name').click();
    await page.waitForURL(new RegExp(`/app/results/${ndwiId}$`));
    const detail = page.getByTestId('result-detail');
    await detail.waitFor();

    assert.deepEqual(await page.getByTestId('detail-breadcrumb').getByRole('link').allInnerTexts(), ['Library', ndwiConversation.title, 'Water change · NDWI difference']);
    assert.equal(await page.getByTestId('detail-title').innerText(), 'Water change · NDWI difference');
    assert.match(await page.getByTestId('detail-meta').innerText(), new RegExp(`Done\\s+(<1|\\d+) s · today \\d\\d:\\d\\d\\s+${ndwiId}$`));
    assert.equal(await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).getAttribute('aria-current'), 'page');

    // The difference, before and after images: loaded, each a 200.
    assert.deepEqual(await page.getByTestId('detail-map').evaluateAll((maps) => maps.map((m) => m.dataset.view)), ['difference', 'before', 'after']);
    await page.waitForFunction(() => [...document.querySelectorAll('img[data-testid="detail-image"]')].every((i) => i.complete && i.naturalWidth > 0));
    for (const src of await page.getByTestId('detail-image').evaluateAll((imgs) => imgs.map((i) => i.src))) {
      assert.equal((await page.request.get(src)).status(), 200, src);
    }
    assert.equal(await page.getByTestId('detail-map-meta').first().innerText(), '2023-06-01 → 2023-06-09 · 30 m · EPSG:32636');

    // Statistics: the real class areas and shares of the sample.
    const statsRows = (await page.getByTestId('statistics-row').allInnerTexts()).map((t) => t.split(/\s*\t\s*|\n/).map((c) => c.trim()).filter(Boolean));
    assert.deepEqual(statsRows, [
      ['Water gained', '156.8', '29.5', '+0.81'],
      ['Water lost', '0.6', '0.1', '−0.23'],
      ['No change', '373.5', '70.4', '+0.16'],
    ]);
    const csvHref = await page.getByRole('region', { name: 'Statistics' }).getByRole('link', { name: 'CSV' }).getAttribute('href');
    const csv = await page.request.get(new URL(csvHref, h.baseURL).href);
    assert.equal(csv.status(), 200);
    assert.match(csv.headers()['content-type'], /^text\/csv/);
    assert.match(await csv.text(), /^class,area_km2,percent,pixels,mean_change\nWater gained,156\.761,29\.53,/);

    // Plan: the four steps, each with its time.
    const plan = page.getByTestId('detail-plan');
    assert.match(await plan.getByRole('heading').innerText(), /^PLAN · 4 STEPS · (<1|\d+) S$/);
    const params = await plan.getByTestId('plan-step-params').allInnerTexts();
    assert.equal(params.length, 4);
    assert.ok(params.every((p) => /(<1|\d+) s$/.test(p)), params.join(' | '));
    assert.match(params[0], /^before · 2023-06-01 · /);
    assert.match(params[2], /^\(Green − NIR\)\/\(Green \+ NIR\) · after − before · threshold 0\.2 · /);

    // Downloads: every file with its kind and size, then the CSV.
    const downloads = await page.getByTestId('download-row').allInnerTexts();
    assert.equal(downloads.length, 10, downloads.join(' | '));
    assert.ok(downloads.every((d) => /(GeoTIFF|PNG|CSV) · \d+(\.\d)? (KB|MB)$/.test(d.replace(/\s+/g, ' '))), downloads.join(' | '));
    assert.match(downloads[0].replace(/\s+/g, ' '), /^2_ChangeMap_change\.tif GeoTIFF · 2\.0 MB$/);
    assert.match(downloads[9].replace(/\s+/g, ' '), /^statistics\.csv CSV · 1 KB$/);
    // No assistant: no interpretation, and never "summarization failed".
    assert.equal(await page.getByTestId('interpretation').count(), 0);
    assert.doesNotMatch(await detail.innerText(), /summarization failed/i);

    await settle(page);
    await page.screenshot({ path: outPath('result-detail.png') });

    // Open on map: the result layers and the sample in map mode.
    await page.getByRole('button', { name: 'Open on map' }).click();
    await page.waitForURL(/\/app\/map$/);
    const layers = page.getByRole('complementary', { name: 'Imagery' }).getByRole('region', { name: 'Map layers' });
    await layers.getByTestId('map-layer-row').filter({ hasText: 'Water change · NDWI' }).waitFor();
    await layers.getByTestId('map-layer-row').filter({ hasText: 'kakhovka_hls_l30' }).waitFor();
    await page.waitForFunction((id) => window.__dt4lcMap?.getStyle()?.layers.some((l) => l.id === id), `result-${ndwiId}-difference`);

    // Re-run with changes: the prompt and the sample in the composer, nothing sent.
    const jobsBefore = await jobCount(page);
    await page.goto(`${h.baseURL}/app/results/${ndwiId}`);
    await page.getByRole('button', { name: 'Re-run with changes' }).click();
    await page.waitForURL(/\/app$/);
    assert.equal(await page.getByLabel('Message', { exact: true }).inputValue(), NDWI);
    // The composer's chips (sent messages have their own).
    const chips = page.getByRole('list', { name: 'Attachments' }).getByTestId('attachment-chip');
    assert.equal(await chips.count(), 1);
    assert.match(await chips.innerText(), /kakhovka_hls_l30/);
    assert.equal((await stored(page)).activeConversationId, ndwiConversation.id);
    await page.waitForTimeout(500);
    assert.equal(await jobCount(page), jobsBefore);

    // Open in workspace from another conversation: the job's conversation is selected.
    await page.getByRole('complementary', { name: 'Conversations' }).getByRole('button', { name: '+ New' }).click();
    await page.goto(`${h.baseURL}/app/results/${ndwiId}`);
    await page.getByRole('button', { name: 'Open in workspace' }).click();
    await page.waitForURL(/\/app$/);
    assert.equal((await stored(page)).activeConversationId, ndwiConversation.id);
    await page.locator(`[data-testid="analysis-message"][data-job-id="${ndwiId}"]`).waitFor();

    // The failed region job: its code and fixes instead of the maps.
    await page.goto(`${h.baseURL}/app/results/${failedId}`);
    const failure = page.getByTestId('detail-failure');
    await failure.waitFor();
    assert.equal(await failure.getByTestId('failure-code').innerText(), 'EE_NOT_CONFIGURED');
    assert.deepEqual(await failure.getByRole('group', { name: 'Fixes' }).getByRole('button').allInnerTexts(), ['Use my own image', 'Try the Kakhovka sample']);
    assert.equal(await page.getByTestId('detail-map').count(), 0);
    assert.match(await page.getByTestId('detail-asked').getByTestId('region-chip').innerText(), /Dnipro floodplain/);
    await settle(page);
    await page.screenshot({ path: outPath('result-detail-failed.png') });

    // A random id: not available.
    await page.goto(`${h.baseURL}/app/results/no-such-job`);
    const gone = page.getByTestId('result-unavailable');
    await gone.waitFor();
    assert.match(await gone.innerText(), /This result isn't available/);
    assert.match(await gone.innerText(), /guest results are kept 24 h/);
    assert.deepEqual(await gone.getByRole('link').allInnerTexts(), ['Library', 'Chat']);
    await page.screenshot({ path: outPath('result-detail-unavailable.png') });

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
