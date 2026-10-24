// The Library against the real API (guest + E2E_LLM=none, no Earth Engine):
// design/screens/library.png (1440×900). Seeded through the UI: the sample, NDWI
// change, statistics for every band, and a region job that fails with
// EE_NOT_CONFIGURED. Then the rows, types, statuses and conversation names; each
// filter narrows the list; CSV downloads; Map opens map mode with the layer; a row
// opens /app/results/:id; Retry and Chat go back to the Workspace; Delete removes a
// row and the API confirms 404; Upload GeoTIFF adds a file row. An empty session
// shows "Nothing here yet".
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NDWI = 'Compare water before and after the dam breach';
const STATS = 'Give me statistics for every band';
const ANALYSIS_TIMEOUT = 60_000;
const UPLOAD = fileURLToPath(new URL('../../../../resources/kahovka_data/hlsl_20230601.tif', import.meta.url));

// About 410 km² of the Lower Dnipro floodplain, as the map store persists a drawn region.
const REGION = {
  id: 'region-e2e',
  name: 'Dnipro floodplain · Kherson',
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

const rowTexts = async (page, testId) => page.getByTestId('library-row').getByTestId(testId).allInnerTexts();
const count = async (page) => page.getByTestId('library-count').innerText();

async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
}

test('library: nothing here yet for a new session', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app/library`);
    const empty = page.getByTestId('library-empty');
    await empty.waitFor();
    assert.match(await empty.innerText(), /^Nothing here yet\n/);
    assert.equal(await empty.getByRole('link', { name: 'Go to Chat' }).getAttribute('href'), '/app');
    assert.equal(
      await page.getByText(/^Every analysis and file from your conversations\./).innerText(),
      'Every analysis and file from your conversations. Guest items expire after 24 h.',
    );
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('library: analyses and files, filters, CSV, Map, Result detail, Retry, Chat, Delete', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  // The drawn region, once: later loads keep whatever the app stored since.
  await context.addInitScript((r) => {
    if (!localStorage.getItem('dt4lc-app-storage')) {
      localStorage.setItem('dt4lc-app-storage', JSON.stringify({ state: { drawnRegions: [r], selectedRegion: r }, version: 0 }));
    }
    // Every in-app navigation, so a redirect after /app/results/:id doesn't hide it.
    window.__urls = [];
    for (const name of ['pushState', 'replaceState']) {
      const original = history[name].bind(history);
      history[name] = (state, title, url) => {
        if (url) window.__urls.push(String(url));
        return original(state, title, url);
      };
    }
  }, REGION);
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    // Seed: the sample, NDWI change and statistics in one conversation.
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await send(page, NDWI);
    const ndwiId = await analysis(page, 'done', 0);
    await send(page, STATS);
    const statsId = await analysis(page, 'done', 1);
    // A region job in a second conversation: no Earth Engine, so it fails.
    await page.getByRole('complementary', { name: 'Conversations' }).getByRole('button', { name: '+ New' }).click();
    await page.getByRole('button', { name: 'From map' }).click();
    await page.getByTestId('region-chip').waitFor();
    await send(page, NDWI);
    const failedId = await analysis(page, 'failed', 0);

    const api = async (path) => (await page.request.get(`${h.baseURL}${path}`)).json();
    const cards = Object.fromEntries((await api('/v1/jobs')).jobs.map((j) => [j.id, j.card]));
    assert.equal(cards[failedId].error.code, 'EE_NOT_CONFIGURED');
    const titles = await page.evaluate(() => JSON.parse(localStorage.getItem('dt4lc-chat-storage')).state.conversations.map((c) => c.title));
    assert.equal(titles.length, 2);
    const [regionConversation, sampleConversation] = titles;

    // The Library from the rail.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).click();
    await page.waitForURL(/\/app\/library$/);
    await page.getByTestId('library-row').nth(3).waitFor();

    assert.equal(await count(page), '4 items');
    assert.deepEqual(await page.getByTestId('library-row').evaluateAll((rows) => rows.map((r) => r.dataset.id)), [failedId, statsId, ndwiId, (await api('/v1/files')).files[0].upload_id]);
    assert.deepEqual(await rowTexts(page, 'library-name'), [cards[failedId].title, cards[statsId].title, 'Water change · NDWI difference', 'kakhovka_hls_l30.tif']);
    assert.match(cards[statsId].title, /^Raster statistics · 2023-06-0[19]$/);
    const subtitles = await rowTexts(page, 'library-subtitle');
    assert.deepEqual(subtitles.slice(0, 3), [cards[failedId].error.message, `"${STATS}"`, `"${NDWI}"`]);
    assert.match(subtitles[0], /Earth Engine/);
    assert.equal(subtitles[3], 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636');
    assert.deepEqual(await rowTexts(page, 'library-type'), [cards[failedId].analysis?.label ?? '—', 'Statistics', 'Change detection', 'File · sample']);
    // The pill's word and run time, without its glyph.
    const statuses = (await rowTexts(page, 'library-status')).map((t) => t.replace(/^[✓!–]\s*/, ''));
    assert.match(statuses[0], /^Failed( · (<1|\d+) s)?$/);
    assert.match(statuses[1], /^Done · (<1|\d+) s$/);
    assert.match(statuses[2], /^Done · (<1|\d+) s$/);
    assert.equal(statuses[3], '—');
    assert.deepEqual(await rowTexts(page, 'library-conversation'), [regionConversation, sampleConversation, sampleConversation, sampleConversation]);
    const when = await rowTexts(page, 'library-when');
    assert.ok(when.slice(0, 3).every((w) => /^today \d\d:\d\d$/.test(w)), when.join(', '));
    assert.equal(when[3], 'bundled');
    const actions = await page.getByTestId('library-row').evaluateAll((rows) => rows.map((r) => [...r.querySelectorAll('td:last-child a, td:last-child button:not([aria-haspopup])')].map((b) => b.textContent)));
    assert.deepEqual(actions, [['Retry'], ['CSV'], ['Map'], ['Chat']]);
    // Thumbnails: the NDWI map and the sample preview load; the others are placeholders.
    await page.waitForFunction(() => [...document.querySelectorAll('img[data-testid="library-thumbnail"]')].every((i) => i.complete && i.naturalWidth > 0));
    assert.equal(await page.locator('img[data-testid="library-thumbnail"]').count(), 2);

    await settle(page);
    await page.screenshot({ path: outPath('library.png') });

    // Each filter narrows the list.
    const filters = page.getByRole('toolbar', { name: 'Filters' });
    await filters.getByRole('button', { name: 'Analyses' }).click();
    assert.equal(await count(page), '3 items');
    await filters.getByRole('button', { name: 'Files' }).click();
    assert.deepEqual(await rowTexts(page, 'library-name'), ['kakhovka_hls_l30.tif']);
    await filters.getByRole('button', { name: 'All' }).click();
    await filters.getByRole('button', { name: 'Failed' }).click();
    assert.deepEqual(await rowTexts(page, 'library-name'), [cards[failedId].title]);
    await filters.getByRole('button', { name: 'Failed' }).click();
    await filters.getByRole('button', { name: 'Done' }).click();
    assert.equal(await count(page), '2 items');
    await filters.getByRole('button', { name: 'Cancelled' }).click();
    await filters.getByRole('button', { name: 'Done' }).click();
    await page.getByTestId('library-filtered-empty').waitFor();
    await page.getByRole('button', { name: 'Clear filters' }).click();
    assert.equal(await count(page), '4 items');
    await filters.getByRole('button', { name: 'Type: any ▾' }).click();
    const typeMenu = page.getByRole('menu', { name: 'Type' });
    assert.deepEqual((await typeMenu.getByRole('menuitemradio').allInnerTexts()).slice(-2), ['Statistics', 'File · sample']);
    await typeMenu.getByRole('menuitemradio', { name: 'Statistics' }).click();
    assert.deepEqual(await rowTexts(page, 'library-name'), [cards[statsId].title]);
    await filters.getByRole('button', { name: 'Type: Statistics ▾' }).click();
    await page.getByRole('menu', { name: 'Type' }).getByRole('menuitemradio', { name: 'Any type' }).click();
    await filters.getByRole('button', { name: 'All time ▾' }).click();
    await page.getByRole('menu', { name: 'When' }).getByRole('menuitemradio', { name: 'Today' }).click();
    assert.equal(await count(page), '4 items');
    assert.equal(await filters.getByRole('button', { name: 'Today ▾' }).count(), 1);

    // CSV: the statistics table.
    const rows = page.getByTestId('library-row');
    const csvHref = await rows.nth(1).getByRole('link', { name: 'CSV' }).getAttribute('href');
    const csv = await page.request.get(new URL(csvHref, h.baseURL).href);
    assert.equal(csv.status(), 200);
    assert.match(csv.headers()['content-type'], /^text\/csv/);
    assert.match((await csv.text()).split('\n')[0], /^band,min,max,mean,std,median/);

    // A row opens Result detail.
    await rows.nth(2).getByTestId('library-name').click();
    await page.waitForFunction((id) => window.__urls.includes(`/app/results/${id}`), ndwiId);
    await page.goto(`${h.baseURL}/app/library`);
    await rows.nth(3).waitFor();

    // Map: the NDWI change as a layer in map mode.
    await rows.nth(2).getByRole('button', { name: 'Map' }).click();
    await page.waitForURL(/\/app\/map$/);
    const layers = page.getByRole('complementary', { name: 'Imagery' }).getByRole('region', { name: 'Map layers' });
    await layers.getByTestId('map-layer-row').filter({ hasText: 'Water change · NDWI' }).waitFor();
    // On the map itself (map mode exposes its instance to automation).
    await page.waitForFunction((id) => window.__dt4lcMap?.getStyle()?.layers.some((l) => l.id === id), `result-${ndwiId}-difference`);

    // Chat: a new conversation with the sample attached.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).click();
    await rows.nth(3).getByRole('button', { name: 'Chat' }).click();
    await page.waitForURL(/\/app$/);
    await page.getByTestId('attachment-chip').waitFor();
    assert.match(await page.getByTestId('attachment-chip').innerText(), /kakhovka_hls_l30/);

    // Retry: the prompt again in the region conversation, which fails again.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).click();
    await rows.nth(0).getByRole('button', { name: 'Retry' }).click();
    await page.waitForURL(/\/app$/);
    await analysis(page, 'failed', 1);
    assert.equal(await page.getByTestId('user-message').count(), 2);
    assert.match(await page.getByTestId('user-message').nth(1).innerText(), new RegExp(NDWI));

    // Delete the statistics row: gone from the list, 404 from the API.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Library' }).click();
    await rows.nth(4).waitFor();
    const statsRow = page.locator(`[data-testid="library-row"][data-id="${statsId}"]`);
    await statsRow.getByRole('button', { name: `More for ${cards[statsId].title}` }).click();
    await statsRow.getByRole('menuitem', { name: 'Delete…' }).click();
    await statsRow.getByRole('alertdialog').getByRole('button', { name: 'Delete' }).click();
    await statsRow.waitFor({ state: 'detached' });
    assert.equal(await count(page), '4 items');
    assert.equal((await page.request.get(`${h.baseURL}/v1/jobs/${statsId}`)).status(), 404);

    // Upload GeoTIFF: a File · upload row with the file's metadata line.
    await page.getByLabel('GeoTIFF file').setInputFiles(UPLOAD);
    const uploaded = page.getByTestId('library-row').filter({ hasText: 'File · upload' });
    await uploaded.waitFor();
    assert.equal(await uploaded.getByTestId('library-name').innerText(), 'hlsl_20230601.tif');
    assert.equal(await uploaded.getByTestId('library-subtitle').innerText(), 'HLS L30 · B2–B6 · 30 m · 2023-06-01 · 4.0 MB');
    assert.equal(await count(page), '5 items');

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
