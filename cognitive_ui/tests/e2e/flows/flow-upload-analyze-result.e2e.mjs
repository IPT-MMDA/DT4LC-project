// Flow 1 (#25): upload → analyze → result, against the real API (guest + E2E_LLM=none,
// `npm run e2e:run3:guest`). A GeoTIFF goes in through the composer's + (the real file
// input, a renamed copy of the 9 June sample file, so it is an upload and not
// POST /v1/sample); its chip shows the metadata line; "Map vegetation health across the
// region" runs NDVI to done (the intent line, the timeline's three steps); the result card
// shows the NDVI map, its legend and statistics; Details › opens Result detail with the
// plan, statistics and downloads, and the GeoTIFF downloads under its readable name.
import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';
import { analysis, send, settle } from '../screens.mjs';

const PROMPT = 'Map vegetation health across the region';
const NAME = 'dnipro-floodplain.tif';
const SOURCE = fileURLToPath(new URL('../../../../resources/kahovka_data/hlsl_20230609.tif', import.meta.url));

let h;
let dir;
before(async () => {
  h = await setup();
  dir = mkdtempSync(path.join(os.tmpdir(), 'dt4lc-flow-upload-'));
  copyFileSync(SOURCE, path.join(dir, NAME));
});
after(async () => {
  if (dir) rmSync(dir, { recursive: true, force: true });
  await h?.browser.close();
});

test('upload a GeoTIFF with +, analyze it, read the result card and its detail, download the GeoTIFF', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const calls = [];
  page.on('request', (req) => {
    if (new URL(req.url()).pathname.startsWith('/v1/')) calls.push(`${req.method()} ${new URL(req.url()).pathname}`);
  });
  try {
    await page.goto(`${h.baseURL}/app`);
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();

    // + opens the file picker; the real input uploads the file.
    const chooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Attach GeoTIFF' }).click();
    const [upload] = await Promise.all([
      page.waitForResponse((res) => new URL(res.url()).pathname === '/v1/upload'),
      (await chooser).setFiles(path.join(dir, NAME)),
    ]);
    assert.equal(upload.status(), 200);
    const uploaded = await upload.json();
    assert.equal(uploaded.filename, NAME);
    assert.ok(!calls.includes('POST /v1/sample'), 'an upload, not the sample endpoint');

    // The chip: the name and the metadata line read from the file (no date in this name).
    const chip = page.getByRole('list', { name: 'Attachments' }).getByTestId('attachment-chip');
    await chip.filter({ hasText: 'HLS L30' }).waitFor();
    assert.match((await chip.innerText()).replace(/\s+/g, ' '), new RegExp(`^${NAME.replace(/\./g, '\\.')} HLS L30 · B2–B6 · 30 m · 3\\.7 MB ×$`));
    assert.equal(await page.getByRole('heading', { name: 'What do you want to know about this area?' }).count(), 1);

    // Ask: the intent line, then done.
    await send(page, PROMPT);
    const sent = page.getByTestId('user-message').first();
    await sent.waitFor();
    assert.match(await sent.innerText(), new RegExp(PROMPT));
    assert.equal(await sent.getByTestId('attachment-chip').count(), 1);
    const message = await analysis(page, 'done');
    assert.match(await message.innerText(), new RegExp(`I'll compute NDVI from ${NAME.replace(/\./g, '\\.')}\\.`));
    const line = message.getByTestId('done-line');
    assert.match(await line.innerText(), /Done\s+3 steps · (<1|\d+) s · NDVI/);
    // The timeline: three steps, each done.
    await line.getByRole('button', { name: 'Show steps' }).click();
    const rows = message.getByTestId('step-row');
    assert.deepEqual(await rows.evaluateAll((els) => els.map((el) => el.dataset.status)), ['done', 'done', 'done']);
    assert.deepEqual(
      (await rows.allInnerTexts()).map((t) => t.replace(/\s+/g, ' ').trim().replace(/ (<1|\d+) s$/, '')),
      [`✓ Read ${NAME}`, '✓ NDVI', '✓ Interpretation'],
    );
    await line.getByRole('button', { name: 'Hide steps' }).click();
    // The server's record: intent, planning, three steps, done, and the job ran on the upload.
    const jobId = await message.getAttribute('data-job-id');
    const job = await (await page.request.get(`${h.baseURL}/v1/jobs/${jobId}`)).json();
    assert.equal(job.status, 'completed');
    const types = job.events.map((e) => e.type).filter((t, i, all) => t !== all[i - 1]);
    assert.deepEqual(types.slice(0, 2), ['intent', 'planning']);
    assert.equal(job.events.filter((e) => e.type === 'step_finished').length, 3);
    assert.equal(types.at(-1), 'done');

    // The result card: the NDVI map, its legend and statistics.
    const card = message.getByRole('region', { name: 'NDVI', exact: true });
    await card.waitFor();
    assert.equal(await card.getAttribute('data-kind'), 'index');
    const img = card.getByTestId('result-image');
    await img.evaluate((el) => (el.complete ? null : new Promise((resolve) => { el.onload = el.onerror = resolve; })));
    assert.ok((await img.evaluate((el) => el.naturalWidth)) > 0, 'the NDVI map loads');
    // The legend: the index ramp between its stretch ends.
    assert.equal((await card.getByTestId('legend').innerText()).replace(/\s+/g, ' ').trim(), '−0.2 0.8');
    assert.equal(await card.getByTestId('legend').getByTestId('legend-ramp').count(), 1);
    assert.equal(await card.getByTestId('source-line').innerText(), 'NASA HLS L30 (Landsat 8/9) · 30 m');
    assert.equal((await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim(), 'MEAN 0.17 MIN −1 MAX 1 VALID PIXELS 589 824');
    const actions = card.getByRole('group', { name: 'Result actions' });
    assert.deepEqual((await actions.locator('a, button').allInnerTexts()).map((t) => t.trim()), ['● On map', 'GeoTIFF', 'PNG', 'Details ›']);
    await settle(page);
    await page.screenshot({ path: outPath('flow-upload-result-card.png') });

    // Details ›: Result detail with the plan, statistics and downloads.
    await actions.getByRole('link', { name: 'Details ›' }).click();
    await page.waitForURL(new RegExp(`/app/results/${jobId}$`));
    await page.getByTestId('result-detail').waitFor();
    assert.equal(await page.getByTestId('detail-title').innerText(), 'NDVI');
    assert.deepEqual(await page.getByTestId('detail-breadcrumb').getByRole('link').allInnerTexts(), ['Library', PROMPT, 'NDVI']);
    await page.waitForFunction(() => [...document.querySelectorAll('img[data-testid="detail-image"]')].every((i) => i.complete && i.naturalWidth > 0));
    const plan = page.getByTestId('detail-plan');
    assert.match(await plan.getByRole('heading').innerText(), /^PLAN · 3 STEPS · (<1|\d+) S$/);
    assert.deepEqual(await plan.getByTestId('plan-step').evaluateAll((els) => els.map((el) => el.dataset.status)), ['done', 'done', 'done']);
    const stats = (await page.getByTestId('statistics-row').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ').trim());
    assert.deepEqual(stats, ['Mean 0.17', 'Min −1', 'Max 1', 'Valid pixels 589 824']);
    assert.match(await page.getByTestId('detail-asked').innerText(), new RegExp(`"${PROMPT}"[\\s\\S]*${NAME.replace(/\./g, '\\.')}`));
    const downloads = (await page.getByTestId('download-row').allInnerTexts()).map((t) => t.replace(/\s+/g, ' ').trim());
    assert.deepEqual(downloads.map((d) => d.split(' ')[0]), ['ndvi.tif', 'ndvi.png', 'ndvi_map.png', 'statistics.csv']);
    assert.match(downloads[0], /^ndvi\.tif GeoTIFF · \d+(\.\d)? MB$/);
    await settle(page);
    await page.screenshot({ path: outPath('flow-upload-result-detail.png') });

    // Download the GeoTIFF: a 200 TIFF saved under the readable name, not the stored one.
    const tifRow = page.getByTestId('download-row').first();
    const href = await tifRow.getAttribute('href');
    assert.match(href, /path=[^&]*1_NDVIMap_ndvi\.tif&name=ndvi\.tif$/);
    const [download] = await Promise.all([page.waitForEvent('download'), tifRow.click()]);
    assert.equal(download.suggestedFilename(), 'ndvi.tif');
    const saved = path.join(dir, `saved-${download.suggestedFilename()}`);
    await download.saveAs(saved);
    assert.equal(readFileSync(saved).subarray(0, 4).toString('latin1'), 'II*\0', 'a little-endian TIFF');
    const res = await page.request.get(new URL(href, h.baseURL).href);
    assert.equal(res.status(), 200);
    assert.equal(res.headers()['content-type'], 'image/tiff');
    assert.match(res.headers()['content-disposition'], /^attachment; filename="ndvi\.tif"/);

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
