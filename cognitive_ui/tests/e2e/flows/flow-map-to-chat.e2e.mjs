// Flow 2 (#25): map draw → send to chat → analysis, against the real API (guest +
// E2E_LLM=none, `npm run e2e:run3:guest`). In /app/map a rectangle is drawn with the
// toolbar (a real mouse drag), renamed, given a dataset and dates, and sent to chat; the
// region chip sits in the composer; "Compare water before and after the dam breach" runs
// on the region.
//
// There is no Earth Engine key on this server (the runner blanks GEE_PROJECT_ID and
// GEE_SERVICE_ACCOUNT_KEY), so fetching imagery for a drawn region cannot work: the
// analysis ends `failed` with EE_NOT_CONFIGURED and its two fixes, and "Try the Kakhovka
// sample" re-runs the same prompt on the bundled sample, which reaches done with the water
// change card. Without Earth Engine that is the honest end-to-end path; with a key the
// region analysis itself would run (not covered here).
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';
import { analysis, NDWI, send, settle } from '../screens.mjs';

const NAME = 'Dnipro floodplain · Kherson';

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

// The fixes and the failure need a server without Earth Engine, not the staged demo.
const skip = process.env.E2E_SERVER === 'demo' && 'needs a server without Earth Engine';

test('draw a region on the map, send it to chat, fail without Earth Engine, retry on the sample to done', { skip }, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const chats = [];
  page.on('request', (req) => {
    if (new URL(req.url()).pathname === '/v1/chat') chats.push(req.postDataJSON());
  });
  try {
    // Map mode, empty: no layers, no regions.
    await page.goto(`${h.baseURL}/app/map`);
    // The style is loaded (the draw control needs it); basemap tiles may still be on their way.
    await page.waitForFunction(() => window.__dt4lcMap?.isStyleLoaded());
    const panel = page.getByRole('complementary', { name: 'Imagery' });
    assert.equal(await panel.getByTestId('region-card').count(), 0);

    // Draw a rectangle with the toolbar: a real mouse drag over the map.
    const mapBox = await page.locator('main[aria-label="Map"]').boundingBox();
    await page.getByRole('toolbar', { name: 'Draw' }).getByRole('button', { name: 'Rectangle' }).click();
    const [x0, y0] = [mapBox.x + mapBox.width * 0.4, mapBox.y + mapBox.height * 0.3];
    const [x1, y1] = [mapBox.x + mapBox.width * 0.6, mapBox.y + mapBox.height * 0.55];
    await page.mouse.move(x0, y0);
    await page.mouse.down();
    await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 5 });
    await page.mouse.move(x1, y1, { steps: 5 });
    await page.mouse.up();
    const regionCard = panel.getByTestId('region-card');
    await regionCard.waitFor();
    // The drawn region is the map store's, persisted with its bbox.
    const drawn = await page.evaluate(() => JSON.parse(localStorage.getItem('dt4lc-app-storage')).state.selectedRegion);
    const ring = drawn.geometry.geometry.coordinates[0];
    assert.equal(ring.length, 5, 'four corners, closed');
    assert.equal(new Set(ring.map(([x]) => x.toFixed(6))).size, 2, 'two longitudes: a rectangle');
    assert.equal(new Set(ring.map(([, y]) => y.toFixed(6))).size, 2, 'two latitudes: a rectangle');

    // Rename it in place.
    await regionCard.getByRole('button', { name: 'Rename' }).click();
    const nameBox = regionCard.getByRole('textbox', { name: 'Region name' });
    await nameBox.fill(NAME);
    await nameBox.press('Enter');
    const detail = await regionCard.getByTestId('region-detail').innerText();
    const area = Number(/^(\d+) km²/.exec(detail)?.[1]);
    assert.ok(area > 10 && area < 5000, `region area (${detail})`);
    assert.equal(await page.getByTestId('region-label').innerText(), `${NAME} · ${area} km²`);

    // Dataset and dates.
    await panel.getByRole('button', { name: 'Sentinel-2' }).click();
    await panel.getByRole('button', { name: 'HLS L30' }).click();
    assert.equal(await panel.getByTestId('dataset-description').innerText(), 'NASA HLS L30 (Landsat 8/9) · 30 m · revisit 8 days');
    await panel.getByLabel('Start date').fill('2023-06-01');
    await panel.getByLabel('End date').fill('2023-06-09');
    await settle(page);
    await page.screenshot({ path: outPath('flow-map-region.png') });

    // Send to chat: the Workspace with the region chip in the composer.
    await panel.getByRole('button', { name: 'Send to chat' }).click();
    await page.waitForURL(/\/app$/);
    const chip = page.getByRole('list', { name: 'Attachments' }).getByTestId('region-chip');
    await chip.waitFor();
    assert.match((await chip.innerText()).replace(/\s+/g, ' '), new RegExp(`^Region · ${NAME} ${area} km² · HLS L30 · 2023-06-01 → 06-09 · ≤\\d+ % cloud ×$`));

    // The question on the region: the request carries it, the analysis fails cleanly.
    await send(page, NDWI);
    const failed = await analysis(page, 'failed');
    const asked = chats.at(-1);
    assert.equal(asked.message, NDWI);
    assert.deepEqual(asked.attachments, []);
    assert.equal(asked.region.name, NAME);
    assert.deepEqual([asked.region.dataset, asked.region.start_date, asked.region.end_date], ['HLS L30', '2023-06-01', '2023-06-09']);
    assert.equal(asked.region.geometry.type, 'Polygon');
    const sent = page.getByTestId('user-message').first();
    assert.equal(await sent.getByTestId('region-chip').count(), 1);
    assert.match(await failed.innerText(), new RegExp(`I'll compute NDWI for .+ from ${NAME}, then difference them`));
    assert.match(await failed.getByTestId('status-pill').innerText(), /Failed/);
    assert.equal(await failed.getByTestId('failure-code').innerText(), 'EE_NOT_CONFIGURED');
    assert.match(await failed.getByTestId('analysis-failure').innerText(), /^Earth Engine isn't configured on this server, so I can't fetch imagery for this region\./);
    const fixes = failed.getByRole('group', { name: 'Fixes' }).getByRole('button');
    assert.deepEqual(await fixes.allInnerTexts(), ['Use my own image', 'Try the Kakhovka sample']);
    const failedId = await failed.getAttribute('data-job-id');
    const record = await (await page.request.get(`${h.baseURL}/v1/jobs/${failedId}`)).json();
    assert.equal(record.status, 'failed');
    assert.equal(record.events.at(-1).data.code, 'EE_NOT_CONFIGURED');
    await settle(page);
    await page.screenshot({ path: outPath('flow-map-failed.png') });

    // Try the Kakhovka sample: the same prompt on the sample, done with the change card.
    await failed.getByRole('button', { name: 'Try the Kakhovka sample' }).click();
    const retry = await analysis(page, 'done', 1);
    const retried = page.getByTestId('user-message').nth(1);
    assert.match(await retried.innerText(), new RegExp(NDWI));
    assert.equal(await retried.getByTestId('attachment-chip').count(), 1);
    assert.match(await retried.getByTestId('attachment-chip').innerText(), /kakhovka_hls_l30/);
    assert.equal(await retried.getByTestId('region-chip').count(), 0);
    assert.equal(chats.at(-1).message, NDWI);
    assert.equal(chats.at(-1).attachments.length, 2, 'the sample pair');
    assert.equal(chats.at(-1).region, undefined);
    assert.match(await retry.getByTestId('done-line').innerText(), /Done\s+4 steps · .* · Change Detection · NDWI/);
    const card = retry.getByRole('region', { name: 'Water change · NDWI' });
    await card.waitFor();
    assert.equal(await card.getAttribute('data-kind'), 'change');
    assert.equal(await card.getByTestId('legend').innerText(), 'water gained\nwater lost\nno change');
    assert.match((await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim(), /GAINED \+157 km² LOST 0\.6 km²$/);
    // The failed message keeps its failure; two jobs, one of each.
    assert.equal(await failed.getAttribute('data-phase'), 'failed');
    const jobs = (await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).jobs;
    assert.deepEqual(jobs.map((j) => j.status).sort(), ['completed', 'failed']);
    await settle(page);
    await page.screenshot({ path: outPath('flow-map-retry-done.png') });

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
