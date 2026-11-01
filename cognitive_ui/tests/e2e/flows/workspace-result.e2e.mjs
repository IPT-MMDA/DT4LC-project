// The result card against the real API (guest + E2E_LLM=none):
// the result card of a done analysis. NDWI change on the sample,
// the Difference / Before / After toggle, the downloads, and On map into the docked map.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NDWI = 'Compare water before and after the dam breach';
const ANALYSIS_TIMEOUT = 60_000;

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

// The change ramp (dta/dti/artifact_files.py): lost → no change → gained.
const CHANGE_RAMP = [[140, 74, 47], [196, 202, 214], [45, 95, 158]];

/** Distance in RGB from `px` to the brown → grey → blue ramp (its two segments). */
function offRamp(px) {
  let best = Infinity;
  for (let i = 0; i < CHANGE_RAMP.length - 1; i++) {
    const [a, b] = [CHANGE_RAMP[i], CHANGE_RAMP[i + 1]];
    const ab = a.map((v, k) => b[k] - v);
    const t = Math.min(1, Math.max(0, ab.reduce((s, v, k) => s + v * (px[k] - a[k]), 0) / ab.reduce((s, v) => s + v * v, 0)));
    best = Math.min(best, Math.hypot(...a.map((v, k) => px[k] - (v + t * ab[k]))));
  }
  return best;
}

/** The opaque pixels of a tile (every 7th, as [r, g, b]), decoded in the page. */
function opaquePixels(page, url) {
  return page.evaluate(async (u) => {
    const bitmap = await createImageBitmap(await (await fetch(u)).blob());
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const out = [];
    for (let i = 0; i < data.length; i += 4 * 7) if (data[i + 3] === 255) out.push([data[i], data[i + 1], data[i + 2]]);
    return out;
  }, url);
}

/** The shown image has loaded (or failed): `naturalWidth` tells which. */
async function imageWidth(card) {
  const img = card.getByTestId('result-image');
  await img.evaluate((el) => (el.complete ? null : new Promise((resolve) => { el.onload = el.onerror = resolve; })));
  return img.evaluate((el) => el.naturalWidth);
}

test('done with a result card: NDWI change, toggle, downloads, On map into the docked map', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const tiles = [];
  const tileURLs = [];
  page.on('response', (res) => {
    if (!res.url().includes('/v1/tiles/')) return;
    tiles.push(res.status());
    if (res.status() === 200) tileURLs.push(res.url());
  });
  try {
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    const input = page.getByLabel('Message', { exact: true });
    await input.fill(NDWI);
    await input.press('Enter');

    const message = page.getByTestId('analysis-message').first();
    await message.and(page.locator('[data-phase="done"]')).waitFor({ timeout: ANALYSIS_TIMEOUT });
    const card = message.getByRole('region', { name: 'Water change · NDWI' });
    await card.waitFor();
    assert.equal(await card.getAttribute('data-kind'), 'change');

    // Legend, source line and the computed areas (statistics.water), not the full scene's numbers.
    assert.equal(await card.getByTestId('legend').innerText(), 'water gained\nwater lost\nno change');
    assert.equal(await card.getByTestId('source-line').innerText(), 'NASA HLS L30 (Landsat 8/9) · 30 m');
    const stats = (await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim();
    assert.match(stats, /^OPEN WATER BEFORE 13\.9 km² OPEN WATER AFTER 170 km² GAINED \+157 km² LOST 0\.6 km²$/);
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'difference · 06-01 → 06-09');
    // The bare map, full-bleed; the legend swatches in the change style's colours.
    assert.match(await card.getByTestId('result-image').getAttribute('src'), /change_map_bare\.png&name=water-change_ndwi_2023-06-01_2023-06-09_map\.png$/);
    assert.ok((await imageWidth(card)) > 0, 'the difference PNG loads');
    assert.deepEqual(
      await card.getByTestId('legend').locator('[aria-hidden]').evaluateAll((els) => els.map((el) => getComputedStyle(el).backgroundColor)),
      ['rgb(45, 95, 158)', 'rgb(140, 74, 47)', 'rgb(196, 202, 214)'],
    );

    // Before, then back to Difference.
    const toggle = card.getByRole('group', { name: 'Result view' });
    await toggle.getByRole('button', { name: 'Before' }).click();
    assert.equal(await toggle.getByRole('button', { name: 'Before' }).getAttribute('aria-pressed'), 'true');
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'NDWI before · 2023-06-01');
    assert.match(await card.getByTestId('result-image').getAttribute('src'), /ndwi_before_bare\.png&name=ndwi_2023-06-01_map\.png$/);
    assert.ok((await imageWidth(card)) > 0, 'the before PNG loads');
    await toggle.getByRole('button', { name: 'Difference' }).click();
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'difference · 06-01 → 06-09');

    // Downloads: real links, 200 for the owner with the right type and a readable name; no
    // GeoPackage for a raster. PNG is the full figure (title and colour bar), not the bare map.
    const actions = card.getByRole('group', { name: 'Result actions' });
    assert.match(await actions.getByRole('link', { name: 'PNG' }).getAttribute('href'), /change_map\.png&name=water-change_ndwi_2023-06-01_2023-06-09\.png$/);
    assert.deepEqual(
      (await actions.locator('a, button').allInnerTexts()).map((t) => t.trim()),
      ['● On map', 'GeoTIFF', 'PNG', 'Details ›'],
    );
    for (const [name, type, file] of [
      ['GeoTIFF', 'image/tiff', 'water-change_ndwi_2023-06-01_2023-06-09.tif'],
      ['PNG', 'image/png', 'water-change_ndwi_2023-06-01_2023-06-09.png'],
    ]) {
      const link = actions.getByRole('link', { name });
      assert.equal(await link.getAttribute('download'), file);
      const res = await page.request.get(new URL(await link.getAttribute('href'), h.baseURL).href);
      assert.equal(res.status(), 200, name);
      assert.equal(res.headers()['content-type'], type, name);
      assert.match(res.headers()['content-disposition'], new RegExp(`^attachment; filename="${file.replace(/\./g, '\\.')}"`), name);
    }
    // Another session gets 404 for the same file.
    const other = await h.browser.newContext();
    const href = await actions.getByRole('link', { name: 'GeoTIFF' }).getAttribute('href');
    assert.equal((await other.request.get(new URL(href, h.baseURL).href)).status(), 404);
    await other.close();

    // The header counts the card.
    assert.equal(await page.locator('main[aria-label="Chat"] > header > span').innerText(), '1 file · 1 result');

    // On map: the docked map lists the layer, its legend, and fetches its tiles.
    await actions.getByRole('button', { name: 'On map' }).click();
    const layers = page.getByRole('region', { name: 'Layers' });
    await layers.getByTestId('layer-row').filter({ hasText: 'Water change · NDWI' }).waitFor();
    assert.equal(await layers.getByTestId('layers-legend').innerText(), 'lost · same · gained');
    // The legend is the change style's ramp, brown → grey → blue, as the tiles are drawn.
    assert.equal(
      await layers.getByTestId('layers-legend-ramp').evaluate((el) => getComputedStyle(el).backgroundImage),
      'linear-gradient(90deg, rgb(140, 74, 47), rgb(196, 202, 214), rgb(45, 95, 158))',
    );
    assert.equal(await page.getByRole('button', { name: 'Fit to result' }).isEnabled(), true);
    await page.waitForResponse((res) => res.url().includes('/v1/tiles/') && res.status() === 200, { timeout: 20_000 }).catch(() => {});
    assert.ok(tiles.includes(200), `result tiles load (${tiles.join(', ') || 'none'})`);
    // A tile on the raster: every opaque pixel lies on that ramp (not viridis).
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    let pixels = [];
    for (const url of tileURLs) {
      pixels = await opaquePixels(page, url);
      if (pixels.length > 0) break;
    }
    assert.ok(pixels.length > 0, `a result tile has opaque pixels (${tileURLs.length} tiles)`);
    const worst = Math.max(...pixels.map(offRamp));
    assert.ok(worst <= 3, `tile pixels on the brown → grey → blue ramp (worst off by ${worst.toFixed(1)})`);

    // The design's framing: the question at the top, the card below it.
    await page.getByTestId('user-message').first().evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    await page.screenshot({ path: outPath('workspace-analysis-done-result-card.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
