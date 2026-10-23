// The result card against the real API (guest + E2E_LLM=none):
// design/screens/workspace-analysis-done-result-card.png. NDWI change on the sample,
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
  page.on('response', (res) => {
    if (res.url().includes('/v1/tiles/')) tiles.push(res.status());
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

    // Legend, source line and the computed areas (statistics.water), not DESIGN §8's numbers.
    assert.equal(await card.getByTestId('legend').innerText(), 'water gained\nwater lost\nno change');
    assert.equal(await card.getByTestId('source-line').innerText(), 'NASA HLS L30 (Landsat 8/9) · 30 m');
    const stats = (await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim();
    assert.match(stats, /^OPEN WATER BEFORE 13\.9 km² OPEN WATER AFTER 170 km² GAINED \+157 km² LOST 0\.6 km²$/);
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'difference · 06-01 → 06-09');
    assert.ok((await imageWidth(card)) > 0, 'the difference PNG loads');

    // Before, then back to Difference.
    const toggle = card.getByRole('group', { name: 'Result view' });
    await toggle.getByRole('button', { name: 'Before' }).click();
    assert.equal(await toggle.getByRole('button', { name: 'Before' }).getAttribute('aria-pressed'), 'true');
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'NDWI before · 2023-06-01');
    assert.match(await card.getByTestId('result-image').getAttribute('src'), /ndwi_before\.png$/);
    assert.ok((await imageWidth(card)) > 0, 'the before PNG loads');
    await toggle.getByRole('button', { name: 'Difference' }).click();
    assert.equal(await card.getByTestId('image-overlay').innerText(), 'difference · 06-01 → 06-09');

    // Downloads: real links, 200 for the owner with the right type; no GeoPackage for a raster.
    const actions = card.getByRole('group', { name: 'Result actions' });
    assert.deepEqual(
      (await actions.locator('a, button').allInnerTexts()).map((t) => t.trim()),
      ['● On map', 'GeoTIFF', 'PNG', 'Details ›'],
    );
    for (const [name, type] of [['GeoTIFF', 'image/tiff'], ['PNG', 'image/png']]) {
      const link = actions.getByRole('link', { name });
      assert.equal(await link.getAttribute('download'), '');
      const res = await page.request.get(new URL(await link.getAttribute('href'), h.baseURL).href);
      assert.equal(res.status(), 200, name);
      assert.equal(res.headers()['content-type'], type, name);
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
    assert.equal(await page.getByRole('button', { name: 'Fit to result' }).isEnabled(), true);
    await page.waitForResponse((res) => res.url().includes('/v1/tiles/') && res.status() === 200, { timeout: 20_000 }).catch(() => {});
    assert.ok(tiles.includes(200), `result tiles load (${tiles.join(', ') || 'none'})`);

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
