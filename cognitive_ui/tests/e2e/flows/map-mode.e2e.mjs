// Map mode against the real API (guest + E2E_LLM=none, no Earth Engine):
// design/screens/map-mode.png (1440×900) and map-mode-1280.png (1280×800). The NDWI
// change result goes On map from the chat; map mode shows it with its legend over the
// sample in true colour and its footprint; a rectangle is drawn with the mouse and
// renamed; Fit to result and Compare move the view and the divider; Fetch to map is
// disabled with its reason; Send to chat puts the region chip in the composer; at
// 1280×800 no panel or control overlaps another, the presets row sits under the region
// card and regions are amber; the presets and Upload GeoJSON add regions.
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

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

/** [r, g, b] at page points, read from a screenshot decoded in the page. */
async function pixelsAt(page, points) {
  const png = (await page.screenshot()).toString('base64');
  return page.evaluate(async ({ data, pts }) => {
    const bitmap = await createImageBitmap(await (await fetch(`data:image/png;base64,${data}`)).blob());
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    return pts.map(([x, y]) => [...ctx.getImageData(Math.round(x), Math.round(y), 1, 1).data.slice(0, 3)]);
  }, { data: png, pts: points });
}

/** A 5 × 5 grid of points 18 px apart around (x, y). */
const around = (x, y) => [-2, -1, 0, 1, 2].flatMap((i) => [-2, -1, 0, 1, 2].map((j) => [x + i * 18, y + j * 18]));

/** The map's view, through the instance map mode exposes to automation. */
const view = (page) =>
  page.evaluate(() => {
    const m = window.__dt4lcMap;
    const b = m.getBounds();
    return { lng: m.getCenter().lng, lat: m.getCenter().lat, zoom: m.getZoom(), bounds: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] };
  });

async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  // Basemap tiles come from the internet; go on whether or not they arrive.
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.evaluate(() => new Promise((resolve) => (window.__dt4lcMap?.loaded() ? resolve() : window.__dt4lcMap?.once('idle', resolve) ?? resolve())));
}

const intersects = (a, b) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

// Fetch to map must be disabled: not on the staged demo server, whose Earth Engine is "configured".
const skip = process.env.E2E_SERVER === 'demo' && 'needs a server without Earth Engine';

test('map mode: result and sample layers, draw and rename, Fit, Compare, Fetch disabled, Send to chat, 1280 without overlaps', { skip }, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const tiles = [];
  page.on('response', (res) => {
    if (res.url().includes('/v1/tiles/')) tiles.push({ url: decodeURIComponent(res.url()), status: res.status() });
  });
  try {
    // Setup: the sample in a conversation, the NDWI change, On map (docked at 1440).
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    const input = page.getByLabel('Message', { exact: true });
    await input.fill(NDWI);
    await input.press('Enter');
    const message = page.getByTestId('analysis-message').first();
    await message.and(page.locator('[data-phase="done"]')).waitFor({ timeout: ANALYSIS_TIMEOUT });
    const card = message.getByRole('region', { name: 'Water change · NDWI' });
    await card.getByRole('group', { name: 'Result actions' }).getByRole('button', { name: 'On map' }).click();

    // Map mode from the rail (the result layer lives in this page's store).
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Map' }).click();
    await page.waitForURL(/\/app\/map$/);
    const panel = page.getByRole('complementary', { name: 'Imagery' });
    const layers = panel.getByRole('region', { name: 'Map layers' });
    await layers.getByTestId('map-layer-row').filter({ hasText: 'kakhovka_hls_l30' }).waitFor();
    await page.waitForFunction(() => window.__dt4lcMap);
    await settle(page);

    // The sample in true colour (its own tiles, no style parameter) with its footprint,
    // under the result layer.
    const sampleRow = layers.getByTestId('map-layer-row').filter({ hasText: 'kakhovka_hls_l30' });
    assert.match(await sampleRow.innerText(), /HLS L30 · true colour · 2023-06-09/);
    assert.ok(tiles.some((t) => /hlsl_20230609\.tif/.test(t.url) && t.status === 200), 'sample tiles load');
    const stack = await page.evaluate(() => window.__dt4lcMap.getStyle().layers.map((l) => l.id));
    const sampleId = stack.find((id) => id.startsWith('sample-') && !id.includes('footprint'));
    const resultId = stack.find((id) => id.startsWith('result-') && id.endsWith('-difference'));
    assert.ok(sampleId && resultId, `both rasters on the map (${stack.join(', ')})`);
    assert.ok(stack.indexOf(sampleId) < stack.indexOf(resultId), 'the sample is under the result');
    assert.ok(stack.includes(`${sampleId}-footprint-line`), 'the footprint is drawn');
    const footprint = await page.evaluate((id) => window.__dt4lcMap.getSource(`${id}-footprint`).serialize().data.geometry.coordinates[0], sampleId);
    assert.deepEqual(footprint.map(([x, y]) => [x.toFixed(3), y.toFixed(3)]), [
      ['32.389', '46.663'], ['32.690', '46.664'], ['32.691', '46.457'], ['32.391', '46.455'], ['32.389', '46.663'],
    ]);

    // The legend: the style's ramp and range, and the real class areas.
    const legend = page.getByTestId('map-legend');
    assert.equal(await legend.getByTestId('map-legend-title').innerText(), 'Water change · NDWI difference');
    assert.equal(await legend.getByTestId('map-legend-ticks').innerText(), '−0.5 · 0 · +0.5');
    assert.equal((await legend.getByTestId('map-legend-classes').innerText()).replace(/\s+/g, ' '), 'lost 0.6 km² no change 373 km² gained 157 km²');

    // Draw a rectangle with the toolbar: a real mouse drag over the north of the raster.
    const mapBox = await page.locator('section[aria-label="Map"]').boundingBox();
    await page.getByRole('toolbar', { name: 'Draw' }).getByRole('button', { name: 'Rectangle' }).click();
    const [x0, y0] = [mapBox.x + mapBox.width * 0.38, mapBox.y + mapBox.height * 0.22];
    const [x1, y1] = [mapBox.x + mapBox.width * 0.62, mapBox.y + mapBox.height * 0.5];
    await page.mouse.move(x0, y0);
    await page.mouse.down();
    await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 5 });
    await page.mouse.move(x1, y1, { steps: 5 });
    await page.mouse.up();
    const regionCard = panel.getByTestId('region-card');
    await regionCard.waitFor();
    assert.equal(await page.getByRole('button', { name: 'Pan' }).getAttribute('aria-pressed'), 'true');

    // Rename it in place.
    await regionCard.getByRole('button', { name: 'Rename' }).click();
    await regionCard.getByRole('textbox', { name: 'Region name' }).fill('Reservoir-north');
    await regionCard.getByRole('textbox', { name: 'Region name' }).press('Enter');
    const detail = await regionCard.getByTestId('region-detail').innerText();
    const area = Number(/^(\d+) km²/.exec(detail)?.[1]);
    assert.ok(area > 20 && area < 2500, `region area (${detail})`);
    assert.match(detail, /^\d+ km² · 46\.\d\d N 32\.\d\d E$/);
    assert.equal(await page.getByTestId('region-label').innerText(), `Reservoir-north · ${area} km²`);

    // HLS L30, 2023-06-01 → 2023-06-09, cloud 20 %, B2–B6 (B7 off).
    await panel.getByRole('button', { name: 'Sentinel-2' }).click();
    await panel.getByRole('button', { name: 'HLS L30' }).click();
    assert.equal(await panel.getByTestId('dataset-description').innerText(), 'NASA HLS L30 (Landsat 8/9) · 30 m · revisit 8 days');
    await panel.getByLabel('Start date').fill('2023-06-01');
    await panel.getByLabel('End date').fill('2023-06-09');
    const cloud = panel.getByRole('slider', { name: 'Max cloud cover' });
    await cloud.focus();
    await cloud.press('Home');
    for (let i = 0; i < 4; i++) await cloud.press('ArrowRight');
    assert.equal(await panel.getByTestId('cloud-value').innerText(), '20 %');
    const bands = panel.getByRole('group', { name: 'Bands' });
    await panel.getByRole('button', { name: 'Clear' }).click();
    for (const b of ['B2 Blue', 'B3 Green', 'B4 Red', 'B5 NIR', 'B6 SWIR1']) await bands.getByRole('button', { name: b }).click();
    assert.equal(await bands.getByRole('button', { name: 'B7 SWIR2' }).getAttribute('aria-pressed'), 'false');

    // Three layers: the result, the region and the sample.
    assert.equal(await layers.getByText(/^Layers · \d+$/).innerText(), 'LAYERS · 3');
    assert.deepEqual(
      (await layers.getByTestId('map-layer-row').allInnerTexts()).map((t) => t.split('\n')[0]),
      ['Water change · NDWI', 'Reservoir-north', 'kakhovka_hls_l30'],
    );

    // Fetch to map: disabled, with the reason from GET /v1/capabilities.
    assert.equal(await panel.getByRole('button', { name: 'Fetch to map' }).isDisabled(), true);
    assert.equal(await panel.getByTestId('fetch-reason').innerText(), "Earth Engine isn't configured on this server");

    await settle(page);
    await page.screenshot({ path: outPath('map-mode.png') });

    // Fit to result: pan away, then back to the result's bounds.
    const fitted = await view(page);
    await page.mouse.move(mapBox.x + mapBox.width * 0.8, mapBox.y + mapBox.height * 0.75);
    await page.mouse.down();
    await page.mouse.move(mapBox.x + mapBox.width * 0.2, mapBox.y + mapBox.height * 0.75, { steps: 8 });
    await page.mouse.up();
    await page.getByRole('button', { name: 'Zoom out' }).click();
    await page.waitForTimeout(600);
    const away = await view(page);
    assert.ok(Math.abs(away.lng - fitted.lng) > 0.05 && away.zoom < fitted.zoom - 0.5, 'panned and zoomed away');
    await page.getByRole('button', { name: 'Fit to result' }).click();
    await page.waitForTimeout(1200);
    const back = await view(page);
    assert.ok(Math.abs(back.lng - 32.54) < 0.02 && Math.abs(back.lat - 46.56) < 0.02, `centred on the result (${back.lng}, ${back.lat})`);
    assert.ok(back.bounds[0] <= 32.39 && back.bounds[2] >= 32.69 && back.bounds[1] <= 46.456 && back.bounds[3] >= 46.664, 'the result fits');
    assert.ok(Math.abs(back.zoom - away.zoom) > 0.5, 'Fit to result changes the view');

    // Compare: the result (opacity 100, region hidden) left of the divider, the sample right of it.
    await layers.getByRole('slider', { name: 'Opacity of Water change · NDWI' }).focus();
    await page.keyboard.press('End');
    await layers.getByRole('button', { name: 'Hide Reservoir-north' }).click();
    await page.getByRole('button', { name: 'Compare' }).click();
    const divider = page.getByTestId('compare-divider');
    await divider.waitFor();
    const handle = page.getByRole('slider', { name: 'Compare divider' });
    assert.equal(await handle.getAttribute('aria-valuetext'), 'Water change · NDWI on the left, kakhovka_hls_l30 on the right');
    await settle(page);
    await page.waitForTimeout(800);
    const dBox = await divider.boundingBox();
    assert.ok(Math.abs(dBox.x + dBox.width / 2 - (mapBox.x + mapBox.width / 2)) < 3, 'the divider starts in the middle');

    await page.screenshot({ path: outPath('map-mode-compare.png') });
    const cy = mapBox.y + mapBox.height * 0.55;
    const leftPts = around(mapBox.x + mapBox.width * 0.4, cy);
    const rightPts = around(mapBox.x + mapBox.width * 0.6, cy);
    const halves = await pixelsAt(page, [...leftPts, ...rightPts]);
    const leftOff = median(halves.slice(0, 25).map(offRamp));
    const rightOff = median(halves.slice(25).map(offRamp));
    assert.ok(leftOff < 12, `left of the divider: the change result (median ${leftOff.toFixed(1)} off its ramp)`);
    assert.ok(rightOff > 30, `right of the divider: the true-colour sample (median ${rightOff.toFixed(1)} off the ramp)`);

    // Move the divider with the keyboard to the left edge: the left points now show the sample.
    await handle.focus();
    await page.keyboard.press('Home');
    assert.equal(await handle.getAttribute('aria-valuenow'), '0');
    await page.waitForTimeout(300);
    const moved = median((await pixelsAt(page, leftPts)).map(offRamp));
    assert.ok(moved > 30, `after moving the divider the left points show the sample (median ${moved.toFixed(1)} off the ramp)`);
    // And drag it back across with the mouse: the right points show the result.
    const hBox = await divider.boundingBox();
    await page.mouse.move(hBox.x + hBox.width / 2, mapBox.y + mapBox.height * 0.3);
    await page.mouse.down();
    await page.mouse.move(mapBox.x + mapBox.width * 0.9, mapBox.y + mapBox.height * 0.3, { steps: 8 });
    await page.mouse.up();
    assert.ok(Number(await handle.getAttribute('aria-valuenow')) >= 85, 'the divider follows the drag');
    await page.waitForTimeout(300);
    const dragged = median((await pixelsAt(page, rightPts)).map(offRamp));
    assert.ok(dragged < 12, `after the drag the right points show the result (median ${dragged.toFixed(1)} off its ramp)`);
    await page.getByRole('button', { name: 'Compare' }).click();
    await divider.waitFor({ state: 'detached' });
    await layers.getByRole('button', { name: 'Show Reservoir-north' }).click();
    const opacity = layers.getByRole('slider', { name: 'Opacity of Water change · NDWI' });
    await opacity.focus();
    for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowLeft');
    assert.equal(await layers.getByTestId('map-layer-row').first().innerText().then((t) => t.trim().split('\n').pop()), '80');

    // Send to chat: the Workspace with the region chip in the composer.
    await panel.getByRole('button', { name: 'Send to chat' }).click();
    await page.waitForURL(/\/app$/);
    const chip = page.getByTestId('region-chip');
    await chip.waitFor();
    assert.match((await chip.innerText()).replace(/\s+/g, ' '), new RegExp(`Region · Reservoir-north .*${area} km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud`));

    // 1280 × 800: map mode is the only map; nothing overlaps.
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Map' }).click();
    await page.waitForURL(/\/app\/map$/);
    await page.getByTestId('map-legend').waitFor();
    await page.getByRole('button', { name: 'Fit to result' }).click();
    await page.waitForTimeout(1200);
    await settle(page);
    await page.screenshot({ path: outPath('map-mode-1280.png') });
    const boxes = {
      panel: await panel.boundingBox(),
      toolbar: await page.getByRole('toolbar', { name: 'Draw' }).boundingBox(),
      controls: await page.getByTestId('map-controls').boundingBox(),
      legend: await page.getByTestId('map-legend').boundingBox(),
      zoom: await page.locator('.maplibregl-ctrl-group').first().boundingBox(),
      scale: await page.locator('.maplibregl-ctrl-scale').first().boundingBox(),
      attribution: await page.locator('.maplibregl-ctrl-attrib').first().boundingBox(),
    };
    // The presets row under the region card, inside the panel.
    const presets = panel.getByRole('group', { name: 'Region presets' });
    assert.deepEqual(await presets.getByRole('button').allInnerTexts(), ['Kakhovka dam', 'Brienz', 'Upload GeoJSON']);
    const presetsBox = await presets.boundingBox();
    const panelBox = boxes.panel;
    assert.ok(presetsBox.x >= panelBox.x && presetsBox.x + presetsBox.width <= panelBox.x + panelBox.width, 'the presets fit the panel');
    assert.ok(presetsBox.y > (await regionCard.boundingBox()).y, 'the presets are under the region card');
    // Regions in light amber (--region), on the map and in the swatches.
    const regionId = await page.evaluate(() => JSON.parse(localStorage.getItem('dt4lc-app-storage')).state.selectedRegion.id);
    const amber = (rgb) => {
      const [r, g, b] = rgb.match(/\d+/g).map(Number);
      return r > 200 && g > 150 && r > g && g > b + 40;
    };
    const outline = await page.evaluate((id) => window.__dt4lcMap.getPaintProperty(`region-outline-${id}`, 'line-color'), regionId);
    assert.ok(amber(String(outline)), `the region outline is amber (${outline})`);
    // --region in the light theme (src/index.css).
    const swatch = await regionCard.locator('span[aria-hidden]').first().evaluate((el) => getComputedStyle(el).backgroundColor);
    assert.equal(swatch, 'oklch(0.85 0.12 85)', 'the region card swatch is the region token');
    for (const [name, b] of Object.entries(boxes)) {
      assert.ok(b && b.width > 0 && b.x >= 0 && b.y >= 0 && b.x + b.width <= 1280 && b.y + b.height <= 800, `${name} on screen (${JSON.stringify(b)})`);
    }
    const names = Object.keys(boxes);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        assert.ok(!intersects(boxes[names[i]], boxes[names[j]]), `${names[i]} and ${names[j]} don't overlap`);
      }
    }

    // Presets: Kakhovka is the sample's footprint here; Brienz zooms to the lake;
    // a GeoJSON that isn't a polygon is refused, a polygon becomes the region.
    await presets.getByRole('button', { name: 'Kakhovka dam' }).click();
    assert.equal(await regionCard.locator('.font-medium').first().innerText(), 'Kakhovka sample area');
    assert.match(await regionCard.getByTestId('region-detail').innerText(), / · 46\.56 N 32\.54 E$/);
    await presets.getByRole('button', { name: 'Brienz' }).click();
    assert.equal(await regionCard.locator('.font-medium').first().innerText(), 'Lake Brienz');
    await page.waitForTimeout(1200);
    const brienz = await view(page);
    assert.ok(Math.abs(brienz.lng - 7.97) < 0.05 && Math.abs(brienz.lat - 46.73) < 0.05, `the map zooms to Lake Brienz (${brienz.lng}, ${brienz.lat})`);
    const geojson = panel.getByLabel('GeoJSON file');
    await geojson.setInputFiles({ name: 'pin.geojson', mimeType: 'application/geo+json', buffer: Buffer.from(JSON.stringify({ type: 'Point', coordinates: [8, 46.7] })) });
    assert.equal(await panel.getByTestId('geojson-error').innerText(), 'pin.geojson has no Polygon or MultiPolygon. Upload a polygon Feature, FeatureCollection or geometry.');
    const shore = { type: 'Feature', properties: { name: 'Bönigen shore' }, geometry: { type: 'Polygon', coordinates: [[[7.88, 46.68], [7.93, 46.68], [7.93, 46.71], [7.88, 46.71], [7.88, 46.68]]] } };
    await geojson.setInputFiles({ name: 'shore.geojson', mimeType: 'application/geo+json', buffer: Buffer.from(JSON.stringify(shore)) });
    await regionCard.getByText('Bönigen shore').waitFor();
    assert.equal(await panel.getByTestId('geojson-error').count(), 0);
    assert.equal(await page.getByTestId('region-label').innerText().then((t) => t.split(' · ')[0]), 'Bönigen shore');

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
