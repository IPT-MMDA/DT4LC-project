// The composer against the real API (guest + E2E_LLM=none, so the assistant is offline):
// design/screens/workspace-assistant-offline.png (the offline line, sample attached) and
// workspace-region-from-map.png (a drawn region attached with From map, as a chip); and
// the + upload and the drop zone, which vitest cannot reach.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const OFFLINE = 'Assistant offline — analyses still run; answers are limited to the catalog.';

// About 410 km² of the Lower Dnipro floodplain, inside the sample's footprint.
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

async function newPage() {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  return { context, page, errors: trackPageErrors(page) };
}

async function screenshot(page, name) {
  await page.evaluate(() => document.fonts.ready);
  // Basemap tiles come from the internet; take the picture whether or not they arrive.
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.screenshot({ path: outPath(name) });
}

test('assistant offline: one line above the controls, with the sample attached', async () => {
  const { context, page, errors } = await newPage();
  try {
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByRole('heading', { name: 'What do you want to know about this area?' }).waitFor();
    await page.getByTestId('attachment-chip').waitFor();
    const line = page.getByTestId('assistant-offline');
    await line.waitFor();
    assert.equal(await line.innerText(), OFFLINE);
    // Five suggestions with the sample, NDMI composed, Prithvi named.
    const suggestions = page.getByRole('list', { name: 'Suggestions' }).getByRole('button');
    assert.equal(await suggestions.count(), 5);
    assert.match(await suggestions.nth(3).innerText(), /Map moisture change with NDMI between the two dates\s+Composed\s+~60 s/);
    assert.match(await suggestions.nth(4).innerText(), /Fill the cloud gaps in the 9 June image\s+Prithvi · ~3 min/);

    await screenshot(page, 'workspace-assistant-offline.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('region from map: From map attaches the selected drawn region as a chip in the composer', async () => {
  const { context, page, errors } = await newPage();
  try {
    // A region drawn earlier on the map: the map store persists it in localStorage.
    await context.addInitScript((region) => {
      localStorage.setItem(
        'dt4lc-app-storage',
        JSON.stringify({ state: { drawnRegions: [region], selectedRegion: region }, version: 0 }),
      );
    }, REGION);
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();

    // A conversation about the sample first, as in the screenshot.
    const input = page.getByLabel('Message', { exact: true });
    await input.fill('Compare water before and after the dam breach');
    await input.press('Enter');
    await page.getByTestId('analysis-message').waitFor();
    assert.equal(await page.getByTestId('user-message').getByTestId('attachment-chip').count(), 1);

    await page.getByRole('button', { name: 'From map' }).click();
    const chip = page.getByTestId('region-chip');
    await chip.waitFor();
    // The sample is in the conversation, so its dates apply; the area is turf's.
    assert.match(
      (await chip.innerText()).replace(/\s+/g, ' '),
      /^Region · Dnipro floodplain 4\d\d km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud ×$/,
    );
    assert.equal(await input.getAttribute('placeholder'), 'Ask a follow-up…');
    assert.equal(await chip.getByRole('button', { name: 'Remove region' }).count(), 1);

    await screenshot(page, 'workspace-region-from-map.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('+ uploads a GeoTIFF and a drop on the chat column attaches another', async () => {
  const { context, page, errors } = await newPage();
  try {
    const sample = (name) => fileURLToPath(new URL(`../../../../resources/kahovka_data/${name}`, import.meta.url));
    await page.goto(`${h.baseURL}/app?sample=0`);
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();

    // + opens the file picker; the hidden input takes the file.
    const chooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Attach GeoTIFF' }).click();
    await (await chooser).setFiles(sample('hlsl_20230601.tif'));
    const chips = page.getByTestId('attachment-chip');
    await chips.first().waitFor();
    assert.match(await chips.first().innerText(), /hlsl_20230601\.tif\s+HLS L30 · B2–B6 · 30 m · [\d.]+ MB · 2023-06-01/);

    // Drop the other date on the chat column (a DataTransfer built in the page).
    const bytes = readFileSync(sample('hlsl_20230609.tif')).toString('base64');
    const transfer = await page.evaluateHandle((b64) => {
      const data = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const dt = new DataTransfer();
      dt.items.add(new File([data], 'hlsl_20230609.tif', { type: 'image/tiff' }));
      return dt;
    }, bytes);
    const column = page.getByRole('main', { name: 'Chat' });
    await column.dispatchEvent('dragover', { dataTransfer: transfer });
    await page.getByText('Drop GeoTIFFs to attach').waitFor();
    await column.dispatchEvent('drop', { dataTransfer: transfer });
    await page.getByText('Drop GeoTIFFs to attach').waitFor({ state: 'detached' });
    await chips.nth(1).waitFor();
    assert.match(await chips.nth(1).innerText(), /hlsl_20230609\.tif\s+HLS L30 · B2–B6 · 30 m · [\d.]+ MB · 2023-06-09/);
    assert.equal(await page.getByRole('heading', { name: 'What do you want to know about this area?' }).count(), 1);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
