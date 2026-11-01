// A member's analysis that first downloads a model, caught mid-download:
// the Preparing model row while the assistant is offline. Needs E2E_DEV_ROLE=member
// (`npm run e2e:workspace:member`; assistant offline, E2E_LLM=none).
//
// Nothing comes from the internet: the runner points HF_ENDPOINT at fake-hf.mjs, which
// streams zeros slowly into the temp model cache. Prithvi needs six HLS bands and the
// sample has five, so the flow uploads a copy of the 9 June crop with a sixth band
// (B7 = B6). Delineate can't stand in: its download fetches a GitHub zip first.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const PRITHVI = 'Fill the cloud gaps in the 9 June image';
const PRITHVI_FILE = '/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt';
const member = { skip: process.env.E2E_DEV_ROLE !== 'member' };
const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url));

// Copy a 5-band HLS crop with B6 repeated as a sixth band named B7.
const SIX_BANDS = `
import sys, rasterio
with rasterio.open(sys.argv[1]) as src:
    data, profile, tags, names = src.read(), src.profile, src.tags(), src.descriptions
with rasterio.open(sys.argv[2], "w", **(profile | {"count": 6})) as dst:
    for i in range(5):
        dst.write(data[i], i + 1)
        dst.set_band_description(i + 1, names[i])
    dst.write(data[4], 6)
    dst.set_band_description(6, "B7")
    dst.update_tags(**tags)
`;

let h;
let tmp;
before(async () => {
  if (member.skip) return;
  h = await setup();
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dt4lc-e2e-six-'));
});
after(async () => {
  if (member.skip) return;
  if (tmp) fs.rmSync(tmp, { recursive: true, force: true });
  await h?.browser.close();
});

function sixBandCrop() {
  const file = path.join(tmp, 'hls6_20230609.tif');
  const src = path.join(repoRoot, 'resources/kahovka_data/hlsl_20230609.tif');
  const run = spawnSync('uv', ['run', 'python', '-c', SIX_BANDS, src, file], { cwd: repoRoot, encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  return file;
}

async function fakeHFRequests() {
  return (await fetch(`${process.env.E2E_FAKE_HF_URL}/__e2e/requests`)).json();
}

test('preparing model: the Preparing model row downloads from the fake Hub, with bytes and time left', member, async () => {
  const file = sixBandCrop();
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app?sample=0`);
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
    await page.getByRole('button', { name: 'Account: Member' }).waitFor();
    const chooser = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Attach GeoTIFF' }).click();
    await (await chooser).setFiles(file);
    const chip = page.getByTestId('attachment-chip');
    await chip.waitFor();
    assert.match(await chip.innerText(), /hls6_20230609\.tif\s+HLS L30 · B2–B7 · 30 m/);

    const input = page.getByLabel('Message', { exact: true });
    await input.fill(PRITHVI);
    await input.press('Enter');

    const message = page.getByTestId('analysis-message').first();
    const prepare = message.locator('[data-testid="step-row"][data-kind="prepare"]');
    // Mid-download: some bytes are in, an ETA is known.
    await prepare.getByText(/MB done · .+ left$/).waitFor({ timeout: 60_000 });
    assert.equal(await prepare.getAttribute('data-status'), 'running');
    assert.match(await prepare.innerText(), /^Preparing model · Prithvi EO v1 \(100M\)/);
    assert.match(
      await prepare.getByTestId('download-progress').innerText(),
      /^Downloading Prithvi EO v1 \(100M\) · 450 MB · \d+ MB done · [\d.]+ (s|min) left$/,
    );
    const bar = Number(await prepare.getByRole('progressbar').getAttribute('aria-valuenow'));
    assert.ok(bar >= 1 && bar < 100, `download bar at ${bar} %`);
    assert.match(await message.innerText(), /Step 2 of 4/);
    const statuses = await message.getByTestId('step-row').evaluateAll((rows) => rows.map((r) => r.dataset.status));
    assert.deepEqual(statuses, ['done', 'running', 'queued', 'queued']);
    assert.equal(await page.getByTestId('assistant-offline').count(), 1);
    // On a local server (DT4LC_DEPLOYMENT unset) with no assistant.
    await page.getByTestId('session-badge').filter({ hasText: 'Local server' }).waitFor();
    assert.equal(await page.getByTestId('session-badge').innerText(), 'Member. Local server · no analysis limit · assistant offline.');
    assert.deepEqual(await fakeHFRequests(), [
      { method: 'HEAD', path: PRITHVI_FILE },
      { method: 'GET', path: PRITHVI_FILE },
    ]);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForLoadState('networkidle', { timeout: 5_000 }).catch(() => {});
    await page.screenshot({ path: outPath('workspace-preparing-model-offline.png') });

    // Cancel stops the download before Prithvi runs on the fake weights.
    await message.getByRole('button', { name: 'Cancel' }).click();
    await message.and(page.locator('[data-phase="cancelled"]')).waitFor({ timeout: 15_000 });
    assert.equal(await prepare.getAttribute('data-status'), 'cancelled');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
