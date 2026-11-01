// Workspace empty states at 1440x900:
// the rail, the conversation list with the guest badge, the chat column and the
// docked map, with the sample attached (?sample=1) and without (?sample=0).
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function openWorkspace(query) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  await page.goto(`${h.baseURL}/app?${query}`);

  const rail = page.getByRole('navigation', { name: 'Main' });
  for (const name of ['Chat', 'Map', 'Library', 'Can do']) {
    await rail.getByRole('link', { name, exact: true }).waitFor();
  }
  await rail.getByRole('button', { name: 'Account: Guest' }).waitFor();
  await page.getByRole('complementary', { name: 'Conversations' }).waitFor();
  await page.getByRole('main', { name: 'Chat' }).waitFor();
  await page.getByTestId('session-badge').waitFor();

  const map = page.getByRole('region', { name: 'Map', exact: true });
  await map.locator('canvas.maplibregl-canvas').waitFor();
  await map.getByRole('link', { name: 'Open full map ↗' }).waitFor();
  await map.getByRole('region', { name: 'Layers' }).waitFor();
  return { context, page, errors };
}

async function screenshot(page, name) {
  await page.evaluate(() => document.fonts.ready);
  // Basemap tiles come from the internet; take the picture whether or not they arrive.
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.screenshot({ path: outPath(name) });
}

test('empty Workspace with the sample attached (?sample=1)', async () => {
  const { context, page, errors } = await openWorkspace('sample=1');
  try {
    const chip = page.getByTestId('attachment-chip');
    await chip.waitFor();
    assert.equal(await chip.count(), 1, 'the pair is one chip');
    assert.match(await chip.innerText(), /kakhovka_hls_l30/);
    assert.match(await chip.innerText(), /HLS L30 · B2–B6 · 30 m · .* · 2023-06-01 → 06-09/);
    assert.match(await page.getByTestId('session-badge').innerText(), /^Guest\. \d+ of \d+ analyses left this hour · kept 24 h · no account\.$/);
    await page.getByRole('heading', { name: 'What do you want to know about this area?' }).waitFor();
    assert.equal(await page.getByRole('list', { name: 'Suggestions' }).getByRole('button').count(), 5);
    // The parameter is consumed, so a reload does not attach the sample again.
    await page.waitForURL((url) => !url.searchParams.has('sample'));

    await screenshot(page, 'workspace-empty-with-sample.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('empty Workspace without the sample (?sample=0)', async () => {
  const { context, page, errors } = await openWorkspace('sample=0');
  try {
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
    await page.getByText('No conversations yet.', { exact: false }).waitFor();
    assert.equal(await page.getByTestId('attachment-chip').count(), 0);
    for (const name of ['+ Attach the Kakhovka sample', 'Upload GeoTIFF', 'Pick a region on the map']) {
      await page.getByRole('button', { name, exact: true }).waitFor();
    }
    const suggestions = page.getByRole('list', { name: 'Suggestions' }).getByRole('button');
    assert.deepEqual(
      (await suggestions.allInnerTexts()).map((t) => t.replace(/\s+/g, ' ')),
      [
        'Compare water before and after the dam breach ~50 s',
        'Detect field boundaries east of the river ~2 min',
        'Classify land cover on the flooded fields ~90 s',
      ],
    );
    assert.equal(
      await page.getByLabel('Message', { exact: true }).getAttribute('placeholder'),
      'Describe an analysis… or drop GeoTIFFs here',
    );

    await screenshot(page, 'workspace-empty-no-sample.png');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
