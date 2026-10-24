// Phone (390×844) and tablet (834×1112) layouts of the six screens against the real API
// (guest + E2E_LLM=none, no Earth Engine), with reduced motion so every shot is still.
// Seeded through the UI per device: the landing first (a first-time guest), then the
// sample and an NDWI change in the Workspace, the result On map with a stored drawn
// region, then Capabilities, the Library and the result's detail page.
//
// Each screen: the document is no wider than the viewport, no element runs past an edge
// (outside the named clippers and scrollers below) and no two pieces of text overlap
// (outside the named overlays); see tests/e2e/layout.mjs. Shots go to
// out/responsive/<screen>-<device>.png, the scrolling pages at full length.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';
import { auditLayout, layoutProblems } from '../layout.mjs';

const NDWI = 'Compare water before and after the dam breach';
const ANALYSIS_TIMEOUT = 60_000;

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

const DEVICES = [
  { device: 'phone', viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true },
  { device: 'tablet', viewport: { width: 834, height: 1112 }, isMobile: true, hasTouch: true },
];

// Allowed exceptions, by name.
const ALLOW = {
  // The map's canvas, tiles and markers are clipped by the map container.
  map: { clippers: ['.maplibregl-map'], overlays: ['[data-testid="region-label"]'] },
  // The hero image drifts at scale(1.04) inside the hero, which clips it.
  landing: { clippers: ['[data-testid="landing-hero"]'] },
};

let h;
before(async () => {
  h = await setup();
  fs.mkdirSync(outPath('responsive'), { recursive: true });
});
after(async () => {
  await h?.browser.close();
});

async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

/** The audit of one screen; its problems are collected, not thrown, so one run reports every screen. */
async function check(page, report, screen, allow = {}) {
  await settle(page);
  const result = await auditLayout(page, allow);
  const problems = layoutProblems(result);
  report[screen] = { scrollWidth: result.scrollWidth, innerWidth: result.innerWidth, scrollers: result.scrollers, problems };
  return problems;
}

/** A shot of the whole page: the document's height, or the app's scroller unrolled by growing the viewport. */
async function shoot(page, name, scroller) {
  const size = page.viewportSize();
  if (scroller) {
    const extra = await page.locator(scroller).evaluate((el) => el.scrollHeight - el.clientHeight);
    if (extra > 0) {
      await page.setViewportSize({ width: size.width, height: Math.min(size.height + extra, 8000) });
      await settle(page);
    }
  }
  await page.screenshot({ path: outPath(`responsive/${name}.png`), fullPage: true });
  await page.setViewportSize(size);
  await settle(page);
}

/** The main navigation: the rail, or on a phone the menu it moves to. */
async function navigate(page, name, url) {
  const menu = page.getByRole('button', { name: 'Open menu' });
  if (await menu.isVisible()) {
    await menu.click();
    assert.equal(await menu.getAttribute('aria-expanded'), 'true');
  }
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name, exact: true }).click();
  await page.waitForURL(url);
}

const box = (locator) => locator.boundingBox();
const overlapping = (a, b) => a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;

// No hover on a touch screen: every layer row shows its visibility button and opacity slider.
async function assertLayerControlsShown(panel) {
  const rows = panel.getByRole('region', { name: 'Map layers' }).getByTestId('map-layer-row');
  const hidden = await rows.evaluateAll((els) =>
    els.flatMap((row) => [...row.querySelectorAll('button, input[type="range"]')])
      .filter((el) => getComputedStyle(el).display === 'none' || getComputedStyle(el).opacity !== '1')
      .map((el) => el.getAttribute('aria-label')),
  );
  assert.ok((await rows.count()) > 1, 'more than one layer row');
  assert.deepEqual(hidden, [], 'layer controls hidden on a touch screen');
}

for (const { device, viewport, isMobile, hasTouch } of DEVICES) {
  test(`responsive ${device} ${viewport.width}×${viewport.height}: the six screens without overflow or overlapping text`, async () => {
    const context = await h.browser.newContext({ viewport, isMobile, hasTouch, reducedMotion: 'reduce' });
    await context.addInitScript((r) => {
      if (!localStorage.getItem('dt4lc-app-storage')) {
        localStorage.setItem('dt4lc-app-storage', JSON.stringify({ state: { drawnRegions: [r], selectedRegion: r }, version: 0 }));
      }
    }, REGION);
    const page = await context.newPage();
    page.setDefaultTimeout(60_000);
    const errors = trackPageErrors(page);
    const report = {};
    const phone = device === 'phone';
    try {
      // Landing: the headline at min(44px, 11vw) on a phone, the caption under the buttons,
      // the nav behind a menu button.
      await page.goto(`${h.baseURL}/`);
      await page.getByTestId('landing').waitFor();
      const headline = page.getByRole('heading', { level: 1 });
      const buttons = page.getByRole('link', { name: 'Describe your own analysis' });
      const caption = page.getByTestId('landing-caption');
      if (phone) {
        assert.equal(await headline.evaluate((el) => getComputedStyle(el).fontSize), `${Math.min(44, viewport.width * 0.11)}px`);
        assert.ok((await box(caption)).y >= (await box(buttons)).y + (await box(buttons)).height, 'the caption is below the buttons');
        const menu = page.getByRole('button', { name: 'Menu' });
        assert.equal(await menu.getAttribute('aria-expanded'), 'false');
        await menu.click();
        assert.equal(await menu.getAttribute('aria-expanded'), 'true');
        const nav = page.getByRole('navigation', { name: 'Landing' });
        assert.deepEqual(await nav.getByRole('link').allInnerTexts(), ['Capabilities', 'How it works', 'About', 'GitHub ↗']);
        assert.equal(await nav.evaluate((el) => getComputedStyle(el).backgroundColor), 'rgb(10, 12, 16)', 'the menu panel is opaque');
        await check(page, report, 'landing-menu', { ...ALLOW.landing, scope: 'header' });
        await page.screenshot({ path: outPath('responsive/landing-phone-menu.png') });
        await page.keyboard.press('Escape');
        assert.equal(await menu.getAttribute('aria-expanded'), 'false');
        assert.equal(await nav.count(), 0);
      } else {
        assert.equal(await page.getByRole('navigation', { name: 'Landing' }).getByRole('link').count(), 4);
      }
      const lines = await headline.evaluate((el) => Math.round(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight)));
      assert.ok(lines <= 4, `the headline takes ${lines} lines`);
      await check(page, report, 'landing', ALLOW.landing);
      await shoot(page, `landing-${device}`);

      // Workspace: the sample and an NDWI change done.
      await page.goto(`${h.baseURL}/app?sample=1`);
      await page.getByTestId('attachment-chip').waitFor();
      const input = page.getByLabel('Message', { exact: true });
      await input.fill(NDWI);
      await input.press('Enter');
      const message = page.getByTestId('analysis-message').first();
      await message.and(page.locator('[data-phase="done"]')).waitFor({ timeout: ANALYSIS_TIMEOUT });
      const chat = page.getByRole('main', { name: 'Chat' });
      const conversations = page.getByRole('complementary', { name: 'Conversations' });
      if (phone) {
        // No list and no docked map beside the chat, which spans the screen; the list is in the menu.
        assert.equal(await conversations.count(), 0);
        assert.equal((await box(chat)).width, viewport.width);
        const menu = page.getByRole('button', { name: 'Open menu' });
        await menu.click();
        const drawer = page.getByRole('dialog', { name: 'Menu' });
        await drawer.getByRole('complementary', { name: 'Conversations' }).getByTestId('conversation').first().waitFor();
        await check(page, report, 'workspace-menu', { scope: '[role="dialog"]' });
        await page.screenshot({ path: outPath('responsive/workspace-phone-menu.png') });
        // The account menu opens inside the screen.
        await drawer.getByRole('button', { name: /^Account:/ }).click();
        const account = page.getByRole('menu', { name: 'Account' });
        const a = await box(account);
        assert.ok(a.x >= 0 && a.x + a.width <= viewport.width && a.y >= 0 && a.y + a.height <= viewport.height, 'the account menu is on screen');
        await page.keyboard.press('Escape');
        await drawer.getByRole('button', { name: 'Close menu' }).click();
        assert.equal(await drawer.count(), 0);
        assert.equal(await menu.getAttribute('aria-expanded'), 'false');
      } else {
        // The list collapses and comes back; the chat takes the room.
        const before = (await box(chat)).width;
        await conversations.getByRole('button', { name: 'Hide conversations' }).click();
        assert.equal(await conversations.count(), 0);
        assert.ok((await box(chat)).width > before, 'the chat widens');
        await check(page, report, 'workspace-collapsed');
        await page.screenshot({ path: outPath('responsive/workspace-tablet-collapsed.png') });
        await page.getByRole('button', { name: 'Show conversations' }).click();
        await conversations.waitFor();
        // The account menu from the rail opens inside the screen.
        await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: /^Account:/ }).click();
        const a = await box(page.getByRole('menu', { name: 'Account' }));
        assert.ok(a.x >= 0 && a.x + a.width <= viewport.width && a.y >= 0 && a.y + a.height <= viewport.height, 'the account menu is on screen');
        await page.keyboard.press('Escape');
      }
      await check(page, report, 'workspace');
      await shoot(page, `workspace-${device}`);

      // The composer with an on-screen keyboard: the layout viewport shrinks (interactive-widget=resizes-content).
      await page.setViewportSize({ width: viewport.width, height: Math.round(viewport.height * 0.55) });
      await input.focus();
      await input.fill('Classify the flooded land');
      const composer = await box(input);
      assert.ok(composer.y >= 0 && composer.y + composer.height <= viewport.height * 0.55, 'the composer stays above the keyboard');
      await input.fill('');
      await page.setViewportSize(viewport);

      // Map: the result layer from the chat and the stored region.
      await message.getByRole('group', { name: 'Result actions' }).getByRole('button', { name: 'On map' }).click();
      await navigate(page, 'Map', /\/app\/map$/);
      const panel = page.getByRole('complementary', { name: 'Imagery' });
      await page.waitForFunction(() => window.__dt4lcMap);
      await page.getByTestId('map-legend').waitFor();
      await page.evaluate(() => new Promise((resolve) => (window.__dt4lcMap.loaded() ? resolve() : window.__dt4lcMap.once('idle', resolve))));
      const mapPane = page.locator('main[aria-label="Map"]');
      const toolbar = page.getByTestId('map-toolbar');
      const controls = page.getByTestId('map-controls');
      assert.ok(!overlapping(await box(toolbar), await box(controls)), 'the draw toolbar and the map controls do not overlap');
      if (phone) {
        // A bottom sheet under a full-width map: collapsed, the map has most of the screen.
        const sheet = page.getByRole('button', { name: 'Imagery panel' });
        assert.equal(await sheet.getAttribute('aria-expanded'), 'false');
        const m = await box(mapPane);
        assert.equal(m.width, viewport.width);
        assert.ok(m.height >= viewport.height * 0.6, `the map is ${m.height} px tall`);
        assert.ok((await box(panel)).y >= m.y + m.height - 1, 'the sheet is below the map');
        await check(page, report, 'map', ALLOW.map);
        await page.screenshot({ path: outPath(`responsive/map-${device}.png`) });
        await sheet.click();
        assert.equal(await sheet.getAttribute('aria-expanded'), 'true');
        await panel.getByRole('region', { name: 'Map layers' }).getByTestId('map-layer-row').first().waitFor();
        await assertLayerControlsShown(panel);
        const open = await box(mapPane);
        assert.ok(open.height >= viewport.height * 0.3, `the map keeps ${open.height} px with the sheet open`);
        for (const el of [panel.getByRole('heading', { name: 'Imagery' }), panel.getByRole('link', { name: 'Back to chat' })]) {
          assert.ok(!overlapping(await box(el), await box(toolbar)) && !overlapping(await box(el), await box(controls)), 'the sheet header is clear of the map controls');
        }
        await check(page, report, 'map-sheet', ALLOW.map);
        await page.screenshot({ path: outPath('responsive/map-phone-sheet.png') });
        await sheet.click();
      } else {
        // A narrower side panel beside the map.
        const p = await box(panel);
        assert.ok(p.width <= 280, `the panel is ${p.width} px wide`);
        assert.ok((await box(mapPane)).width >= viewport.width - 60 - 280, 'the map takes the rest');
        await panel.getByRole('region', { name: 'Map layers' }).getByTestId('map-layer-row').first().waitFor();
        await assertLayerControlsShown(panel);
        await check(page, report, 'map', ALLOW.map);
        await page.screenshot({ path: outPath(`responsive/map-${device}.png`) });
      }

      // Capabilities: one column of cards on a phone, two on a tablet.
      await navigate(page, 'Can do', /\/app\/capabilities$/);
      await page.getByTestId('builtin-item').first().waitFor();
      const columns = new Set(await page.getByTestId('builtin-item').evaluateAll((els) => els.map((el) => Math.round(el.getBoundingClientRect().left))));
      assert.equal(columns.size, phone ? 1 : 2, `builtin cards in ${columns.size} columns`);
      await check(page, report, 'capabilities');
      await shoot(page, `capabilities-${device}`, '[data-testid="capabilities-page"]');

      // Library: cards on a phone (thumbnail, name, type, status, when, the action); the table without its secondary columns on a tablet.
      await navigate(page, 'Library', /\/app\/library$/);
      const row = page.getByTestId('library-row').first();
      await row.waitFor();
      if (phone) {
        assert.equal(await page.getByRole('table').count(), 0);
        for (const id of ['library-thumbnail', 'library-name', 'library-type', 'library-status', 'library-when']) {
          assert.ok(await row.getByTestId(id).isVisible(), `${id} on the card`);
        }
        assert.ok(await row.getByRole('button', { name: 'Map' }).isVisible(), 'the row action on the card');
      } else {
        const headers = await page.getByRole('columnheader').allInnerTexts();
        assert.ok(!headers.some((t) => /conversation/i.test(t)), headers.join(' | '));
      }
      await check(page, report, 'library');
      await shoot(page, `library-${device}`, '[data-testid="library-page"]');

      // Result detail: one column, the side cards under the maps; download names wrap.
      await row.getByTestId('library-name').click();
      await page.waitForURL(/\/app\/results\//);
      await page.getByTestId('detail-downloads').waitFor();
      await page.waitForFunction(() => [...document.querySelectorAll('img[data-testid="detail-image"]')].every((i) => i.complete && i.naturalWidth > 0));
      const side = await box(page.getByRole('complementary', { name: 'About this result' }));
      const stats = await box(page.getByRole('region', { name: 'Statistics' }));
      assert.ok(side.y >= stats.y + stats.height, 'the side column is under the maps and statistics');
      const clipped = await page.getByTestId('download-name').evaluateAll((els) => els.filter((el) => el.scrollWidth > el.clientWidth + 1).map((el) => el.textContent));
      assert.deepEqual(clipped, [], 'every download name shows in full');
      await check(page, report, 'result-detail');
      await shoot(page, `result-detail-${device}`, '[data-testid="result-detail"]');

      const failures = Object.entries(report).flatMap(([screen, r]) => r.problems.map((p) => `${screen}: ${p}`));
      assert.deepEqual(failures, []);
      assert.deepEqual(errors, []);
    } finally {
      fs.writeFileSync(outPath(`responsive/report-${device}.json`), `${JSON.stringify(report, null, 2)}\n`);
      await context.close();
    }
  });
}
