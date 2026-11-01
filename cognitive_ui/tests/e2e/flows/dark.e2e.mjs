// Every screen in the dark theme at 1440×900, against the real API: out/dark/<screen>.png,
// with the stored theme dark from the first page (the landing included). `npm run e2e:dark`
// runs two groups: as a guest (E2E_LLM=none) the Landing, the Workspace empty, with a
// proposal, done and failed, Map, Capabilities, Library, Result detail and Not found; as an
// operator (E2E_SETTINGS=1, E2E_LLM=fake) Settings and the Workspace mid-run.
// Each screen also passes tests/e2e/layout.mjs (no overflow, no overlapping text); the
// audits go to out/dark/report-<group>.json.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { after, before, test } from 'node:test';
import { outPath, resetLLMRules, setLLMRules, setup, trackPageErrors } from '../harness.mjs';
import { auditLayout, layoutProblems } from '../layout.mjs';
import * as screens from '../screens.mjs';

const guest = { skip: (process.env.E2E_DEV_ROLE || 'guest') !== 'guest' || process.env.E2E_LLM !== 'none' };
const operator = { skip: process.env.E2E_DEV_ROLE !== 'operator' || process.env.E2E_SETTINGS !== '1' || process.env.E2E_LLM !== 'fake' };
const INTERPRETATION_MS = 8000;

// Allowed exceptions, by name (as in flows/responsive.e2e.mjs).
const ALLOW = {
  // The map's canvas, tiles and markers are clipped by the map container.
  map: { clippers: ['.maplibregl-map'], overlays: ['[data-testid="region-label"]'] },
  // The hero image drifts at scale(1.04) inside the hero, which clips it.
  landing: { clippers: ['[data-testid="landing-hero"]'] },
  // The save bar is sticky: the sections scroll under it.
  settings: { overlays: ['[data-testid="settings-savebar"]'] },
};

let h;
before(async () => {
  if (guest.skip && operator.skip) return;
  h = await setup();
  fs.mkdirSync(outPath('dark'), { recursive: true });
});
after(async () => {
  await h?.browser.close();
});

/** The dark shot of the screen that is open, and its layout audit. */
async function shoot(page, report, screen, allow = {}, { fullPage = false } = {}) {
  await screens.settle(page);
  const theme = await page.evaluate(() => document.documentElement.dataset.theme);
  const result = await auditLayout(page, allow);
  report[screen] = { theme, problems: [...(theme === 'dark' ? [] : [`data-theme is ${theme}`]), ...layoutProblems(result)] };
  await page.screenshot({ path: outPath(`dark/${screen}.png`), fullPage });
}

const failures = (report) => Object.entries(report).flatMap(([screen, r]) => r.problems.map((p) => `${screen}: ${p}`));

test('dark: every guest screen', guest, async () => {
  const context = await screens.newContext(h.browser, { theme: 'dark' });
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);
  const errors = trackPageErrors(page);
  const report = {};
  try {
    await screens.landing(page, h.baseURL);
    await shoot(page, report, 'landing', ALLOW.landing);
    await shoot(page, report, 'landing-full', ALLOW.landing, { fullPage: true });

    // Workspace: empty, then a proposal and a done analysis in one conversation.
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await page.getByRole('heading', { name: 'What do you want to know about this area?' }).waitFor();
    await shoot(page, report, 'workspace-empty');

    await screens.send(page, screens.NDMI);
    const proposal = await screens.analysis(page, 'proposal');
    await proposal.getByTestId('proposal').waitFor();
    await shoot(page, report, 'workspace-proposal');

    await screens.send(page, screens.NDWI);
    const done = await screens.analysis(page, 'done', 1);
    await done.getByTestId('legend').waitFor();
    await done.getByTestId('result-card').scrollIntoViewIfNeeded();
    await shoot(page, report, 'workspace-done');

    await screens.mapWithLayers(page, done);
    await shoot(page, report, 'map', ALLOW.map);

    await screens.capabilities(page);
    await shoot(page, report, 'capabilities');

    await screens.library(page);
    await shoot(page, report, 'library');

    await screens.resultDetail(page);
    await shoot(page, report, 'result-detail');

    await page.goto(`${h.baseURL}/app/no-such-page`);
    await page.getByRole('heading', { level: 1 }).waitFor();
    await shoot(page, report, 'not-found');

    // Failed: a new conversation on the stored region, without Earth Engine.
    await screens.navigate(page, 'Chat', /\/app$/);
    await page.getByRole('complementary', { name: 'Conversations' }).getByRole('button', { name: '+ New' }).click();
    await page.getByRole('button', { name: 'From map' }).click();
    await page.getByTestId('region-chip').waitFor();
    await screens.send(page, screens.NDWI);
    const failed = await screens.analysis(page, 'failed');
    await failed.getByTestId('failure-code').waitFor();
    await shoot(page, report, 'workspace-failed');

    assert.deepEqual(failures(report), []);
    assert.deepEqual(errors, []);
  } finally {
    fs.writeFileSync(outPath('dark/report-guest.json'), `${JSON.stringify(report, null, 2)}\n`);
    await context.close();
  }
});

test('dark: Settings and the Workspace mid-run, as an operator', operator, async () => {
  // Only the interpretation step's LLM call is slow, so the run is caught on its last step.
  await setLLMRules({
    default: 'This is the E2E fake LLM. No rule matched this request.',
    rules: [{ match: /interpreting satellite imagery analysis results/, reply: 'Open water grew below the dam.', delay_ms: INTERPRETATION_MS }],
  });
  const context = await screens.newContext(h.browser, { theme: 'dark', region: false });
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);
  const errors = trackPageErrors(page);
  const report = {};
  try {
    await screens.settings(page, h.baseURL);
    await shoot(page, report, 'settings', ALLOW.settings);

    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await screens.send(page, screens.NDWI);
    const message = page.getByTestId('analysis-message').first();
    const interpretation = message.getByTestId('step-row').filter({ hasText: 'Interpretation' });
    await interpretation.and(page.locator('[data-status="running"]')).waitFor({ timeout: 30_000 });
    // Not settle(): the run would finish while the network goes idle.
    await page.evaluate(() => document.fonts.ready);
    const theme = await page.evaluate(() => document.documentElement.dataset.theme);
    report['workspace-running'] = { theme, problems: [...(theme === 'dark' ? [] : [`data-theme is ${theme}`]), ...layoutProblems(await auditLayout(page))] };
    await page.screenshot({ path: outPath('dark/workspace-running.png') });
    await message.and(page.locator('[data-phase="done"]')).waitFor({ timeout: INTERPRETATION_MS + 15_000 });

    assert.deepEqual(failures(report), []);
    assert.deepEqual(errors, []);
  } finally {
    fs.writeFileSync(outPath('dark/report-operator.json'), `${JSON.stringify(report, null, 2)}\n`);
    await context.close();
    await resetLLMRules();
  }
});
