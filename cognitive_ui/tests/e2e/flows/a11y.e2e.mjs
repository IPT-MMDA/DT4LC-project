// axe (WCAG 2.0/2.1 A and AA) on the seven screens, in the light and the dark theme, against
// the real API. `npm run e2e:a11y` runs two groups: as a guest (E2E_LLM=none) the Landing,
// the Workspace with a done analysis, Capabilities, the Map with layers, the Library with
// rows and the Result detail; as an operator (E2E_SETTINGS=1, E2E_LLM=fake) Settings.
// No violation of impact serious or critical is allowed. Each report goes to
// out/a11y/<screen>-<theme>.json (every violation, whatever its impact).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { after, before, test } from 'node:test';
import { AxeBuilder } from '@axe-core/playwright';
import { outPath, setup, trackPageErrors } from '../harness.mjs';
import * as screens from '../screens.mjs';

const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];
const THEMES = ['light', 'dark'];
const guest = { skip: (process.env.E2E_DEV_ROLE || 'guest') !== 'guest' || process.env.E2E_LLM !== 'none' };
const operator = { skip: process.env.E2E_DEV_ROLE !== 'operator' || process.env.E2E_SETTINGS !== '1' || process.env.E2E_LLM !== 'fake' };

// Third-party widgets axe may not judge (e.g. the MapLibre canvas or attribution), by exact
// selector with the reason. None is needed today: the map and its controls pass.
const EXCLUDE = [];

let h;
before(async () => {
  if (guest.skip && operator.skip) return;
  h = await setup();
  fs.mkdirSync(outPath('a11y'), { recursive: true });
});
after(async () => {
  await h?.browser.close();
});

/** axe on the page as it is; the report is written and the serious and critical violations returned. */
async function check(page, screen, theme) {
  let builder = new AxeBuilder({ page }).withTags(TAGS);
  for (const selector of EXCLUDE) builder = builder.exclude(selector);
  const result = await builder.analyze();
  const violations = result.violations.map(({ id, impact, help, helpUrl, nodes }) => ({
    id,
    impact,
    help,
    helpUrl,
    nodes: nodes.map((n) => ({ target: n.target, html: n.html, summary: n.failureSummary })),
  }));
  const report = { screen, theme, url: page.url(), tags: TAGS, exclude: EXCLUDE, violations, passes: result.passes.length, incomplete: result.incomplete.length };
  fs.writeFileSync(outPath(`a11y/${screen}-${theme}.json`), `${JSON.stringify(report, null, 2)}\n`);
  return violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map((v) => `${screen}-${theme}: ${v.impact} ${v.id} on ${v.nodes.length} node(s), first ${v.nodes[0].target.join(' ')}`);
}

/** Both themes on the screen that is open, light first; leaves it light. */
async function bothThemes(page, screen, failures) {
  for (const theme of THEMES) {
    // Collected, not thrown: one run reports every screen.
    await screens.setTheme(page, theme).catch(() => failures.push(`${screen}-${theme}: the theme did not apply`));
    failures.push(...(await check(page, screen, theme)));
  }
  await screens.setTheme(page, 'light').catch(() => failures.push(`${screen}: the light theme did not come back`));
}

test('axe: Landing, Workspace, Capabilities, Map, Library and Result detail in both themes', guest, async () => {
  const context = await screens.newContext(h.browser);
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);
  const errors = trackPageErrors(page);
  const failures = [];
  try {
    await screens.landing(page, h.baseURL);
    await bothThemes(page, 'landing', failures);

    const message = await screens.workspaceDone(page, h.baseURL);
    await bothThemes(page, 'workspace', failures);

    await screens.mapWithLayers(page, message);
    await bothThemes(page, 'map', failures);

    await screens.capabilities(page);
    await bothThemes(page, 'capabilities', failures);

    await screens.library(page);
    await bothThemes(page, 'library', failures);

    await screens.resultDetail(page);
    await bothThemes(page, 'result-detail', failures);

    assert.deepEqual(failures, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('axe: Settings as an operator in both themes', operator, async () => {
  const context = await screens.newContext(h.browser, { region: false });
  const page = await context.newPage();
  page.setDefaultTimeout(60_000);
  const errors = trackPageErrors(page);
  const failures = [];
  try {
    await screens.settings(page, h.baseURL);
    await bothThemes(page, 'settings', failures);
    assert.deepEqual(failures, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
