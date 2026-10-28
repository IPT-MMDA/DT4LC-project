// Capabilities on the offline laptop at 1440x900: the top of the page with the offline banner.
// A member, no assistant, no Earth Engine (`npm run e2e:capabilities:offline`:
// E2E_DEV_ROLE=member, E2E_LLM=none). Also the prompt hand-off with the sample attached:
// the prompt lands in the composer, the files stay, nothing is sent.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const offline = { skip: process.env.E2E_DEV_ROLE !== 'member' || process.env.E2E_LLM !== 'none' || process.env.E2E_SERVER !== 'local' };
const BANNER =
  'The assistant is offline on this server, so tiers 1 and 2 are unavailable. Built-in analyses run as usual; answers are limited to the catalog.';

let h;
before(async () => {
  if (offline.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function open(url) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  // Nothing on these pages may start an analysis or a chat turn.
  const sent = [];
  page.on('request', (req) => {
    const { pathname } = new URL(req.url());
    if (req.method() === 'POST' && /^\/v1\/(jobs|chat|plan|execute)\b/.test(pathname)) sent.push(pathname);
  });
  await page.goto(`${h.baseURL}${url}`);
  return { context, page, errors, sent };
}

async function screenshot(page, name) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.screenshot({ path: outPath(name) });
}

test('offline laptop: banner, tiers 1-2 greyed, member summary, models download on first use', offline, async () => {
  const { context, page, errors } = await open('/app/capabilities');
  try {
    await page.getByRole('heading', { name: 'What DT4LC can do · on this server', level: 1 }).waitFor();
    assert.equal(
      await page.getByTestId('capabilities-summary').innerText(),
      'Live for this installation: member account · assistant offline · models download on first use · no Earth Engine.',
    );
    assert.equal(await page.getByTestId('capabilities-banner').innerText(), BANNER);
    for (const [id, dimmed] of [['ask', 'true'], ['composed', 'true'], ['builtin', 'false'], ['models', 'false'], ['data', 'false']]) {
      assert.equal(await page.getByTestId(`tier-${id}`).getAttribute('data-dimmed'), dimmed, id);
    }
    assert.equal(await page.getByTestId('tier-ask').getByTestId('capability-status').innerText(), 'Assistant offline — limited answers');
    assert.equal(await page.getByTestId('tier-ask').getByTestId('example-prompt').count(), 3);
    assert.equal(await page.getByTestId('tier-composed').getByTestId('example-prompt').count(), 4);
    assert.deepEqual(await page.getByTestId('tier-models').getByTestId('capability-status').allInnerTexts(), [
      'Downloads on first use · 85 MB · ~14 s · AGPL-3.0',
      'Downloads on first use · 450 MB · ~75 s · Apache-2.0',
    ]);
    assert.equal(await page.locator('[data-testid="data-item"][data-id="earth-engine"]').getByTestId('capability-status').innerText(), 'Not configured');
    await screenshot(page, 'capabilities-offline-top.png');

    // Not a design screen: the member's Models / Data, for the record.
    await page.getByRole('navigation', { name: 'Tiers' }).getByRole('button', { name: 'Models' }).click();
    // Smooth scroll: Models at the top, or the page's end.
    await page.waitForFunction(() => {
      const main = document.querySelector('[data-testid="capabilities-page"]');
      const top = document.querySelector('[data-testid="tier-models"]').getBoundingClientRect().top;
      return top < 60 || Math.ceil(main.scrollTop + main.clientHeight) >= main.scrollHeight;
    });
    await page.waitForTimeout(300);
    await screenshot(page, 'capabilities-offline-models-data.png');
    // The Data note follows the role: a member has no guest limits.
    await page.getByTestId('tier-data').getByText('Member on a local server: no analysis limit; results are kept until you delete them.').waitFor();
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('offline laptop: a built-in prompt opens the Workspace with it in the composer, the sample kept, nothing sent', offline, async () => {
  const { context, page, errors, sent } = await open('/app?sample=1');
  try {
    const chip = page.getByTestId('attachment-chip');
    await chip.waitFor();
    await page.waitForURL((url) => !url.searchParams.has('sample'));

    // In-app navigation: the composer's files live in memory, as they do for a user.
    await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name: 'Can do', exact: true }).click();
    await page.getByTestId('tier-builtin').waitFor();
    await page.locator('[data-testid="builtin-item"][data-id="algorithms/change-detection"]').getByTestId('example-prompt').click();

    await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('prompt'));
    const input = page.getByLabel('Message', { exact: true });
    await input.waitFor();
    assert.equal(await input.inputValue(), 'Compare water before and after the dam breach');
    assert.equal(await chip.count(), 1, 'the sample is still attached');
    assert.match(await chip.innerText(), /kakhovka_hls_l30/);
    await page.waitForTimeout(500);
    assert.deepEqual(sent, []);
    const { jobs } = await (await page.request.get(`${h.baseURL}/v1/jobs`)).json();
    assert.deepEqual(jobs, [], 'no job was created');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
