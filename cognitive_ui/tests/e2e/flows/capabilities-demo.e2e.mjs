// Capabilities on the public demo at 1440x900: design/screens/capabilities-demo-top.png,
// capabilities-demo-models-data.png and capabilities-prithvi-gated.png (a guest on the demo).
// `npm run e2e:capabilities:demo`: E2E_SERVER=demo stages what the backend computes from
// env and files (DT4LC_DEPLOYMENT=demo, empty Delineate-Anything files in the model cache,
// GEE_PROJECT_ID and a dummy key file), guest role, E2E_LLM=fake, E2E_GUEST_LIMIT=5.
// The staged Earth Engine key is a dummy: the flows assert that no /v1/gee/ request is made.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const demo = { skip: process.env.E2E_SERVER !== 'demo' || process.env.E2E_LLM !== 'fake' || process.env.E2E_DEV_ROLE !== 'guest' };

let h;
before(async () => {
  if (demo.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function open(url) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const sent = [];
  const gee = [];
  page.on('request', (req) => {
    const { pathname } = new URL(req.url());
    if (pathname.startsWith('/v1/gee')) gee.push(pathname);
    if (req.method() === 'POST' && /^\/v1\/(jobs|chat|plan|execute)\b/.test(pathname)) sent.push(pathname);
  });
  await page.goto(`${h.baseURL}${url}`);
  await page.getByRole('heading', { name: 'What DT4LC can do · on this server', level: 1 }).waitFor();
  return { context, page, errors, sent, gee };
}

async function screenshot(page, name) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.screenshot({ path: outPath(name) });
}

// Wait until a smooth scroll has put `selector` near the top of the viewport (or the page end).
async function scrolledTo(page, selector) {
  await page.waitForFunction((sel) => {
    const main = document.querySelector('[data-testid="capabilities-page"]');
    const top = document.querySelector(sel).getBoundingClientRect().top;
    return top < 60 || Math.ceil(main.scrollTop + main.clientHeight) >= main.scrollHeight;
  }, selector);
  await page.waitForTimeout(300);
}

const builtin = (page, id) => page.locator(`[data-testid="builtin-item"][data-id="${id}"]`);

test('public demo, top: live summary, assistant connected, no banner, nothing greyed', demo, async () => {
  const { context, page, errors, gee } = await open('/app/capabilities');
  try {
    assert.equal(
      await page.getByTestId('capabilities-summary').innerText(),
      'Live for this installation: public demo · guest access · fake connected · Delineate-Anything installed · Earth Engine configured.',
    );
    assert.equal(await page.getByTestId('capabilities-banner').count(), 0);
    const pill = page.getByTestId('tier-ask').getByTestId('capability-status');
    assert.equal(await pill.innerText(), 'Assistant · fake · connected');
    assert.equal(await pill.getAttribute('data-hue'), 'done');
    for (const id of ['ask', 'composed', 'builtin', 'models', 'data']) {
      assert.equal(await page.getByTestId(`tier-${id}`).getAttribute('data-dimmed'), 'false', id);
    }
    await screenshot(page, 'capabilities-demo-top.png');
    assert.deepEqual(gee, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('public demo, Models and Data: Delineate installed, Prithvi gated, Earth Engine configured, Coming collapsed', demo, async () => {
  const { context, page, errors, gee } = await open('/app/capabilities');
  try {
    await page.getByRole('navigation', { name: 'Tiers' }).getByRole('button', { name: 'Models' }).click();
    await scrolledTo(page, '[data-testid="tier-models"]');
    assert.deepEqual(await page.getByTestId('tier-models').getByTestId('capability-status').allInnerTexts(), [
      'Installed on this server',
      'Not available for guests — ask an operator',
    ]);
    assert.deepEqual(await page.getByTestId('tier-data').getByTestId('capability-status').allInnerTexts(), ['Always', 'Always', 'Configured']);
    await page.getByText('Guests: up to 5 analyses per hour, results kept 24 h in this browser, no account needed.').waitFor();
    assert.equal(
      await page.locator('[data-testid="data-item"][data-id="sample"]').innerText().then((t) => t.split('\n')[1]?.trim()),
      'kakhovka_hls_l30 · HLS L30 · 2023-06-01 → 06-09',
    );
    const coming = page.getByTestId('tier-coming').getByRole('button');
    assert.equal(await coming.getAttribute('aria-expanded'), 'false');
    assert.match(await coming.innerText(), /Show$/);
    await screenshot(page, 'capabilities-demo-models-data.png');

    await coming.click();
    assert.equal(await coming.getAttribute('aria-expanded'), 'true');
    await page.getByTestId('tier-coming').getByText('ConvLSTM NDVI forecast', { exact: true }).nth(0).waitFor();
    // Four entries, the runoff ensemble last.
    assert.deepEqual(await page.locator('#tier-coming-items > div > span:first-child').allInnerTexts(), [
      'ConvLSTM NDVI forecast',
      'PINN-LINDA nowcasting',
      'VI forecasting DTI',
      'Runoff ensemble',
    ]);
    assert.deepEqual(gee, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('public demo, Prithvi gated for a guest: the analysis inherits the model status, its prompt is disabled', demo, async () => {
  const { context, page, errors, gee } = await open('/app/capabilities');
  try {
    const prithvi = builtin(page, 'models/prithvi-reconstruction');
    assert.equal(await prithvi.getByTestId('capability-status').innerText(), 'Not available for guests — ask an operator');
    assert.equal(await prithvi.getByTestId('example-prompt').isDisabled(), true);
    const fields = builtin(page, 'models/delineate-anything');
    assert.equal(await fields.getByTestId('capability-status').innerText(), 'Runs without the assistant');
    assert.equal(await fields.getByTestId('example-prompt').isDisabled(), false);

    // As in the design: the Change group at the top, Model-based and Models below.
    await page.getByTestId('builtin-group-change').evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.evaluate(() => document.querySelector('[data-testid="capabilities-page"]').scrollBy(0, -24));
    await screenshot(page, 'capabilities-prithvi-gated.png');
    assert.deepEqual(gee, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('public demo, prompt hand-off: an Ask prompt opens the Workspace with it in the composer, nothing sent', demo, async () => {
  const { context, page, errors, sent, gee } = await open('/app/capabilities');
  try {
    await page.getByTestId('tier-ask').getByRole('button', { name: '"Why is NDWI better than NDVI for water?"' }).click();
    await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('prompt'));
    const input = page.getByLabel('Message', { exact: true });
    await input.waitFor();
    assert.equal(await input.inputValue(), 'Why is NDWI better than NDVI for water?');
    assert.equal(await page.getByTestId('attachment-chip').count(), 0, 'no sample attached');
    await page.waitForTimeout(500);
    assert.deepEqual(sent, []);
    const { jobs } = await (await page.request.get(`${h.baseURL}/v1/jobs`)).json();
    assert.deepEqual(jobs, [], 'no job was created');
    // A reload does not bring the prompt back.
    await page.reload();
    await input.waitFor();
    assert.equal(await input.inputValue(), '');
    assert.deepEqual(gee, []);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
