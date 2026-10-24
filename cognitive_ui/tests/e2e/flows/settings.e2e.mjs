// Operator Settings against the real API (`npm run e2e:settings`: E2E_DEV_ROLE=operator,
// E2E_SETTINGS=1, E2E_LLM=fake). The five sections render from GET /v1/settings; reordering
// providers and changing the guest limit sends a PUT with only those fields, and a reload
// shows them (as overrides); Reset to .env sends null; the Test button probes the fake
// Ollama; Pre-download follows a (fake Hub) download. The provider keys the runner set
// (FAKE_PROVIDER_KEYS) are reported as present and appear nowhere in the page.
// Guests and members get the not-found page: flows/signin.e2e.mjs and signin-off.e2e.mjs.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { FAKE_PROVIDER_KEYS, outPath, setup, trackPageErrors } from '../harness.mjs';

const operator = {
  skip: process.env.E2E_DEV_ROLE !== 'operator' || process.env.E2E_SETTINGS !== '1' || process.env.E2E_LLM !== 'fake',
};
const SECTIONS = [
  ['assistant', 'Assistant'],
  ['models', 'Models'],
  ['access', 'Access'],
  ['data_sources', 'Data sources'],
  ['server', 'Server'],
];
const KEYS = Object.values(FAKE_PROVIDER_KEYS);

let h;
before(async () => {
  if (operator.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

const section = (page, id) => page.getByTestId(`settings-section-${id}`);
async function open(page, id) {
  await page.getByTestId(`settings-tab-${id}`).click();
  await section(page, id).waitFor();
}
async function assertNoKeys(page) {
  const html = await page.content();
  for (const key of KEYS) assert.ok(!html.includes(key), `a provider key is in the page`);
}
async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
}
const saveButton = (page) => page.getByRole('button', { name: 'Save', exact: true });

/** Save, and the PUT body that went out. */
async function save(page) {
  const [request] = await Promise.all([
    page.waitForRequest((req) => req.method() === 'PUT' && new URL(req.url()).pathname === '/v1/settings'),
    saveButton(page).click(),
  ]);
  await page.getByTestId('settings-savebar').filter({ hasText: 'Saved.' }).waitFor();
  return request.postDataJSON();
}

test('the five sections render from GET /v1/settings, with no key value anywhere', operator, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    const res = await page.request.get(`${h.baseURL}/v1/settings`);
    assert.equal(res.status(), 200);
    const raw = await res.text();
    for (const key of KEYS) assert.ok(!raw.includes(key), 'a provider key is in GET /v1/settings');
    const doc = JSON.parse(raw);

    await page.goto(`${h.baseURL}/app/settings`);
    const assistant = section(page, 'assistant');
    await assistant.waitFor();
    assert.equal(await assistant.getByTestId('assistant-connected').innerText(), 'Connected · Ollama · fake');
    const ordered = assistant.getByRole('list', { name: 'Providers in order' }).getByTestId('provider');
    assert.deepEqual(await ordered.evaluateAll((els) => els.map((e) => e.dataset.name)), doc.assistant.provider_order);
    for (const name of ['gemini', 'groq']) {
      const card = assistant.locator(`[data-testid="provider"][data-name="${name}"]`);
      assert.equal(await card.getByTestId('key-status').innerText(), 'Key present');
    }
    assert.equal(await assistant.locator('[data-name="mistral"]').getByTestId('key-status').innerText(), 'No key');
    assert.equal(await page.locator('input[type="password"]').count(), 0);
    await settle(page);
    await page.screenshot({ path: outPath('settings.png') });

    // The probe: the fake Ollama answers.
    await assistant.getByRole('button', { name: 'Test', exact: true }).click();
    const probe = assistant.getByTestId('probe');
    await probe.waitFor({ timeout: 30_000 });
    assert.equal(await probe.getAttribute('data-ok'), 'true');
    assert.match(await probe.innerText(), /Ollama\s+fake\s+OK · \d+ ms/);

    for (const [id] of SECTIONS) {
      await open(page, id);
      await assertNoKeys(page);
      await settle(page);
      await page.screenshot({ path: outPath(`settings-${id.replace('_', '-')}.png`) });
    }

    // What each read-only section says comes from the GET.
    await open(page, 'models');
    const models = section(page, 'models');
    assert.equal(await models.getByTestId('settings-model').count(), doc.models.items.length);
    assert.equal(await models.getByRole('combobox', { name: 'Downloads on first use' }).inputValue(), doc.models.download_policy);
    await open(page, 'access');
    assert.equal(await section(page, 'access').getByTestId('sign-in-status').innerText(), 'Sign-in not configured');
    await open(page, 'data_sources');
    const data = section(page, 'data_sources');
    assert.equal(await data.getByTestId('ee-status').innerText(), 'Not configured');
    assert.ok((await data.innerText()).includes(doc.data_sources.uploads.path));
    assert.equal(await data.locator('input, select, textarea').count(), 0);
    await open(page, 'server');
    const server = section(page, 'server');
    assert.ok((await server.innerText()).includes(doc.server.version));
    assert.equal(await server.getByTestId('storage-row').count(), Object.keys(doc.server.paths).length);
    assert.equal(await server.locator('input, select, textarea').count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('reorder providers and change the guest limit: the PUT has just those, and a reload shows them', operator, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app/settings`);
    const assistant = section(page, 'assistant');
    await assistant.waitFor();
    assert.equal(await saveButton(page).isDisabled(), true);
    await assistant.getByRole('button', { name: 'Move Mistral up' }).click();
    await open(page, 'access');
    await section(page, 'access').getByRole('spinbutton', { name: 'Guest analyses per hour' }).fill('7');
    assert.equal(await page.getByTestId('settings-savebar').getAttribute('data-dirty'), 'true');

    assert.deepEqual(await save(page), {
      assistant: { provider_order: ['mistral', 'ollama'] },
      access: { guest_analyses_per_hour: 7 },
    });

    await page.reload();
    await section(page, 'assistant').waitFor();
    const ordered = section(page, 'assistant').getByRole('list', { name: 'Providers in order' }).getByTestId('provider');
    assert.deepEqual(await ordered.evaluateAll((els) => els.map((e) => e.dataset.name)), ['mistral', 'ollama']);
    await open(page, 'access');
    const limit = page.locator('[data-path="access.guest_analyses_per_hour"]');
    assert.equal(await limit.getByRole('spinbutton').inputValue(), '7');
    assert.equal(await limit.getByTestId('setting-source').innerText(), 'override');
    // The guest quota follows at once.
    assert.equal((await (await page.request.get(`${h.baseURL}/v1/settings`)).json()).access.guest_analyses_per_hour, 7);

    // Reset to .env: null goes out, and the env value is back.
    await limit.getByRole('button', { name: 'Reset to .env' }).click();
    await open(page, 'assistant');
    await page.locator('[data-path="assistant.provider_order"]').getByRole('button', { name: 'Reset to .env' }).click();
    assert.deepEqual(await save(page), { assistant: { provider_order: null }, access: { guest_analyses_per_hour: null } });
    await page.reload();
    await open(page, 'access');
    assert.equal(await limit.getByRole('spinbutton').inputValue(), '1000');
    assert.equal(await limit.getByTestId('setting-source').innerText(), 'from .env');
    await assertNoKeys(page);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('Pre-download starts a download and shows its progress', operator, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app/settings`);
    await section(page, 'assistant').waitFor();
    await open(page, 'models');
    const row = section(page, 'models').locator('[data-testid="settings-model"][data-id="prithvi-eo-v1-100m"]');
    const [post] = await Promise.all([
      page.waitForResponse((res) => res.request().method() === 'POST' && new URL(res.url()).pathname === '/v1/ml-models/prithvi-eo-v1-100m/download'),
      row.getByRole('button', { name: 'Pre-download Prithvi EO v1 (100M)' }).click(),
    ]);
    assert.equal(post.status(), 202);
    await row.getByText(/Downloading · \d+ %/).waitFor({ timeout: 20_000 });
    // Stop it: the weights are zeros from the fake Hub.
    const cancel = await page.request.post(`${h.baseURL}/v1/ml-models/prithvi-eo-v1-100m/cancel`);
    assert.equal(cancel.status(), 200);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
