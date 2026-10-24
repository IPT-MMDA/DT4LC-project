// A member at / sees the landing, with the member's line and not the guest quota; a
// Workspace deep link (?sample=1, ?prompt=) still opens /app with its query
// (`npm run e2e:landing:member`: E2E_DEV_ROLE=member).
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const member = { skip: process.env.E2E_DEV_ROLE !== 'member' };
const MEMBER_LINE = 'Member. Conversations stay in this browser and results on this server until you delete them. No analysis limit.';

let h;
before(async () => {
  if (member.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function open(url, { reducedMotion } = {}) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  if (reducedMotion) await page.emulateMedia({ reducedMotion });
  await page.goto(`${h.baseURL}${url}`);
  return { context, page, errors };
}

test("a member visiting / sees the landing, with the member's line and no guest quota", member, async () => {
  // Reduced motion: the demo beside the line is its still frame in the screenshot.
  const { context, page, errors } = await open('/?from=e2e', { reducedMotion: 'reduce' });
  try {
    await page.getByTestId('landing').waitFor();
    await page.getByRole('heading', { level: 1, name: 'What changed on the ground? Ask in plain language.' }).waitFor();
    const line = page.getByText(MEMBER_LINE);
    await line.waitFor({ state: 'attached' });
    const url = new URL(page.url());
    assert.deepEqual([url.pathname, url.searchParams.get('from')], ['/', 'e2e']);
    const text = await page.getByTestId('landing').innerText();
    assert.doesNotMatch(text, /No account|analyses per hour/);
    await line.scrollIntoViewIfNeeded();
    await page.screenshot({ path: outPath('landing-member-line.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('/?sample=1 opens the Workspace with the sample chip', member, async () => {
  const { context, page, errors } = await open('/?sample=1');
  try {
    const chip = page.getByTestId('attachment-chip');
    await chip.waitFor();
    assert.match(await chip.innerText(), /kakhovka_hls_l30/);
    await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('sample'));
    await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'Account: Member' }).waitFor();
    assert.equal(await page.getByTestId('landing').count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('/?prompt= opens the Workspace with the prompt in the composer', member, async () => {
  const { context, page, errors } = await open(`/?${new URLSearchParams({ prompt: 'Map water change' })}`);
  try {
    await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('prompt'));
    const input = page.getByLabel('Message', { exact: true });
    await input.waitFor();
    assert.equal(await input.inputValue(), 'Map water change');
    assert.equal(await page.getByTestId('landing').count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
