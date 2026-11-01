// Without GITHUB_CLIENT_ID and friends (every group but E2E_SIGNIN=fake; `npm run
// e2e:signin:off`): the account menu says "Sign-in not configured on this server" and offers
// no button, the login route is a 404, and a guest gets the not-found page at /app/settings.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const off = { skip: process.env.E2E_SIGNIN === 'fake' || process.env.E2E_DEV_ROLE !== 'guest' };

let h;
before(async () => {
  if (off.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

test('sign-in not configured: the menu says so, with no button; /app/settings is not found', off, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app`);
    await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'Account: Guest' }).click();
    const menu = page.getByRole('menu', { name: 'Account' });
    assert.equal(await menu.getByTestId('sign-in-unavailable').innerText(), 'Sign-in not configured on this server');
    assert.equal(await menu.getByRole('menuitem', { name: /Sign in/ }).count(), 0);
    assert.equal(await menu.getByRole('button', { name: /Sign in/ }).count(), 0);
    assert.equal(await menu.getByRole('link', { name: /Sign in/ }).count(), 0);
    await page.screenshot({ path: outPath('signin-not-configured.png') });

    const s = await (await page.request.get(`${h.baseURL}/v1/session`)).json();
    assert.equal(s.sign_in.configured, false);
    const login = await page.request.get(`${h.baseURL}/v1/auth/github/login`, { maxRedirects: 0 });
    assert.equal(login.status(), 404);
    assert.equal((await login.json()).error.code, 'SIGN_IN_NOT_CONFIGURED');

    await page.goto(`${h.baseURL}/app/settings`);
    await page.getByTestId('not-found').waitFor();
    assert.equal(await page.getByRole('heading', { name: 'Page not found' }).count(), 1);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
