// GitHub sign-in against the fake GitHub (`npm run e2e:signin:fake`: E2E_SIGNIN=fake, no
// DEV_ROLE, so the role comes from the allowlists). A guest sees "Sign in with GitHub"; the
// allowlisted member signs in (Member badge, no quota line, no analysis limit, no Settings)
// and out again; the operator gets Settings, and the server keeps their own address in
// operator_emails; the outsider, an unverified email and a cancelled consent each come back
// to /app as a guest with a notice, the query cleaned from the URL.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, signInAs, trackPageErrors } from '../harness.mjs';

const signin = { skip: process.env.E2E_SIGNIN !== 'fake' };

let h;
before(async () => {
  if (signin.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

const rail = (page) => page.getByRole('navigation', { name: 'Main' });
async function openMenu(page, name) {
  await rail(page).getByRole('button', { name }).click();
  return page.getByRole('menu', { name: 'Account' });
}
const session = async (page) => (await page.request.get(`${h.baseURL}/v1/session`)).json();

/** From /app as a guest: the menu's link, through the fake GitHub, back to /app. */
async function signIn(page, user) {
  await signInAs(user);
  await page.goto(`${h.baseURL}/app`);
  const menu = await openMenu(page, 'Account: Guest');
  const link = menu.getByRole('menuitem', { name: 'Sign in with GitHub' });
  assert.equal(await link.getAttribute('href'), '/v1/auth/github/login');
  await link.click();
  await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('code'));
}

async function signOut(page, name) {
  const menu = await openMenu(page, name);
  await menu.getByRole('menuitem', { name: 'Sign out' }).click();
  await page.waitForURL((url) => url.pathname === '/');
  const fresh = await session(page);
  assert.equal(fresh.role, 'guest');
  assert.equal(fresh.user, null);
}

async function notFoundAtSettings(page) {
  await page.goto(`${h.baseURL}/app/settings`);
  await page.getByTestId('not-found').waitFor();
  assert.equal(await page.getByRole('heading', { name: 'Settings · this server' }).count(), 0);
  assert.equal((await page.request.get(`${h.baseURL}/v1/settings`)).status(), 404);
}

test('a guest sees "Sign in with GitHub"; /app/settings is not found', signin, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app`);
    const menu = await openMenu(page, 'Account: Guest');
    await menu.getByRole('menuitem', { name: 'Sign in with GitHub' }).waitFor();
    assert.equal(await menu.getByTestId('sign-in-unavailable').count(), 0);
    assert.equal(await menu.getByTestId('role-badge').innerText(), 'Guest');
    const s = await session(page);
    assert.equal(s.sign_in.configured, true);
    assert.equal(s.analyses.limit, 1000);
    await page.screenshot({ path: outPath('signin-guest-menu.png') });
    await notFoundAtSettings(page);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('the allowlisted member: Member badge, no quota line, no limit, no Settings; sign out is a guest again', signin, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await signIn(page, 'member');
    const s = await session(page);
    assert.equal(s.role, 'member');
    assert.equal(s.guest, false);
    assert.deepEqual(s.user, { id: 'github:7001', login: 'e2e-member', name: 'Mira Member', email: 'mira@member.e2e', avatar_url: null });
    // No analysis limit: a seventh analysis would not be refused.
    assert.equal(s.analyses.limit, null);
    assert.equal(s.analyses.remaining, null);
    assert.equal(s.kept_hours, null);

    const badge = page.getByTestId('session-badge');
    await badge.filter({ hasText: 'Member.' }).waitFor();
    const line = await badge.innerText();
    assert.match(line, /^Member\. Local server · no analysis limit · assistant offline\.$/);
    assert.doesNotMatch(line, /analyses left/);
    assert.equal(await page.getByTestId('low-quota').count(), 0);
    assert.equal(await page.getByTestId('quota-line').count(), 0);

    const button = rail(page).getByRole('button', { name: 'Account: Mira Member (Member)' });
    assert.equal(await button.getAttribute('title'), 'Mira Member');
    const menu = await openMenu(page, 'Account: Mira Member (Member)');
    const user = menu.getByTestId('account-user');
    for (const text of ['Mira Member', '@e2e-member', 'mira@member.e2e']) assert.ok((await user.innerText()).includes(text), text);
    assert.equal(await menu.getByTestId('role-badge').innerText(), 'Member');
    assert.equal(await menu.getByRole('menuitem', { name: 'Settings' }).count(), 0);
    assert.equal(await menu.getByRole('menuitem', { name: /Sign in/ }).count(), 0);
    await menu.getByRole('menuitem', { name: 'Sign out' }).waitFor();
    await page.screenshot({ path: outPath('signin-member-menu.png') });
    await page.keyboard.press('Escape');

    // The Library header has no guest expiry for a member.
    await page.goto(`${h.baseURL}/app/library`);
    await page.getByRole('heading', { name: 'Library' }).waitFor();
    assert.doesNotMatch(await page.getByTestId('library-page').innerText(), /Guest items expire/);

    await notFoundAtSettings(page);
    await signOut(page, 'Account: Mira Member (Member)');
    await page.goto(`${h.baseURL}/app`);
    const guestMenu = await openMenu(page, 'Account: Guest');
    await guestMenu.getByRole('menuitem', { name: 'Sign in with GitHub' }).waitFor();
    await page.getByTestId('session-badge').filter({ hasText: 'Guest.' }).waitFor();
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('the operator: Settings in the menu, /app/settings renders, and their own address stays an operator', signin, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await signIn(page, 'operator');
    assert.equal((await session(page)).role, 'operator');
    const menu = await openMenu(page, 'Account: Oscar Operator (Operator)');
    assert.equal(await menu.getByTestId('role-badge').innerText(), 'Operator');
    await menu.getByRole('menuitem', { name: 'Settings' }).click();
    await page.waitForURL((url) => url.pathname === '/app/settings');
    await page.getByTestId('settings-section-assistant').waitFor();
    assert.equal(await page.getByTestId('not-found').count(), 0);

    // Access: sign-in is configured here; removing their own operator address is refused inline.
    await page.getByTestId('settings-tab-access').click();
    const access = page.getByTestId('settings-section-access');
    assert.equal(await access.getByTestId('sign-in-status').innerText(), 'Sign-in configured');
    await access.getByRole('button', { name: 'Remove oscar@operator.e2e' }).click();
    const [put] = await Promise.all([
      page.waitForResponse((res) => res.request().method() === 'PUT' && new URL(res.url()).pathname === '/v1/settings'),
      page.getByRole('button', { name: 'Save', exact: true }).click(),
    ]);
    assert.equal(put.status(), 422);
    const error = page.locator('[data-path="access.operator_emails"]').getByTestId('field-error');
    await error.waitFor();
    assert.match(await error.innerText(), /^Keep your own address \(oscar@operator\.e2e\): without it you lose operator access$/);
    await page.getByRole('button', { name: 'Discard', exact: true }).click();
    await access.getByRole('button', { name: 'Remove oscar@operator.e2e' }).waitFor();

    await signOut(page, 'Account: Oscar Operator (Operator)');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

for (const [user, text, shot] of [
  ['outsider', 'olga@outside.e2e is not on the allowlist for this server', 'signin-denied.png'],
  ['unverified', 'Your primary GitHub email is not verified — verify it on GitHub, then sign in again', null],
  ['denied', 'Sign-in failed — try again', null],
]) {
  test(`${user}: back on /app as a guest with "${text}"`, signin, async () => {
    const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const errors = trackPageErrors(page);
    try {
      await signIn(page, user);
      const notice = page.getByTestId('signin-notice');
      await notice.waitFor();
      assert.equal(await notice.innerText(), text);
      // The query is read once and cleaned from the URL.
      await page.waitForURL((url) => url.pathname === '/app' && url.search === '');
      assert.equal((await session(page)).role, 'guest');
      await rail(page).getByRole('button', { name: 'Account: Guest' }).waitFor();
      if (shot) await page.screenshot({ path: outPath(shot) });
      await notice.getByRole('button', { name: 'Dismiss' }).click();
      assert.equal(await notice.count(), 0);
      assert.deepEqual(errors, []);
    } finally {
      await context.close();
    }
  });
}
