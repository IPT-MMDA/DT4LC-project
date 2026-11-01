// Flow 5 (#25): sign-in with the mocked GitHub, as a user meets it (`npm run e2e:run3:signin`:
// E2E_SIGNIN=fake, so no DEV_ROLE and the role comes from the allowlists; E2E_GUEST_LIMIT=1,
// E2E_LLM=none). A guest uses their one analysis of the hour; "Sign in with GitHub" in the
// account menu goes through the fake GitHub and back as the allowlisted member: the Member
// badge with no quota, the guest's job adopted, and two more analyses that a guest could not
// run. Sign out gives a guest again, limited as before. `signin` (`e2e:signin:fake`) covers
// the operator, the denied states and the menus in detail.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, signInAs, trackPageErrors } from '../harness.mjs';
import { analysis, NDWI, send, settle } from '../screens.mjs';

const CHIP = 'Compare with NDVI change between the same dates';
const signin = { skip: process.env.E2E_SIGNIN !== 'fake' || process.env.E2E_GUEST_LIMIT !== '1' };

let h;
before(async () => {
  if (signin.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

const rail = (page) => page.getByRole('navigation', { name: 'Main' });
const session = async (page) => (await page.request.get(`${h.baseURL}/v1/session`)).json();
const jobs = async (page) => (await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).jobs;
const badgeSays = (page, text) => page.getByTestId('session-badge').filter({ hasText: text }).waitFor();
async function openMenu(page, name) {
  await rail(page).getByRole('button', { name }).click();
  return page.getByRole('menu', { name: 'Account' });
}

test('guest → Sign in with GitHub → member without a limit, analyses run → sign out is a guest again', signin, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    // A guest uses the hour's one analysis.
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await badgeSays(page, 'Guest. 1 of 1 analyses left this hour · kept 24 h · no account.');
    await send(page, NDWI);
    const guestJob = await (await analysis(page, 'done', 0)).getAttribute('data-job-id');
    await badgeSays(page, 'Guest. 0 of 1 analyses left this hour');
    await page.getByTestId('quota-line').waitFor();
    assert.equal((await session(page)).role, 'guest');

    // Sign in with GitHub from the account menu, through the fake GitHub, back to /app.
    await signInAs('member');
    const menu = await openMenu(page, 'Account: Guest');
    const link = menu.getByRole('menuitem', { name: 'Sign in with GitHub' });
    assert.equal(await link.getAttribute('href'), '/v1/auth/github/login');
    await link.click();
    await page.waitForURL((url) => url.pathname === '/app' && url.search === '');

    // The member: badge, menu, no quota anywhere, the guest's job adopted.
    const member = await session(page);
    assert.equal(member.role, 'member');
    assert.equal(member.user.login, 'e2e-member');
    assert.deepEqual([member.analyses.limit, member.analyses.remaining], [null, null]);
    await badgeSays(page, 'Member. Local server · no analysis limit · assistant offline.');
    assert.equal(await page.getByTestId('low-quota').count(), 0);
    assert.equal(await page.getByTestId('quota-line').count(), 0);
    const memberMenu = await openMenu(page, 'Account: Mira Member (Member)');
    assert.equal(await memberMenu.getByTestId('role-badge').innerText(), 'Member');
    await memberMenu.getByRole('menuitem', { name: 'Sign out' }).waitFor();
    await page.keyboard.press('Escape');
    assert.deepEqual((await jobs(page)).map((j) => j.id), [guestJob], 'the guest job is the member\'s now');

    // Two analyses where a guest has none left: a follow-up chip, then a typed one.
    const first = page.getByTestId('analysis-message').first();
    await first.and(page.locator('[data-phase="done"]')).waitFor();
    await first.getByRole('list', { name: 'Follow-ups' }).getByRole('button', { name: CHIP }).click();
    await analysis(page, 'done', 1);
    await send(page, NDWI);
    await analysis(page, 'done', 2);
    assert.equal(await page.getByTestId('assistant-message').getByRole('alert').count(), 0, 'nothing refused');
    assert.equal((await jobs(page)).length, 3);
    const ran = await session(page);
    assert.deepEqual([ran.role, ran.analyses.limit, ran.analyses.remaining], ['member', null, null]);
    await badgeSays(page, 'no analysis limit');
    assert.equal(await page.getByTestId('quota-line').count(), 0);
    await settle(page);
    await page.screenshot({ path: outPath('flow-signin-member.png') });

    // Sign out: a guest again, from the same address, so the hour's analysis is still used.
    const out = await openMenu(page, 'Account: Mira Member (Member)');
    await out.getByRole('menuitem', { name: 'Sign out' }).click();
    await page.waitForURL((url) => url.pathname === '/');
    const guest = await session(page);
    assert.deepEqual([guest.role, guest.user], ['guest', null]);
    assert.deepEqual([guest.analyses.limit, guest.analyses.remaining], [1, 0]);
    await page.goto(`${h.baseURL}/app`);
    await badgeSays(page, 'Guest. 0 of 1 analyses left this hour');
    const guestMenu = await openMenu(page, 'Account: Guest');
    await guestMenu.getByRole('menuitem', { name: 'Sign in with GitHub' }).waitFor();
    assert.deepEqual(await jobs(page), [], 'the member\'s jobs are not the new guest\'s');

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
