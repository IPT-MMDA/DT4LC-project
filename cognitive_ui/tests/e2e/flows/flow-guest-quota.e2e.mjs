// Flow 4 (#25): the guest quota as a user meets it, against the real API (guest,
// E2E_GUEST_LIMIT=2, E2E_LLM=none: `npm run e2e:run3:quota`). Two analyses an hour: a typed
// NDWI change and then a follow-up chip; the badge counts down to "0 of 2 analyses left" and
// the composer says when the next one is available; a third request is refused inline by the
// server (429 GUEST_QUOTA) with its reset time and creates no job; conversation still works
// ("hello" gets a reply); a reload keeps the count, which lives on the server.
// workspace-quota (`e2e:workspace:quota`) takes the design screenshot of the same state.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';
import { analysis, NDWI, send, settle } from '../screens.mjs';

const CHIP = 'Compare with NDVI change between the same dates';
const THIRD = 'Map moisture change with NDMI between the two dates';
const quota = { skip: process.env.E2E_GUEST_LIMIT !== '2' };

let h;
before(async () => {
  if (quota.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

const badge = (page) => page.getByTestId('session-badge');
const badgeSays = (page, text) => badge(page).filter({ hasText: text }).waitFor();
const jobs = async (page) => (await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).jobs;
/** HH:MM of an ISO time, in the page's time zone or in UTC. */
const clock = (page, iso, utc = false) =>
  page.evaluate(({ at, inUTC }) => {
    const d = new Date(at);
    const [hh, mm] = inUTC ? [d.getUTCHours(), d.getUTCMinutes()] : [d.getHours(), d.getMinutes()];
    return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  }, { at: iso, inUTC: utc });

test('a guest runs two analyses, the third is refused inline with the reset time, conversation still works', quota, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  try {
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await badgeSays(page, 'Guest. 2 of 2 analyses left this hour · kept 24 h · no account.');

    // One: a typed NDWI change.
    await send(page, NDWI);
    await analysis(page, 'done', 0);
    await badgeSays(page, 'Guest. 1 of 2 analyses left this hour');
    assert.equal(await page.getByTestId('low-quota').innerText(), '1 of 2 analyses left this hour');
    assert.equal(await page.getByTestId('quota-line').count(), 0);

    // Two: a follow-up chip on the result.
    const first = page.getByTestId('analysis-message').first();
    await first.getByRole('list', { name: 'Follow-ups' }).getByRole('button', { name: CHIP }).click();
    await analysis(page, 'done', 1);
    await badgeSays(page, 'Guest. 0 of 2 analyses left this hour');
    const session = await (await page.request.get(`${h.baseURL}/v1/session`)).json();
    assert.deepEqual([session.analyses.limit, session.analyses.remaining], [2, 0]);
    const line = page.getByTestId('quota-line');
    await line.waitFor();
    assert.equal(
      await line.innerText(),
      `You've used your 2 analyses for this hour — the next one is available at ${await clock(page, session.analyses.resets_at)}.`,
    );

    // Three: the server refuses it (429 GUEST_QUOTA) and the refusal is inline; no job.
    const [refused] = await Promise.all([
      page.waitForResponse((res) => new URL(res.url()).pathname === '/v1/chat'),
      send(page, THIRD),
    ]);
    assert.equal(refused.status(), 429);
    const body = await refused.json();
    assert.equal(body.error.code, 'GUEST_QUOTA');
    const alert = page.getByTestId('assistant-message').last().getByRole('alert');
    await alert.waitFor();
    assert.equal(
      await alert.innerText(),
      `Guests can run 2 analyses per hour on this server. The next one is available at ${await clock(page, session.analyses.resets_at, true)} UTC.`,
    );
    assert.equal(await page.getByTestId('user-message').last().innerText(), THIRD);
    assert.equal(await page.getByTestId('analysis-message').count(), 2);
    assert.equal((await jobs(page)).length, 2);
    assert.equal(await page.getByLabel('Message', { exact: true }).isEditable(), true);
    await line.waitFor();

    // Conversation still works at 0: "hello" gets a reply and no job.
    const replies = await page.getByTestId('assistant-message').count();
    await send(page, 'hello');
    await page.getByTestId('user-message').filter({ hasText: /^hello$/ }).waitFor();
    await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
    await page.getByTestId('assistant-message').nth(replies).waitFor();
    const hello = page.getByTestId('assistant-message').last();
    assert.equal(await hello.getByRole('alert').count(), 0);
    assert.ok((await hello.innerText()).replace(/^DT\s+/, '').trim().length > 0, 'a reply');
    assert.equal((await jobs(page)).length, 2);

    // The badge says none are left, and still does after a reload (the server counts).
    await badgeSays(page, '0 of 2 analyses left');
    await settle(page);
    await page.screenshot({ path: outPath('flow-guest-quota.png') });
    await page.reload();
    await badgeSays(page, 'Guest. 0 of 2 analyses left this hour');
    await page.getByTestId('quota-line').waitFor();
    assert.equal(await page.getByTestId('analysis-message').count(), 2);

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
