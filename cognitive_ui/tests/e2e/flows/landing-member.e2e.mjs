// A member at / goes straight to the Workspace, keeping the query, and never sees the
// landing (`npm run e2e:landing:member`: E2E_DEV_ROLE=member).
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { setup, trackPageErrors } from '../harness.mjs';

const member = { skip: process.env.E2E_DEV_ROLE !== 'member' };

let h;
before(async () => {
  if (member.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

test('a member visiting / lands on /app with the query kept, no flash of the landing', member, async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  try {
    await context.addInitScript(() => {
      window.__landingSeen = false;
      new MutationObserver(() => {
        if (document.querySelector('[data-testid="landing"]')) window.__landingSeen = true;
      }).observe(document, { childList: true, subtree: true });
    });
    const page = await context.newPage();
    const errors = trackPageErrors(page);
    await page.goto(`${h.baseURL}/?from=e2e`);
    await page.waitForURL((url) => url.pathname === '/app' && url.searchParams.get('from') === 'e2e');
    await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'Account: Member' }).waitFor();
    assert.equal(await page.evaluate(() => window.__landingSeen), false);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
