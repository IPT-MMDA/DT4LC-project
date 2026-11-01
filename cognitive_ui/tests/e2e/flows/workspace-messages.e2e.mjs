// Conversation through POST /v1/chat (SSE) against the real API: replies render as
// markdown and never create a job. Guest + E2E_LLM=none: the deterministic replies.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function openWorkspace() {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  await page.goto(`${h.baseURL}/app?sample=0`);
  await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
  return { context, page, errors };
}

async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
}

/** The newest assistant reply once its stream has finished (the message list is no longer busy). */
async function lastReply(page) {
  await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
  const reply = page.getByTestId('assistant-message').last();
  await reply.waitFor();
  return reply;
}

async function jobs(page) {
  // page.request shares the context's session cookie, so this is the same guest.
  const res = await page.request.get(`${h.baseURL}/v1/jobs`);
  assert.equal(res.status(), 200);
  return (await res.json()).jobs;
}

test('"hello" gets a markdown reply and creates no job', async () => {
  const { context, page, errors } = await openWorkspace();
  try {
    await send(page, 'hello');
    await page.getByTestId('user-message').filter({ hasText: 'hello' }).waitFor();
    const reply = await lastReply(page);

    assert.match(await reply.innerText(), /^DT\s+Hello! I'm the DT4LC assistant\./);
    assert.equal(await reply.locator('strong').first().innerText(), 'What can you do?');
    assert.equal(await reply.locator('em').first().innerText(), 'Calculate NDVI');
    assert.equal(await reply.locator('ul > li').count(), 2);
    assert.doesNotMatch(await reply.innerText(), /\*\*|^- /m, 'no raw markdown');

    const header = page.getByRole('main', { name: 'Chat' }).locator('header');
    assert.match(await header.innerText(), /hello/);
    assert.equal(await page.getByLabel('Message', { exact: true }).getAttribute('placeholder'), 'Ask a follow-up…');
    assert.deepEqual(await jobs(page), []);

    await page.screenshot({ path: outPath('workspace-messages-hello.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('"what can you do?" gets the catalog as a markdown list', async () => {
  const { context, page, errors } = await openWorkspace();
  try {
    await send(page, 'what can you do?');
    const reply = await lastReply(page);

    const items = reply.locator('li');
    assert.ok((await items.count()) >= 5, `a list of analyses, got ${await items.count()} items`);
    assert.ok((await reply.locator('strong').count()) >= 1, 'bold analysis names');
    assert.deepEqual(await jobs(page), []);

    await page.screenshot({ path: outPath('workspace-messages-catalog.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
