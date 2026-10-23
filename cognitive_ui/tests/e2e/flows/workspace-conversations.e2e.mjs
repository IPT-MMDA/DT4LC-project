// The conversation list against the real API (guest + E2E_LLM=none): the empty state
// ("No conversations yet", out/workspace-no-conversations.png), the active
// conversation restored by a reload, and rename and delete from a list item.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const NO_CONVERSATIONS = 'No conversations yet. They appear here as you work and stay for 24 h.';

let h;
before(async () => {
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

async function newPage() {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  return { context, page, errors: trackPageErrors(page) };
}

async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
  await page.getByTestId('assistant-message').last().waitFor();
  await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
}

const list = (page) => page.getByRole('complementary', { name: 'Conversations' });
// Each item's select button; Rename and Delete carry an aria-label.
const titles = (page) => list(page).getByTestId('conversation').locator('button:not([aria-label])').allInnerTexts();
const header = (page) => page.getByRole('main', { name: 'Chat' }).getByRole('heading', { level: 1 });

test('no conversations: the empty list says so and how long they stay', async () => {
  const { context, page, errors } = await newPage();
  try {
    await page.goto(`${h.baseURL}/app?sample=0`);
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
    const empty = page.getByTestId('no-conversations');
    await empty.waitFor();
    assert.equal(await empty.innerText(), NO_CONVERSATIONS);
    assert.equal(await list(page).getByTestId('conversation').count(), 0);
    await page.getByTestId('session-badge').waitFor();

    await page.evaluate(() => document.fonts.ready);
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    await page.screenshot({ path: outPath('workspace-no-conversations.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('a reload restores the active conversation; rename and delete work from the list', async () => {
  const { context, page, errors } = await newPage();
  try {
    await page.goto(`${h.baseURL}/app?sample=0`);
    await send(page, 'hello');
    await list(page).getByRole('button', { name: '+ New' }).click();
    await send(page, 'what can you do?');
    assert.equal(await header(page).innerText(), 'what can you do?');

    // The older one is active when the page reloads.
    await list(page).getByRole('button', { name: 'hello', exact: true }).click();
    assert.equal(await header(page).innerText(), 'hello');
    await page.reload();
    await header(page).waitFor();
    assert.equal(await header(page).innerText(), 'hello');
    assert.deepEqual(await titles(page), ['what can you do?', 'hello']);

    // Rename in place; it survives a reload.
    await list(page).getByRole('button', { name: 'Rename “hello”' }).click();
    const name = list(page).getByLabel('Conversation name');
    await name.fill('Greetings');
    await name.press('Enter');
    assert.deepEqual(await titles(page), ['what can you do?', 'Greetings']);
    assert.equal(await header(page).innerText(), 'Greetings');
    await page.reload();
    await header(page).waitFor();
    assert.deepEqual(await titles(page), ['what can you do?', 'Greetings']);

    // Delete asks first; Cancel keeps it.
    await list(page).getByRole('button', { name: 'Delete “Greetings”' }).click();
    const confirm = list(page).getByRole('group', { name: 'Delete this conversation?' });
    await confirm.getByRole('button', { name: 'Cancel' }).click();
    assert.deepEqual(await titles(page), ['what can you do?', 'Greetings']);

    // Deleting the active one goes back to a new conversation.
    await list(page).getByRole('button', { name: 'Delete “Greetings”' }).click();
    await confirm.getByRole('button', { name: 'Delete', exact: true }).click();
    assert.deepEqual(await titles(page), ['what can you do?']);
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();

    await list(page).getByRole('button', { name: 'Delete “what can you do?”' }).click();
    await confirm.getByRole('button', { name: 'Delete', exact: true }).click();
    assert.equal(await page.getByTestId('no-conversations').innerText(), NO_CONVERSATIONS);
    await page.reload();
    await page.getByTestId('no-conversations').waitFor();
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
