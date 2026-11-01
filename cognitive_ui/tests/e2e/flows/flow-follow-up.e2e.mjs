// Flow 3 (#25): a follow-up turn, against the real API as a guest (`npm run e2e:run3:guest`
// with E2E_LLM=none and `npm run e2e:run3:fake` with E2E_LLM=fake). After an NDWI change on
// the sample is done, the follow-up chip "Compare with NDVI change between the same dates"
// runs a second analysis, re-sending the conversation's files (POST /v1/chat keeps none),
// to done; then a typed question, "Explain the last result in two sentences", gets a
// conversation reply and no new job. With the fake LLM the reply is scripted and the flow
// checks that the LLM saw the conversation's history (both analyses); with no LLM the
// reply is the offline catalog. The conversation ends with two analyses and one reply.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { llmRequests, outPath, resetLLMRules, setLLMRules, setup, trackPageErrors } from '../harness.mjs';
import { analysis, NDWI, send, settle } from '../screens.mjs';

const CHIP = 'Compare with NDVI change between the same dates';
const QUESTION = 'Explain the last result in two sentences';
const FAKE_REPLY =
  'The last result is the NDVI change between 1 and 9 June: about 198 km² of vegetation was lost and only 2 km² gained. ' +
  'Most of the loss lies along the drained reservoir and the flooded river margins.';
const OFFLINE = /^The assistant is offline on this server, so I can only describe the catalog\./;

let h;
before(async () => {
  h = await setup();
  if (h.llm === 'fake') {
    await setLLMRules({
      default: 'This is the E2E fake LLM. No rule matched this request.',
      rules: [{ match: new RegExp(`^${QUESTION}$`), reply: FAKE_REPLY }],
    });
  }
});
after(async () => {
  if (h?.llm === 'fake') await resetLLMRules();
  await h?.browser.close();
});

const stored = (page) => page.evaluate(() => JSON.parse(localStorage.getItem('dt4lc-chat-storage')).state);

test('a follow-up chip runs a second analysis on the same files; a typed question gets a reply, no job', async () => {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  const chats = [];
  page.on('request', (req) => {
    if (new URL(req.url()).pathname === '/v1/chat') chats.push(req.postDataJSON());
  });
  try {
    await page.goto(`${h.baseURL}/app?sample=1`);
    await page.getByTestId('attachment-chip').waitFor();
    await send(page, NDWI);
    const first = await analysis(page, 'done', 0);
    await first.getByRole('region', { name: 'Water change · NDWI' }).waitFor();
    const files = chats[0].attachments.map((a) => a.path);
    assert.equal(files.length, 2, 'the sample pair');

    // The follow-up chip: a second analysis on the conversation's files.
    const chips = first.getByRole('list', { name: 'Follow-ups' }).getByRole('button');
    assert.ok((await chips.allInnerTexts()).includes(CHIP), `the chip is offered (${(await chips.allInnerTexts()).join(' | ')})`);
    await chips.filter({ hasText: CHIP }).click();
    await page.getByTestId('user-message').nth(1).filter({ hasText: CHIP }).waitFor();
    const second = await analysis(page, 'done', 1);
    assert.equal(chats[1].message, CHIP);
    assert.deepEqual(chats[1].attachments.map((a) => a.path), files, 'the follow-up re-sends the same files');
    assert.match(chats[1].history.at(-1).text, /Change Detection · NDWI/);
    // The chip's message shows no chip of its own: the files are the conversation's.
    assert.equal(await page.getByTestId('user-message').nth(1).getByTestId('attachment-chip').count(), 0);
    assert.match(await second.innerText(), /I'll compute NDVI for 1 June and 9 June from the sample, then difference them/);
    assert.match(await second.getByTestId('done-line').innerText(), /Done\s+4 steps · .* · Change Detection · NDVI/);
    const card = second.getByRole('region', { name: 'Vegetation change · NDVI' });
    await card.waitFor();
    assert.equal(await card.getByTestId('legend').innerText(), 'vegetation gained\nvegetation lost\nno change');
    assert.match((await card.getByTestId('stats').innerText()).replace(/\s+/g, ' ').trim(), /^VEGETATION BEFORE \d+ km² VEGETATION AFTER \d+ km² GAINED \+[\d.]+ km² LOST \d+ km²$/);

    // A typed follow-up question: a conversation reply that uses the history, no new job.
    await send(page, QUESTION);
    await page.getByTestId('user-message').nth(2).filter({ hasText: QUESTION }).waitFor();
    await page.locator('ol[aria-label="Messages"][aria-busy="false"]').waitFor();
    const reply = page.getByTestId('assistant-message').last();
    await reply.waitFor();
    assert.equal(await reply.getByRole('alert').count(), 0, 'not an error');
    assert.equal(chats[2].message, QUESTION);
    assert.equal(chats[2].history.length, 4, 'two questions and two analyses');
    // The reply's text, after the assistant's avatar mark.
    const said = (await reply.innerText()).replace(/^DT\s+/, '');
    if (h.llm === 'fake') {
      assert.equal(said, FAKE_REPLY);
      const call = (await llmRequests()).find((r) => r.reply === FAKE_REPLY);
      assert.ok(call, 'the reply came from the LLM');
      assert.equal(call.messages.at(-1).content, QUESTION);
      const transcript = call.messages[0].content.split('The conversation so far:')[1] ?? '';
      assert.match(transcript, /User: Compare water before and after the dam breach[\s\S]*Change Detection · NDWI/);
      assert.match(transcript, new RegExp(`User: ${CHIP}[\\s\\S]*Change Detection · NDVI`));
    } else {
      assert.match(said, OFFLINE);
    }

    // Two jobs; the conversation has two analyses and one reply.
    const jobs = (await (await page.request.get(`${h.baseURL}/v1/jobs`)).json()).jobs;
    assert.deepEqual(jobs.map((j) => [j.prompt, j.status]).sort(), [[NDWI, 'completed'], [CHIP, 'completed']]);
    const state = await stored(page);
    const conversation = state.conversations.find((c) => c.id === state.activeConversationId);
    assert.deepEqual(conversation.messages.map((m) => m.kind), ['user', 'analysis', 'user', 'analysis', 'user', 'assistant']);
    assert.equal(await page.getByTestId('analysis-message').count(), 2);
    await settle(page);
    await page.screenshot({ path: outPath(`flow-follow-up-${h.llm}.png`) });

    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
