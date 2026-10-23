// @vitest-environment node
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { startFakeOllama } from './fake-ollama.mjs';

const RULES = {
  default: 'no rule matched',
  rules: [
    { match: '^classify', reply: '{"intent": "chat"}' },
    { match: 'hello', flags: 'i', reply: 'Hi from the fake' },
  ],
};

async function chat(url, messages) {
  const res = await fetch(`${url}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'fake', messages, stream: false }),
  });
  expect(res.status).toBe(200);
  const body = await res.json();
  expect(body.done).toBe(true);
  expect(body.message.role).toBe('assistant');
  return body.message.content;
}

describe('fake Ollama', () => {
  let fake;
  beforeAll(async () => {
    fake = await startFakeOllama({ rules: RULES });
  });
  afterAll(async () => {
    await fake.close();
  });

  it('reports the fake model as installed, as OllamaProvider.is_available() checks', async () => {
    expect((await fetch(`${fake.url}/api/version`)).status).toBe(200);
    const tags = await (await fetch(`${fake.url}/api/tags`)).json();
    expect(tags.models.map((m) => m.name)).toContain('fake:latest');
  });

  it('answers the canned reply of the first rule matching the last user message', async () => {
    expect(await chat(fake.url, [{ role: 'user', content: 'HELLO there' }])).toBe('Hi from the fake');
    expect(
      await chat(fake.url, [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: 'Hi' },
        { role: 'user', content: 'something else' },
      ]),
    ).toBe('no rule matched');
  });

  it('matches the system message too', async () => {
    const messages = [
      { role: 'system', content: 'classify the request' },
      { role: 'user', content: 'hello' },
    ];
    expect(await chat(fake.url, messages)).toBe('{"intent": "chat"}');
  });

  it('replaces and resets rules through the control endpoints', async () => {
    const put = await fetch(`${fake.url}/__e2e/rules`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ default: 'swapped', rules: [] }),
    });
    expect(put.status).toBe(200);
    expect(await chat(fake.url, [{ role: 'user', content: 'hello' }])).toBe('swapped');

    expect((await fetch(`${fake.url}/__e2e/reset`, { method: 'POST' })).status).toBe(200);
    expect(await chat(fake.url, [{ role: 'user', content: 'hello' }])).toBe('Hi from the fake');
  });
});
