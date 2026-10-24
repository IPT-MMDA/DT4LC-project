// Scripted fake Ollama server for E2E_LLM=fake.
//
// Implements the three endpoints dta/dti/coe/llm/ollama.py calls:
//   GET  /api/version  -> {version}
//   GET  /api/tags     -> lists the configured model, so is_available() is true
//   POST /api/chat     -> {message: {role: "assistant", content: <reply>}, done: true}
//
// The reply is the first rule whose `match` regex matches the last user message
// or the last system message; otherwise `default`. A rule's `delay_ms` holds the
// reply back that long, e.g. to keep an Interpretation step running. Rules:
//   {default: string, rules: [{match: string, flags?: string, reply: string, delay_ms?: number}]}
//
// Flows change the rules at run time through the control endpoints:
//   PUT  /__e2e/rules  body = rules object  -> replaces the rules
//   POST /__e2e/reset                       -> restores the rules it started with
//   GET  /__e2e/requests                    -> [{messages, reply}] of every /api/chat so far
import http from 'node:http';

export const FAKE_MODEL = 'fake';

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJSON(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

function compile(rules) {
  return {
    default: rules.default ?? '',
    rules: (rules.rules ?? []).map((r) => ({
      re: new RegExp(r.match, r.flags ?? ''),
      reply: r.reply,
      delayMs: Number(r.delay_ms) > 0 ? Number(r.delay_ms) : 0,
    })),
  };
}

function lastContent(messages, role) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === role) return String(messages[i].content ?? '');
  }
  return '';
}

/** The matching rule (`{reply, delayMs}`) for an Ollama chat message list; `default` when none matches. */
export function pickRule(compiled, messages) {
  const user = lastContent(messages, 'user');
  const system = lastContent(messages, 'system');
  return compiled.rules.find((r) => r.re.test(user) || r.re.test(system)) ?? { reply: compiled.default, delayMs: 0 };
}

/** Pick the reply for an Ollama chat message list. */
export function pickReply(compiled, messages) {
  return pickRule(compiled, messages).reply;
}

/**
 * Start the fake server on 127.0.0.1.
 * @param {{rules: object, port?: number, model?: string}} options
 * @returns {Promise<{url: string, close: () => Promise<void>}>}
 */
export async function startFakeOllama({ rules, port = 0, model = FAKE_MODEL }) {
  const initial = compile(rules);
  let current = initial;
  const requests = [];

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost');
      if (req.method === 'GET' && url.pathname === '/api/version') {
        return sendJSON(res, 200, { version: '0.0.0-fake' });
      }
      if (req.method === 'GET' && url.pathname === '/api/tags') {
        return sendJSON(res, 200, { models: [{ name: `${model}:latest`, model: `${model}:latest` }] });
      }
      if (req.method === 'POST' && url.pathname === '/api/chat') {
        const body = JSON.parse(await readBody(req));
        const { reply: content, delayMs } = pickRule(current, body.messages ?? []);
        requests.push({ messages: body.messages ?? [], reply: content });
        if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
        return sendJSON(res, 200, {
          model: body.model ?? model,
          created_at: new Date().toISOString(),
          message: { role: 'assistant', content },
          done: true,
          done_reason: 'stop',
          prompt_eval_count: 0,
          eval_count: 0,
        });
      }
      if (req.method === 'PUT' && url.pathname === '/__e2e/rules') {
        current = compile(JSON.parse(await readBody(req)));
        return sendJSON(res, 200, { ok: true, rules: current.rules.length });
      }
      if (req.method === 'GET' && url.pathname === '/__e2e/requests') {
        return sendJSON(res, 200, requests);
      }
      if (req.method === 'POST' && url.pathname === '/__e2e/reset') {
        current = initial;
        return sendJSON(res, 200, { ok: true, rules: current.rules.length });
      }
      sendJSON(res, 404, { error: `fake ollama: no route ${req.method} ${url.pathname}` });
    } catch (err) {
      sendJSON(res, 400, { error: String(err) });
    }
  });

  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  const { port: bound } = server.address();
  return {
    url: `http://127.0.0.1:${bound}`,
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      }),
  };
}
