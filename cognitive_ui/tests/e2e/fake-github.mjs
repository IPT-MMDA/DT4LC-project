// Fake GitHub OAuth app and API for E2E_SIGNIN=fake: sign-in never leaves this machine.
//
// server/github_oauth.py, with GITHUB_OAUTH_BASE and GITHUB_API_BASE pointing here, makes:
//   GET  /login/oauth/authorize?client_id&redirect_uri&scope&state  (the browser)
//        -> 302 to redirect_uri?code&state, or ?error=access_denied&state for the `denied` user
//   POST /login/oauth/access_token  {client_id, client_secret, code, redirect_uri}  (JSON)
//        -> {access_token, token_type, scope}; a wrong secret or code -> {error} (200, as GitHub)
//   GET  /user          (Authorization: Bearer <token>) -> {id, login, name, avatar_url}
//   GET  /user/emails   -> [{email, primary, verified}]
//
// Who signs in is scripted by the flow: PUT /__e2e/user {user: "member"} picks one of
// FAKE_USERS for the next authorize (default `member`). GET /__e2e/requests lists
// [{method, path}] seen so far.
import http from 'node:http';

export const FAKE_CLIENT_ID = 'e2e-client-id';
export const FAKE_CLIENT_SECRET = 'e2e-client-secret';

/** The scripted accounts. The runner allowlists `member` and makes `operator` an operator. */
export const FAKE_USERS = {
  member: { id: 7001, login: 'e2e-member', name: 'Mira Member', email: 'mira@member.e2e', verified: true },
  operator: { id: 7002, login: 'e2e-operator', name: 'Oscar Operator', email: 'oscar@operator.e2e', verified: true },
  outsider: { id: 7003, login: 'e2e-outsider', name: 'Olga Outsider', email: 'olga@outside.e2e', verified: true },
  unverified: { id: 7004, login: 'e2e-unverified', name: null, email: 'ula@member.e2e', verified: false },
  // GitHub sends the browser back with ?error=access_denied (the user clicked Cancel).
  denied: null,
};

const json = (res, status, body) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

async function readBody(req) {
  let raw = '';
  for await (const chunk of req) raw += chunk;
  const type = req.headers['content-type'] ?? '';
  if (type.includes('application/json')) return raw ? JSON.parse(raw) : {};
  return Object.fromEntries(new URLSearchParams(raw));
}

/**
 * @param {{port?: number}} options
 * @returns {Promise<{url: string, requests: Array<{method: string, path: string}>, close: () => Promise<void>}>}
 */
export async function startFakeGitHub({ port = 0 } = {}) {
  const requests = [];
  let next = 'member';

  const userFor = (req) => {
    const token = /^Bearer tok-(\w+)$/.exec(req.headers.authorization ?? '')?.[1];
    return token && FAKE_USERS[token] ? FAKE_USERS[token] : null;
  };

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    try {
      if (url.pathname === '/__e2e/requests' && req.method === 'GET') return json(res, 200, requests);
      if (url.pathname === '/__e2e/user' && req.method === 'PUT') {
        const { user } = await readBody(req);
        if (!(user in FAKE_USERS)) return json(res, 400, { error: `fake github: no user ${user}` });
        next = user;
        return json(res, 200, { user });
      }
      requests.push({ method: req.method ?? '', path: url.pathname });

      if (url.pathname === '/login/oauth/authorize' && req.method === 'GET') {
        const redirect = url.searchParams.get('redirect_uri');
        const state = url.searchParams.get('state') ?? '';
        if (url.searchParams.get('client_id') !== FAKE_CLIENT_ID || !redirect) {
          return json(res, 400, { error: 'fake github: wrong client_id or no redirect_uri' });
        }
        const back = new URL(redirect);
        if (FAKE_USERS[next] === null) back.searchParams.set('error', 'access_denied');
        else back.searchParams.set('code', `code-${next}`);
        back.searchParams.set('state', state);
        res.writeHead(302, { location: back.toString() });
        return res.end();
      }

      if (url.pathname === '/login/oauth/access_token' && req.method === 'POST') {
        const body = await readBody(req);
        const user = /^code-(\w+)$/.exec(body.code ?? '')?.[1];
        if (body.client_id !== FAKE_CLIENT_ID || body.client_secret !== FAKE_CLIENT_SECRET) {
          return json(res, 200, { error: 'incorrect_client_credentials' });
        }
        if (!user || !FAKE_USERS[user]) return json(res, 200, { error: 'bad_verification_code' });
        return json(res, 200, { access_token: `tok-${user}`, token_type: 'bearer', scope: 'read:user,user:email' });
      }

      if ((url.pathname === '/user' || url.pathname === '/user/emails') && req.method === 'GET') {
        const user = userFor(req);
        if (!user) return json(res, 401, { message: 'Bad credentials' });
        if (url.pathname === '/user') {
          return json(res, 200, { id: user.id, login: user.login, name: user.name, avatar_url: null });
        }
        return json(res, 200, [
          { email: `${user.login}@users.noreply.github.com`, primary: false, verified: true },
          { email: user.email, primary: true, verified: user.verified },
        ]);
      }

      return json(res, 404, { message: `fake github: no route ${req.method} ${url.pathname}` });
    } catch (err) {
      return json(res, 500, { message: `fake github: ${err?.message ?? err}` });
    }
  });

  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  const { port: bound } = server.address();
  return {
    url: `http://127.0.0.1:${bound}`,
    requests,
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      }),
  };
}
