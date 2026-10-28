// @vitest-environment node
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_USERS, startFakeGitHub } from './fake-github.mjs';

const CALLBACK = 'http://127.0.0.1:1/v1/auth/github/callback';

describe('fake GitHub OAuth', () => {
  let gh;
  beforeAll(async () => {
    gh = await startFakeGitHub();
  });
  afterAll(async () => {
    await gh.close();
  });

  const authorize = async (clientId = FAKE_CLIENT_ID) => {
    const query = new URLSearchParams({ client_id: clientId, redirect_uri: CALLBACK, scope: 'read:user user:email', state: 's1' });
    return fetch(`${gh.url}/login/oauth/authorize?${query}`, { redirect: 'manual' });
  };
  const exchange = async (code, secret = FAKE_CLIENT_SECRET) =>
    (
      await fetch(`${gh.url}/login/oauth/access_token`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ client_id: FAKE_CLIENT_ID, client_secret: secret, code, redirect_uri: CALLBACK }),
      })
    ).json();
  const pick = (user) =>
    fetch(`${gh.url}/__e2e/user`, { method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ user }) });

  it('authorize 302s back with code and state; the code buys a token for /user and /user/emails', async () => {
    expect((await pick('operator')).status).toBe(200);
    const res = await authorize();
    expect(res.status).toBe(302);
    const back = new URL(res.headers.get('location'));
    expect(back.origin + back.pathname).toBe(CALLBACK);
    expect(back.searchParams.get('state')).toBe('s1');

    const token = await exchange(back.searchParams.get('code'));
    expect(token.access_token).toBe('tok-operator');
    const headers = { authorization: `Bearer ${token.access_token}` };
    expect(await (await fetch(`${gh.url}/user`, { headers })).json()).toEqual({
      id: FAKE_USERS.operator.id,
      login: 'e2e-operator',
      name: 'Oscar Operator',
      avatar_url: null,
    });
    const emails = await (await fetch(`${gh.url}/user/emails`, { headers })).json();
    expect(emails.find((e) => e.primary)).toEqual({ email: 'oscar@operator.e2e', primary: true, verified: true });
  });

  it('scripted users: unverified primary email, and a denied consent', async () => {
    await pick('unverified');
    const code = new URL((await authorize()).headers.get('location')).searchParams.get('code');
    const { access_token } = await exchange(code);
    const emails = await (await fetch(`${gh.url}/user/emails`, { headers: { authorization: `Bearer ${access_token}` } })).json();
    expect(emails.find((e) => e.primary).verified).toBe(false);

    await pick('denied');
    const back = new URL((await authorize()).headers.get('location'));
    expect(back.searchParams.get('error')).toBe('access_denied');
    expect(back.searchParams.has('code')).toBe(false);
    await pick('member');
  });

  it('refuses a wrong client, secret, code or token', async () => {
    expect((await authorize('someone-else')).status).toBe(400);
    expect(await exchange('code-member', 'wrong')).toEqual({ error: 'incorrect_client_credentials' });
    expect(await exchange('nonsense')).toEqual({ error: 'bad_verification_code' });
    expect((await fetch(`${gh.url}/user`, { headers: { authorization: 'Bearer nope' } })).status).toBe(401);
    expect((await pick('nobody')).status).toBe(400);
    const seen = await (await fetch(`${gh.url}/__e2e/requests`)).json();
    expect(seen.some((r) => r.path === '/login/oauth/access_token')).toBe(true);
  });
});
