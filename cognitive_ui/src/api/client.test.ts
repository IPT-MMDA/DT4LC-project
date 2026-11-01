import { describe, expect, it } from 'vitest';
import { API_BASE_URL, artifactUrl, resolveApiUrl } from './client';

describe('artifact URL helpers', () => {
  it('reads url from a reference only', () => {
    const ref = { kind: 'png', path: '/out/a.png', url: '/v1/download?path=%2Fout%2Fa.png' };
    expect(artifactUrl(ref)).toMatch(/^\/v1\/download\?path=/);
    expect(artifactUrl('iVBORw0KGgo')).toBeUndefined();
    expect(artifactUrl(null)).toBeUndefined();
    expect(artifactUrl({ url: '' })).toBeUndefined();
  });

  it('resolves against the API base URL the axios client uses', () => {
    expect(resolveApiUrl('/v1/download?path=x')).toBe(`${API_BASE_URL}/v1/download?path=x`);
  });
});

describe('API base URL', () => {
  // The session cookie is same-origin only (docs/API.md "Sessions"): without
  // VITE_API_URL every request goes to the page's own origin.
  it.runIf(import.meta.env.VITE_API_URL === undefined)('defaults to the same origin', () => {
    expect(API_BASE_URL).toBe('');
    expect(resolveApiUrl('/v1/session')).toBe('/v1/session');
  });
});

describe('ApiError', () => {
  it('a rejected request keeps the message, the HTTP status and the body (a 422 detail[])', async () => {
    const { AxiosError } = await import('axios');
    const { default: apiClient, ApiError } = await import('./client');
    const body = { detail: [{ loc: ['body', 'access', 'operator_emails'], msg: 'Keep your own address' }] };
    const err = await apiClient
      .put('/v1/settings', {}, {
        adapter: async (config) => {
          const response = { data: body, status: 422, statusText: 'Unprocessable Entity', headers: {}, config };
          throw new AxiosError('Request failed with status code 422', 'ERR_BAD_REQUEST', config, null, response);
        },
      })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toBeInstanceOf(Error);
    expect((err as InstanceType<typeof ApiError>).message).toBe('Keep your own address');
    expect((err as InstanceType<typeof ApiError>).status).toBe(422);
    expect((err as InstanceType<typeof ApiError>).body).toEqual(body);
  });
});
