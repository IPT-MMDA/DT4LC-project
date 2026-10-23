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
