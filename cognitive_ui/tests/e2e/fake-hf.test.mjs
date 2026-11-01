// @vitest-environment node
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { startFakeHF } from './fake-hf.mjs';

const FILE = '/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt';

describe('fake HuggingFace Hub', () => {
  let hf;
  beforeAll(async () => {
    hf = await startFakeHF({ size: 64 * 1024, bytesPerSecond: 256 * 1024 });
  });
  afterAll(async () => {
    await hf.close();
  });

  it('HEAD answers the three headers hf_hub_download requires', async () => {
    const res = await fetch(`${hf.url}${FILE}`, { method: 'HEAD' });
    expect(res.status).toBe(200);
    expect(res.headers.get('x-repo-commit')).toMatch(/^[0-9a-f]{40}$/);
    expect(res.headers.get('etag')).toMatch(/^"e2e-/);
    expect(res.headers.get('content-length')).toBe(String(64 * 1024));
  });

  it('GET streams `size` bytes at the given rate and records the requests', async () => {
    const started = Date.now();
    const body = await (await fetch(`${hf.url}${FILE}`)).arrayBuffer();
    expect(body.byteLength).toBe(64 * 1024);
    // 64 KB at 256 KB/s: about a quarter of a second, not at once.
    expect(Date.now() - started).toBeGreaterThanOrEqual(150);
    const seen = await (await fetch(`${hf.url}/__e2e/requests`)).json();
    expect(seen).toEqual([
      { method: 'HEAD', path: FILE },
      { method: 'GET', path: FILE },
    ]);
  });

  it('anything else is a 404', async () => {
    expect((await fetch(`${hf.url}/api/models/x`)).status).toBe(404);
  });
});
