// Fake HuggingFace Hub for E2E_FAKE_HF=1: model downloads never leave this machine.
//
// huggingface_hub (as dta/dti/models/model_manager.py calls hf_hub_download with
// HF_ENDPOINT pointing here) makes two requests per file:
//   HEAD /{owner}/{repo}/resolve/{revision}/{file}
//        -> 200 with X-Repo-Commit (40 hex), ETag and Content-Length; without any of
//           the three it refuses the file
//   GET  the same path -> the body
// Every file is `size` zero bytes, streamed at `bytesPerSecond`, so a flow can catch
// the "Preparing model" row mid-download. The body is never a real model.
//
// Control endpoint: GET /__e2e/requests -> [{method, path}] seen so far.
import http from 'node:http';

const COMMIT = 'e2e0000000000000000000000000000000000000';
const CHUNK_MS = 100;

/**
 * @param {{size?: number, bytesPerSecond?: number, port?: number}} options
 * @returns {Promise<{url: string, requests: Array<{method: string, path: string}>, close: () => Promise<void>}>}
 */
export async function startFakeHF({ size = 120 * 1024 * 1024, bytesPerSecond = 8 * 1024 * 1024, port = 0 } = {}) {
  const requests = [];
  const timers = new Set();

  const server = http.createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/__e2e/requests') {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify(requests));
    }
    requests.push({ method: req.method ?? '', path: url.pathname });
    const file = /^\/[^/]+\/[^/]+\/resolve\/[^/]+\/(.+)$/.exec(url.pathname);
    if (!file || (req.method !== 'HEAD' && req.method !== 'GET')) {
      res.writeHead(404, { 'content-type': 'application/json' });
      return res.end(JSON.stringify({ error: `fake hf: no route ${req.method} ${url.pathname}` }));
    }
    res.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': String(size),
      'x-repo-commit': COMMIT,
      etag: `"e2e-${Buffer.from(file[1]).toString('hex').slice(0, 32)}"`,
    });
    if (req.method === 'HEAD') return res.end();

    const perChunk = Math.max(1, Math.round((bytesPerSecond * CHUNK_MS) / 1000));
    let sent = 0;
    const timer = setInterval(() => {
      const n = Math.min(perChunk, size - sent);
      sent += n;
      res.write(Buffer.alloc(n));
      if (sent >= size) {
        clearInterval(timer);
        timers.delete(timer);
        res.end();
      }
    }, CHUNK_MS);
    timers.add(timer);
    res.on('close', () => {
      clearInterval(timer);
      timers.delete(timer);
    });
  });

  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  const { port: bound } = server.address();
  return {
    url: `http://127.0.0.1:${bound}`,
    requests,
    close: () =>
      new Promise((resolve) => {
        timers.forEach(clearInterval);
        timers.clear();
        server.close(() => resolve());
        server.closeAllConnections();
      }),
  };
}
