// Tiny static server for the built UI: files from dist/, SPA fallback to
// index.html, and /v1/* proxied (streamed, so SSE works) to the API.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.geojson': 'application/geo+json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json',
};

// A plain `npm run build` bakes VITE_API_URL's fallback into the bundle
// (src/api/client.ts). Rewrite it to this server's origin so the browser talks
// to the proxy below instead of whatever listens on :8000.
const BUILD_API_ORIGIN = 'http://localhost:8000';

function proxy(req, res, apiURL) {
  const target = new URL(req.url ?? '/', apiURL);
  const upstream = http.request(
    target,
    { method: req.method, headers: { ...req.headers, host: target.host } },
    (up) => {
      res.writeHead(up.statusCode ?? 502, up.headers);
      up.pipe(res);
    },
  );
  upstream.on('error', (err) => {
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(`e2e proxy: ${err.message}`);
  });
  req.pipe(upstream);
}

/**
 * @param {{distDir: string, apiURL: string, port?: number}} options
 * @returns {Promise<{url: string, close: () => Promise<void>}>}
 */
export async function startStaticServer({ distDir, apiURL, port = 0 }) {
  const root = path.resolve(distDir);
  const indexFile = path.join(root, 'index.html');
  if (!fs.existsSync(indexFile)) {
    throw new Error(`${indexFile} not found: run npm run build first`);
  }
  let origin = '';

  const server = http.createServer((req, res) => {
    const { pathname } = new URL(req.url ?? '/', 'http://localhost');
    if (pathname === '/v1' || pathname.startsWith('/v1/')) return proxy(req, res, apiURL);

    let file;
    try {
      file = path.join(root, decodeURIComponent(pathname));
    } catch {
      res.writeHead(400).end();
      return;
    }
    if (!file.startsWith(root + path.sep) && file !== root) {
      res.writeHead(403).end();
      return;
    }
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      // Missing assets are real 404s; everything else is a client-side route.
      if (path.extname(pathname)) {
        res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
        return;
      }
      file = indexFile;
    }
    const ext = path.extname(file).toLowerCase();
    let body = fs.readFileSync(file);
    if (ext === '.js') body = Buffer.from(body.toString('utf8').replaceAll(BUILD_API_ORIGIN, origin));
    res.writeHead(200, { 'content-type': MIME[ext] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(body);
  });

  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  return {
    url: origin,
    close: () =>
      new Promise((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      }),
  };
}
