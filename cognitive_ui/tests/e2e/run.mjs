// E2E runner: starts the API (isolated state), an optional fake Ollama, a static
// server for dist/ and a headless Chromium, runs flows/*.e2e.mjs under
// `node --test`, and tears everything down even when a flow fails.
//
//   node tests/e2e/run.mjs            # every flow
//   node tests/e2e/run.mjs smoke      # flows/smoke.e2e.mjs (names or paths)
//
// See tests/e2e/README.md for the env vars.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchBrowserServer } from './browser.mjs';
import { FAKE_MODEL, startFakeOllama } from './fake-ollama.mjs';
import { startStaticServer } from './static-server.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const uiDir = path.resolve(here, '..', '..');
const repoRoot = path.resolve(uiDir, '..');
const flowsDir = path.join(here, 'flows');
const outDir = path.resolve(process.env.E2E_OUT || path.join(here, 'out'));
const llmMode = process.env.E2E_LLM || 'none';
const rulesFile = path.resolve(process.env.E2E_LLM_RULES || path.join(here, 'fake-llm-rules.json'));
const API_START_TIMEOUT_MS = 180_000;

const cleanups = [];
let tearingDown;
// Runs the cleanups once, newest first; a signal and the normal exit path share it.
function teardown() {
  tearingDown ??= (async () => {
    while (cleanups.length) {
      try {
        await cleanups.pop()();
      } catch (err) {
        console.error(`e2e teardown: ${err?.message ?? err}`);
      }
    }
  })();
  return tearingDown;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function resolveFlows(args) {
  if (args.length === 0) {
    return fs
      .readdirSync(flowsDir)
      .filter((f) => f.endsWith('.e2e.mjs'))
      .sort()
      .map((f) => path.join(flowsDir, f));
  }
  return args.map((arg) => {
    const file = fs.existsSync(arg) ? path.resolve(arg) : path.join(flowsDir, `${arg}.e2e.mjs`);
    if (!fs.existsSync(file)) throw new Error(`No flow ${arg} (looked for ${file})`);
    return file;
  });
}

function apiEnv(stateDir, fakeURL) {
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    JOB_STORAGE: 'sqlite',
    DT4LC_JOBS_DB: path.join(stateDir, 'jobs.db'),
    DT4LC_UPLOADS_PATH: path.join(stateDir, 'uploads'),
    DT4LC_MODEL_CACHE: path.join(stateDir, 'models'),
  };
  if (llmMode === 'live') return env;
  // server/app.py calls load_dotenv() without override, so a variable that is
  // already set (even to "") wins over .env. Blank every provider key.
  return {
    ...env,
    GEMINI_API_KEY: '',
    GROQ_API_KEY: '',
    ANTHROPIC_API_KEY: '',
    MISTRAL_API_KEY: '',
    OPENAI_API_KEY: '',
    GPT_SECRET: '',
    APERTUS_ENABLED: 'false',
    LLM_STRATEGY: 'fallback',
    LLM_PROVIDER_ORDER: 'ollama',
    LLM_ENABLE_OLLAMA: llmMode === 'fake' ? 'true' : 'false',
    OLLAMA_BASE_URL: fakeURL ?? '',
    OLLAMA_MODEL: FAKE_MODEL,
    // No Earth Engine either: region flows must fail cleanly without credentials.
    GEE_PROJECT_ID: '',
    GEE_SERVICE_ACCOUNT_KEY: '',
  };
}

function tail(file, lines = 40) {
  try {
    return fs.readFileSync(file, 'utf8').trimEnd().split('\n').slice(-lines).join('\n');
  } catch {
    return '';
  }
}

async function startApi(env, logFile) {
  const port = await freePort();
  const url = `http://127.0.0.1:${port}`;
  const log = fs.openSync(logFile, 'w');
  const child = spawn(
    'uv',
    ['run', 'uvicorn', 'server.app:app', '--host', '127.0.0.1', '--port', String(port)],
    // Own process group, so teardown can kill uv and the Python process it spawned.
    { cwd: repoRoot, env, detached: true, stdio: ['ignore', log, log] },
  );
  fs.closeSync(log);
  const exited = new Promise((resolve) => child.once('exit', resolve));
  const killGroup = (signal) => {
    try {
      process.kill(-child.pid, signal);
    } catch {
      // already gone
    }
  };
  // Last resort if the runner itself dies without running teardown.
  const onExit = () => killGroup('SIGKILL');
  process.once('exit', onExit);
  cleanups.push(async () => {
    process.removeListener('exit', onExit);
    if (child.exitCode === null && child.signalCode === null) {
      killGroup('SIGTERM');
      const timer = setTimeout(() => killGroup('SIGKILL'), 10_000);
      await exited;
      clearTimeout(timer);
    }
    killGroup('SIGKILL');
  });

  const deadline = Date.now() + API_START_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(`API exited before it was healthy. Tail of ${logFile}:\n${tail(logFile)}`);
    }
    try {
      const res = await fetch(`${url}/v1/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return url;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`API not healthy after ${API_START_TIMEOUT_MS / 1000}s. Tail of ${logFile}:\n${tail(logFile)}`);
}

async function runFlows(flows, env) {
  const child = spawn(process.execPath, ['--test', '--test-concurrency=1', ...flows], {
    cwd: uiDir,
    env: { ...process.env, ...env },
    stdio: 'inherit',
    // Own process group, so teardown also reaches the per-file test processes.
    detached: true,
  });
  const exited = new Promise((resolve) => child.once('exit', (code, signal) => resolve(code ?? (signal ? 1 : 0))));
  cleanups.push(async () => {
    if (child.exitCode === null && child.signalCode === null) {
      try {
        process.kill(-child.pid, 'SIGTERM');
      } catch {
        // already gone
      }
      await exited;
    }
  });
  return exited;
}

async function main() {
  if (!['none', 'fake', 'live'].includes(llmMode)) {
    throw new Error(`E2E_LLM must be none, fake or live (got ${llmMode})`);
  }
  const flows = resolveFlows(process.argv.slice(2));
  fs.mkdirSync(outDir, { recursive: true });

  // The browser first: a missing shared library fails in a second, not after the API boots.
  const browserServer = await launchBrowserServer();
  cleanups.push(() => browserServer.close());

  const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dt4lc-e2e-'));
  cleanups.push(async () => fs.rmSync(stateDir, { recursive: true, force: true }));

  let fakeURL;
  if (llmMode === 'fake') {
    const fake = await startFakeOllama({ rules: JSON.parse(fs.readFileSync(rulesFile, 'utf8')) });
    cleanups.push(fake.close);
    fakeURL = fake.url;
  }

  const apiURL = await startApi(apiEnv(stateDir, fakeURL), path.join(outDir, 'api.log'));
  const ui = await startStaticServer({ distDir: path.join(uiDir, 'dist'), apiURL });
  cleanups.push(ui.close);

  console.log(`e2e: UI ${ui.url}, API ${apiURL}, LLM ${llmMode}${fakeURL ? ` (fake Ollama ${fakeURL})` : ''}, out ${outDir}`);
  return runFlows(flows, {
    E2E_BASE_URL: ui.url,
    E2E_API_URL: apiURL,
    E2E_WS_ENDPOINT: browserServer.wsEndpoint(),
    E2E_OUT: outDir,
    E2E_LLM: llmMode,
    E2E_FAKE_LLM_URL: fakeURL ?? '',
  });
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.once(signal, () => {
    teardown().finally(() => process.exit(130));
  });
}

let code = 1;
try {
  code = await main();
} catch (err) {
  console.error(`e2e: ${err?.message ?? err}`);
} finally {
  await teardown();
}
process.exit(code);
