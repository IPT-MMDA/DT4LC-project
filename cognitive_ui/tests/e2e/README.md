# Browser E2E harness

Runs browser flows against the **built** UI and the **real** API. Uses `playwright-core` (no test runner of its own, no browser download) and `node --test`.

```bash
cd cognitive_ui
npm run build          # the harness serves dist/; it refuses to run without it
npm run e2e:smoke      # flows/smoke.e2e.mjs only
npm run e2e            # every flows/*.e2e.mjs
node tests/e2e/run.mjs smoke other   # pick flows by name (flows/<name>.e2e.mjs) or by path
```

The API runs with `uv run`, so the Python env must exist (`uv sync --extra dev --extra server` at the repo root).

## What the runner does

`run.mjs`:

1. Launches `chrome-headless-shell` once as a browser server.
2. With `E2E_LLM=fake`, starts the fake Ollama server.
3. Starts the API from the repo root with `uv run uvicorn server.app:app --host 127.0.0.1 --port <free port>` and waits for `GET /v1/health`. Its state goes to a fresh temp dir: `JOB_STORAGE=sqlite`, `DT4LC_JOBS_DB`, `DT4LC_UPLOADS_PATH` and `DT4LC_MODEL_CACHE` all point into it. Its log goes to `<out>/api.log`.
4. Serves `dist/` on a free port. Unknown paths without an extension fall back to `index.html`, and `/v1/*` is proxied to the API with streaming, so SSE works. A plain `npm run build` bakes `http://localhost:8000` (the `VITE_API_URL` fallback) into the bundle. The server rewrites it to its own origin, so the browser only ever talks to the harness.
5. Runs the flows with `node --test --test-concurrency=1`. They share one API, so they run one file at a time.
6. Tears everything down in reverse order, whether flows pass, fail or the runner gets SIGINT/SIGTERM/SIGHUP. That covers the flow processes, the static server, the API process group, the temp dir, the fake LLM and the browser. The exit code is the `node --test` exit code.

Some paths stay under `resources/.cache/` because `dta/config` does not read an env var for them: `CACHE_PATH`, `TEMP_PATH` (`tmp/`) and `OUTPUTS_PATH` (`outputs/`). They are created at import if missing. Job artefacts still land in `resources/.cache/outputs/<job_id>/` until the backend makes that path configurable.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `E2E_LLM` | `none` | LLM mode: `none`, `fake` or `live` (see below). |
| `E2E_LLM_RULES` | `tests/e2e/fake-llm-rules.json` | Rules file the fake Ollama starts with (`fake` mode only). |
| `E2E_OUT` | `tests/e2e/out/` | Where screenshots and `api.log` go. Git-ignored. |
| `PLAYWRIGHT_CHROMIUM_EXECUTABLE` | newest `~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell` | Browser binary. The cached revision does not have to match the `playwright-core` version. |

Flows receive these from the runner (don't set them yourself): `E2E_BASE_URL` (UI origin), `E2E_API_URL` (API origin, direct), `E2E_WS_ENDPOINT` (browser server), `E2E_OUT`, `E2E_LLM` and `E2E_FAKE_LLM_URL`.

## LLM modes

`server/app.py` calls `load_dotenv()` without `override`, so a variable the runner sets, even to `""`, beats `.env`.

- **`none`** (default, used in gates and CI): every provider key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GPT_SECRET`) is blanked, `APERTUS_ENABLED=false`, `LLM_PROVIDER_ORDER=ollama` and `LLM_ENABLE_OLLAMA=false`. The router has no provider, so the backend is "assistant offline".
- **`fake`**: same blanking, plus `LLM_ENABLE_OLLAMA=true`, `OLLAMA_BASE_URL=http://127.0.0.1:<port>` and `OLLAMA_MODEL=fake`, all pointing at `fake-ollama.mjs`. It implements `/api/version`, `/api/tags` and `/api/chat` as `dta/dti/coe/llm/ollama.py` calls them. A reply is the first rule whose `match` regex matches the **last user message or the last system message**; if none matches, it is `default`.
- **`live`**: the runner leaves the LLM env alone, so `.env` providers are used. It is for local checks only: it is slow, costs quota, and is never part of gates or CI.

In `none` and `fake` the runner also blanks `GEE_PROJECT_ID` and `GEE_SERVICE_ACCOUNT_KEY`, so Earth Engine is never initialised and region flows fail cleanly. Other `.env` values still apply, and a `dta/config/llm.yaml` would replace the env-based LLM config entirely.

Rules file format:

```json
{
  "default": "reply when nothing matches",
  "rules": [{ "match": "regex source", "flags": "i", "reply": "canned reply" }]
}
```

## Writing a flow

Put `flows/<name>.e2e.mjs` next to `smoke.e2e.mjs`:

```js
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, resetLLMRules, setLLMRules, setup, trackPageErrors } from '../harness.mjs';

let h;
before(async () => { h = await setup(); });          // { baseURL, apiURL, browser, outDir, llm }
after(async () => { await h?.browser.close(); });    // disconnects; the runner owns the browser

test('hello answers in chat', async () => {
  await setLLMRules({ default: '{}', rules: [{ match: /hello/i, reply: 'Hi! I map land cover.' }] });
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  await page.goto(h.baseURL);
  // ...
  await page.screenshot({ path: outPath('hello.png') });
  assert.deepEqual(errors, []);
  await context.close();
  await resetLLMRules();
});
```

`setLLMRules` and `resetLLMRules` throw unless `E2E_LLM=fake`. A flow that needs the fake LLM should skip itself otherwise (`test('…', { skip: process.env.E2E_LLM !== 'fake' }, …)`: the options are read before `before()` runs). The fake server exposes two control endpoints, `PUT /__e2e/rules` (replace the rules) and `POST /__e2e/reset` (restore the file's rules); the helpers wrap them.

`fake-ollama.test.mjs` is a vitest unit test (runs in `npm test`) that checks the fake server's canned replies and control endpoints.

## Linux: missing shared libraries

On a bare Linux or WSL install, `chrome-headless-shell` may fail to start because `libnss3.so`, `libnspr4.so`, `libnssutil3.so` or `libasound.so.2` are missing, and installing them needs root. Without root:

```bash
tests/e2e/setup-libs.sh
```

The script runs `apt-get download libnss3 libnspr4 libasound2t64` (falling back to `libasound2`) and `dpkg -x` into `tests/e2e/.libroot/`, which is git-ignored. When `tests/e2e/.libroot/usr/lib/x86_64-linux-gnu` exists, the runner adds it to the browser's `LD_LIBRARY_PATH`. If the browser still fails with a missing `.so`, the runner names the library and points at the script.

CI installs the shell and its system dependencies with `npx playwright-core install --with-deps chromium-headless-shell` instead. Don't run that locally without `--no-remove`: it deletes cached browser revisions that `playwright-core` 1.64.1 doesn't use.
