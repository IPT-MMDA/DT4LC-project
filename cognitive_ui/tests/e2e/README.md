# Browser E2E harness

Runs browser flows against the **built** UI and the **real** API. Uses `playwright-core` (no test runner of its own, no browser download) and `node --test`.

```bash
cd cognitive_ui
npm run build          # the harness serves dist/; it refuses to run without it
npm run e2e:smoke      # flows/smoke.e2e.mjs only
npm run e2e            # every flows/*.e2e.mjs (guest, E2E_LLM=none; fake- and member-only flows skip)
npm run e2e:workspace  # the Workspace screens: the four env groups below, one API run each
npm run e2e:run2       # the run-2 screens: e2e:capabilities, e2e:landing and e2e:map, their groups below
node tests/e2e/run.mjs smoke other   # pick flows by name (flows/<name>.e2e.mjs) or by path
```

One API process serves a run, so flows that need another role or LLM mode run in their own group:

| Script | Env | Flows | Screens (`out/`) |
|---|---|---|---|
| `e2e:workspace:guest` | guest, `E2E_LLM=none` | `workspace-empty`, `workspace-messages`, `workspace-composer`, `workspace-analysis`, `workspace-result`, `workspace-conversations` | empty states, assistant offline, region chip, proposal, composed done, failed and retry, model gated for guest, done with the result card, no conversations (`workspace-no-conversations.png`); rename, delete and reload of the conversation list |
| `e2e:workspace:fake` | guest, `E2E_LLM=fake` | `workspace-running` | analysis running (and cancel; a reload mid-run resumes the stream) |
| `e2e:workspace:member` | `E2E_DEV_ROLE=member`, `E2E_LLM=none` | `workspace-preparing` | preparing model, assistant offline, the member badge |
| `e2e:workspace:quota` | guest, `E2E_GUEST_LIMIT=2`, `E2E_LLM=none` | `workspace-quota` | two analyses, then the quota refusal inline and the composer's quota line (`workspace-guest-quota.png`) |
| `e2e:capabilities:offline` | `E2E_DEV_ROLE=member`, `E2E_LLM=none` | `capabilities-offline` | the offline laptop (`capabilities-offline-top.png`; `capabilities-offline-models-data.png`, not a design screen); a built-in prompt opens the Workspace with it in the composer and the sample kept, nothing sent |
| `e2e:capabilities:demo` | guest, `E2E_SERVER=demo`, `E2E_LLM=fake`, `E2E_GUEST_LIMIT=5` | `capabilities-demo` | the public demo (`capabilities-demo-top.png`, `capabilities-demo-models-data.png`, `capabilities-prithvi-gated.png`); an Ask prompt opens the Workspace with it in the composer, nothing sent |
| `e2e:landing:guest` | guest, `E2E_GUEST_LIMIT=5`, `E2E_LLM=none` | `landing` | the guest landing (`landing-hero.png`, `landing-below-fold.png` after the demo has played; `landing-footer.png` and `landing-below-fold-reduced-motion.png`, not design screens); both hero buttons; a guest with a stored conversation goes to `/app`; reduced motion: no animations, a still demo; the step buttons scrub the demo |
| `e2e:landing:member` | `E2E_DEV_ROLE=member`, `E2E_LLM=none` | `landing-member` | none: a member at `/` lands on `/app` with the query kept, the landing never renders |
| `e2e:map` | guest, `E2E_LLM=none` | `map-mode` | map mode (`map-mode.png` at 1440×900, `map-mode-1280.png` at 1280×800; `map-mode-compare.png`, not a design screen): NDWI change On map from the chat, the sample in true colour with its footprint under it, the legend with the class areas, a rectangle drawn with the mouse and renamed, Fit to result, Compare (pixels on both sides of the divider, moved by keyboard and drag), Fetch to map disabled without Earth Engine, Send to chat to the region chip; at 1280×800 no panel or control overlaps another, with the region presets row under the region card and regions in amber; Kakhovka dam (the sample footprint here), Brienz (zooms to the lake) and Upload GeoJSON (a point is refused, a polygon becomes the region) |

`workspace-running` skips itself unless `E2E_LLM=fake`, `workspace-preparing` unless `E2E_DEV_ROLE=member`, and `workspace-quota` unless `E2E_GUEST_LIMIT=2`. `capabilities-offline` runs only as a member with `E2E_LLM=none` on a local server, and `capabilities-demo` only as a guest with `E2E_SERVER=demo` and `E2E_LLM=fake`. `landing` runs only as a guest, `landing-member` only with `E2E_DEV_ROLE=member`.

The API runs with `uv run`, so the Python env must exist (`uv sync --extra dev --extra server` at the repo root).

## What the runner does

`run.mjs`:

1. Launches `chrome-headless-shell` once as a browser server.
2. Starts the fake HuggingFace Hub (`fake-hf.mjs`, always) and, with `E2E_LLM=fake`, the fake Ollama server.
3. Starts the API from the repo root with `uv run uvicorn server.app:app --host 127.0.0.1 --port <free port>` and waits for `GET /v1/health`. Its state goes to a fresh temp dir: `JOB_STORAGE=sqlite`, and `DT4LC_CACHE_PATH`, `DT4LC_TEMP_PATH`, `DT4LC_OUTPUTS_PATH`, `DT4LC_JOBS_DB`, `DT4LC_UPLOADS_PATH` and `DT4LC_MODEL_CACHE` all point into it, so job runs write nothing to `resources/.cache/`. `HF_ENDPOINT` points at the fake Hub (`HF_HUB_OFFLINE=0`, `HF_HOME` in the temp dir), so no run downloads model weights from huggingface.co. Every browser is a guest (`DEV_ROLE=guest`) with a guest limit of 1000 analyses per hour, so a run of flows never hits the quota; `E2E_DEV_ROLE` and `E2E_GUEST_LIMIT` change them. Its log goes to `<out>/api.log`.
4. Serves `dist/` on a free port. Unknown paths without an extension fall back to `index.html`, and `/v1/*` is proxied to the API with streaming, so SSE works. The UI calls the API on its own origin by default (`VITE_API_URL` unset), which is this proxy. A build made with `VITE_API_URL=http://localhost:8000` is rewritten to the server's own origin, so the browser only ever talks to the harness.
5. Runs the flows with `node --test --test-concurrency=1`. They share one API, so they run one file at a time.
6. Tears everything down in reverse order, whether flows pass, fail or the runner gets SIGINT/SIGTERM/SIGHUP. That covers the flow processes, the static server, the API process group, the temp dir, the fake LLM and the browser. The exit code is the `node --test` exit code.

One path stays outside the temp dir: Delineate-Anything works in `<system temp>/dt4lc_delineate`, which does not derive from `DT4LC_CACHE_PATH`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `E2E_LLM` | `none` | LLM mode: `none`, `fake` or `live` (see below). |
| `E2E_LLM_RULES` | `tests/e2e/fake-llm-rules.json` | Rules file the fake Ollama starts with (`fake` mode only). |
| `E2E_OUT` | `tests/e2e/out/` | Where screenshots and `api.log` go. Git-ignored. |
| `E2E_DEV_ROLE` | `guest` | Role of every request (`DEV_ROLE` for the API): `guest`, `member` or `operator`. One API serves the whole run, so a flow that needs another role needs its own run. |
| `E2E_GUEST_LIMIT` | `1000` | Guest analyses per hour (`GUEST_ANALYSES_PER_HOUR` for the API); set it low to test the quota refusal. |
| `E2E_SERVER` | `local` | Server state. `local` sets `DT4LC_DEPLOYMENT=local`. `demo` stages the public demo as the backend computes it from env and files: `DT4LC_DEPLOYMENT=demo`; Delineate-Anything "installed" (empty `DelineateAnything-S.pt` and `delineate.py` in the temp `DT4LC_MODEL_CACHE`, the files `ModelManager.get_model_status` checks); Earth Engine "configured" (`GEE_PROJECT_ID=e2e-demo`, `GEE_SERVICE_ACCOUNT_KEY` pointing at a dummy JSON in the temp dir, which is all `gee_common.is_configured()` checks). Nothing real is behind them: a demo flow must never run Delineate or call `/v1/gee/*`. |
| `PLAYWRIGHT_CHROMIUM_EXECUTABLE` | newest `~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell` | Browser binary. The cached revision does not have to match the `playwright-core` version. |

Flows receive these from the runner (don't set them yourself): `E2E_BASE_URL` (UI origin), `E2E_API_URL` (API origin, direct), `E2E_WS_ENDPOINT` (browser server), `E2E_OUT`, `E2E_LLM`, `E2E_FAKE_LLM_URL`, `E2E_FAKE_HF_URL`, `E2E_DEV_ROLE` and `E2E_SERVER`.

## Fake HuggingFace Hub

`huggingface_hub` honours `HF_ENDPOINT`. For each file `hf_hub_download` sends `HEAD /{repo}/resolve/{revision}/{file}` and needs `X-Repo-Commit`, an `ETag` and `Content-Length` back, then `GET`s the same path. `fake-hf.mjs` answers every such path with 120 MB of zeros streamed at 8 MB/s, so a member's Preparing model step shows real `step_progress` (bytes, total, ETA) for about 15 s per file. `GET /__e2e/requests` lists the requests it saw. A flow cancels the job once it has its picture, before the model step would load the fake weights.

Delineate-Anything can't be faked this way: its download fetches a zip from `https://github.com/...` first (`ModelManager._download_github_repo`), so the preparing flow uses Prithvi on a six-band copy of the sample instead.

## LLM modes

`server/app.py` calls `load_dotenv()` without `override`, so a variable the runner sets, even to `""`, beats `.env`.

- **`none`** (default, used in gates and CI): every provider key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GPT_SECRET`) is blanked, `APERTUS_ENABLED=false`, `LLM_PROVIDER_ORDER=ollama` and `LLM_ENABLE_OLLAMA=false`. The router has no provider, so the backend is "assistant offline".
- **`fake`**: same blanking, plus `LLM_ENABLE_OLLAMA=true`, `OLLAMA_BASE_URL=http://127.0.0.1:<port>` and `OLLAMA_MODEL=fake`, all pointing at `fake-ollama.mjs`. It implements `/api/version`, `/api/tags` and `/api/chat` as `dta/dti/coe/llm/ollama.py` calls them. A reply is the first rule whose `match` regex matches the **last user message or the last system message**; if none matches, it is `default`. A rule's `delay_ms` holds its reply back that long: `workspace-running` delays the agent-analysis call (system prompt "interpreting satellite imagery analysis results") by 6 s to catch the Interpretation row running.
- **`live`**: the runner leaves the LLM env alone, so `.env` providers are used. It is for local checks only: it is slow, costs quota, and is never part of gates or CI.

In `none` and `fake` the runner also blanks `GEE_PROJECT_ID` and `GEE_SERVICE_ACCOUNT_KEY`, so Earth Engine is never initialised and region flows fail cleanly (`E2E_SERVER=demo` sets them to its dummy values instead). Other `.env` values still apply, and a `dta/config/llm.yaml` would replace the env-based LLM config entirely.

Rules file format:

```json
{
  "default": "reply when nothing matches",
  "rules": [{ "match": "regex source", "flags": "i", "reply": "canned reply", "delay_ms": 0 }]
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
