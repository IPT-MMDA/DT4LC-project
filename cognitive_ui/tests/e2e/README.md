# Browser E2E harness

Runs browser flows against the **built** UI and the **real** API. Uses `playwright-core` (no test runner of its own, no browser download) and `node --test`.

```bash
cd cognitive_ui
npm run build          # the harness serves dist/; it refuses to run without it
npm run e2e:smoke      # flows/smoke.e2e.mjs only
npm run e2e            # every flows/*.e2e.mjs (guest, E2E_LLM=none; fake- and member-only flows skip)
npm run e2e:workspace  # the Workspace screens: the four env groups below, one API run each
npm run e2e:run2       # the run-2 screens: e2e:capabilities, e2e:landing, e2e:map, e2e:library and e2e:result, their groups below
npm run e2e:signin     # GitHub sign-in: e2e:signin:fake (the fake GitHub) and e2e:signin:off (not configured)
npm run e2e:settings   # operator Settings
npm run e2e:responsive # phone 390×844 and tablet 834×1112 of the six screens (not part of e2e:run2)
npm run e2e:a11y       # axe (WCAG 2.0/2.1 A and AA) on the seven screens in both themes: e2e:a11y:guest and e2e:a11y:operator
npm run e2e:dark       # every screen in the dark theme at 1440×900: e2e:dark:guest and e2e:dark:operator
npm run e2e:run3       # the five user flows (#25): e2e:run3:guest, e2e:run3:fake, e2e:run3:quota and e2e:run3:signin
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
| `e2e:map` | guest, `E2E_LLM=none` | `map-mode` | map mode (`map-mode.png` at 1440×900, `map-mode-1280.png` at 1280×800; `map-mode-compare.png`, not a design screen): NDWI change On map from the chat, the sample in true colour with its footprint under it, the legend with the class areas, a rectangle drawn with the mouse and renamed, Fit to result, Compare (pixels on both sides of the divider, moved by keyboard and drag), Fetch to map disabled without Earth Engine, Send to chat to the region chip; at 1280×800 no panel or control overlaps another |
| `e2e:library` | guest, `E2E_LLM=none` | `library` | the Library (`library.png`): seeded through the UI with the sample, NDWI change, statistics for every band and a region job failing with `EE_NOT_CONFIGURED`; rows, types, statuses, conversation names and WHEN; each filter narrows the list; CSV is a 200 `text/csv`; a row opens `/app/results/:id`; Map puts the result on the map in map mode; Chat attaches the sample; Retry resends in the region conversation; Delete removes the row and `GET /v1/jobs/{id}` is 404; Upload GeoTIFF adds a `File · upload` row. A new session shows "Nothing here yet" |
| `e2e:result` | guest, `E2E_LLM=none` | `result-detail` | Result detail (`result-detail.png`; `result-detail-failed.png` and `result-detail-unavailable.png`, not design screens): seeded through the UI with the sample, NDWI change and a region job failing with `EE_NOT_CONFIGURED`; the Library row opens `/app/results/:id`; the breadcrumb; the difference, before and after images load (200); the class statistics with the sample's areas (156.8 / 0.6 / 373.5 km²) and shares; the plan's four steps with parameters and times; the downloads with kinds and sizes; CSV is a 200 `text/csv`. Open on map shows map mode with the result and sample layers; Re-run with changes puts the prompt and the sample chip in the composer and creates no job; Open in workspace selects the job's conversation. The failed job shows its code and fixes; an unknown id is "not available" |
| `e2e:signin:fake` | `E2E_SIGNIN=fake` (no `DEV_ROLE`: the role comes from sign-in), `E2E_LLM=none` | `signin` | `signin-guest-menu.png`, `signin-member-menu.png`, `signin-denied.png` (not design screens): a guest's menu has "Sign in with GitHub" (a link to `/v1/auth/github/login`) and gets the not-found page at `/app/settings`; the allowlisted member signs in through the fake GitHub (Member badge with no quota line, `analyses.limit` null, name, login and email in the menu, no Settings, no "Guest items expire" in the Library, not found at `/app/settings`) and signs out to a guest; the operator gets Settings, `/app/settings` renders, and removing their own address from `operator_emails` is refused inline (422); the outsider, an unverified email and a cancelled consent each land on `/app` as a guest with their notice, the query cleaned from the URL |
| `e2e:signin:off` | guest, `E2E_LLM=none` | `signin-off` | `signin-not-configured.png` (not a design screen): "Sign-in not configured on this server" and no sign-in control; `/v1/auth/github/login` is 404 `SIGN_IN_NOT_CONFIGURED`; not found at `/app/settings` |
| `e2e:settings` | `E2E_DEV_ROLE=operator`, `E2E_SETTINGS=1`, `E2E_LLM=fake` | `settings` | `settings.png` and `settings-assistant.png`, `settings-models.png`, `settings-access.png`, `settings-data-sources.png`, `settings-server.png` (no design screen exists): the five sections from `GET /v1/settings`; the Test button probes the fake Ollama; Move Mistral up and a guest limit of 7 send a PUT with only those two fields, and a reload shows them as overrides; Reset to .env sends `null` for both; Pre-download starts a (fake Hub) download and shows its percentage; the fake provider keys appear neither in the GET nor in the page |
| `e2e:responsive` | guest, `E2E_LLM=none` | `responsive` | `out/responsive/<screen>-phone.png` and `<screen>-tablet.png` for `landing`, `workspace`, `map`, `capabilities`, `library` and `result-detail` (the scrolling pages at full length), plus `landing-phone-menu.png`, `workspace-phone-menu.png`, `map-phone-sheet.png` and `workspace-tablet-collapsed.png`; `report-<device>.json` holds each screen's audit. One context per device with reduced motion, seeded through the UI: the landing as a first-time guest, then the sample and NDWI change, On map with a stored drawn region, Capabilities, the Library and the result's detail. Every screen passes `layout.mjs`: the document no wider than the viewport, no element past an edge (outside the map's container and the landing hero, which clip their imagery, and truncated lines), every horizontal scroller named (none today), and no two pieces of visible text overlapping by more than 1 px (buttons and block links by their box), except the region label drawn on the map; with a menu open only the menu is checked. Phone: the hero at `min(44px, 11vw)` in at most four lines with the caption under the buttons, the landing nav behind Menu; the Workspace without the list (it is in the top bar's menu, with the account menu on screen) and the composer above a keyboard (the viewport at 55 % height); the imagery panel a sheet under the map (≥ 60 % of the height collapsed, ≥ 30 % open), its header clear of the map controls; Capabilities in one column; the Library as cards; Result detail in one column with download names in full. Tablet: the landing nav inline; the conversation list hides and comes back; a 272 px imagery panel; Capabilities in two columns; the Library without the Conversation column; Result detail stacked |
| `e2e:a11y:guest` | guest, `E2E_LLM=none` | `a11y` | `out/a11y/<screen>-<theme>.json` for `landing`, `workspace`, `map`, `capabilities`, `library` and `result-detail`, `light` and `dark`: every axe violation with its nodes. One 1440×900 context with reduced motion, seeded through the UI as in `screens.mjs` (the landing as a first-time guest, the sample and NDWI change done, On map with a stored region, Capabilities, the Library, the result's detail). The theme switches with the rail's toggle (the landing: the stored theme and a reload). Fails on any violation of impact serious or critical with the tags `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`. No selector is excluded |
| `e2e:a11y:operator` | `E2E_DEV_ROLE=operator`, `E2E_SETTINGS=1`, `E2E_LLM=fake` | `a11y` | `out/a11y/settings-light.json` and `settings-dark.json`, the same rule |
| `e2e:dark:guest` | guest, `E2E_LLM=none` | `dark` | `out/dark/<screen>.png` at 1440×900 with the stored theme dark from the first page: `landing` (and `landing-full`, the whole page), `workspace-empty`, `workspace-proposal`, `workspace-done`, `map`, `capabilities`, `library`, `result-detail`, `not-found`, `workspace-failed`. Each screen has `data-theme="dark"` and passes `layout.mjs` at 1440 (`report-guest.json`) |
| `e2e:dark:operator` | `E2E_DEV_ROLE=operator`, `E2E_SETTINGS=1`, `E2E_LLM=fake` | `dark` | `out/dark/settings.png` and `workspace-running.png` (the fake Ollama holds the Interpretation call for 8 s); `report-operator.json`. The Settings save bar is sticky, so it is named as an overlay |
| `e2e:run3:guest` | guest, `E2E_LLM=none` | `flow-upload-analyze-result`, `flow-map-to-chat`, `flow-follow-up` | flows 1–3 below (`flow-upload-result-card.png`, `flow-upload-result-detail.png`, `flow-map-region.png`, `flow-map-failed.png`, `flow-map-retry-done.png`, `flow-follow-up-none.png`; not design screens) |
| `e2e:run3:fake` | guest, `E2E_LLM=fake` | `flow-follow-up` | flow 3 with the fake LLM: the typed question gets the scripted reply, and the LLM saw both analyses in the history (`flow-follow-up-fake.png`) |
| `e2e:run3:quota` | guest, `E2E_GUEST_LIMIT=2`, `E2E_LLM=none` | `flow-guest-quota` | flow 4 (`flow-guest-quota.png`) |
| `e2e:run3:signin` | `E2E_SIGNIN=fake` (no `DEV_ROLE`), `E2E_GUEST_LIMIT=1`, `E2E_LLM=none` | `flow-signin` | flow 5 (`flow-signin-member.png`) |

`signin` runs only with `E2E_SIGNIN=fake`, `signin-off` only as a guest without it, and `settings` only as an operator with `E2E_SETTINGS=1` and `E2E_LLM=fake`. `a11y` and `dark` run their guest test as a guest with `E2E_LLM=none` and their operator test as an operator with `E2E_SETTINGS=1` and `E2E_LLM=fake`; `screens.mjs` holds the steps both share.

`workspace-running` skips itself unless `E2E_LLM=fake`, `workspace-preparing` unless `E2E_DEV_ROLE=member`, and `workspace-quota` unless `E2E_GUEST_LIMIT=2`. `capabilities-offline` runs only as a member with `E2E_LLM=none` on a local server, and `capabilities-demo` only as a guest with `E2E_SERVER=demo` and `E2E_LLM=fake`. `landing` runs only as a guest, `landing-member` only with `E2E_DEV_ROLE=member`.

`map-mode` also covers the region presets: at 1280×800 the presets row sits under the region card and regions are amber; Kakhovka dam (the sample footprint here), Brienz (zooms to the lake) and Upload GeoJSON (a point is refused, a polygon becomes the region).

## The five user flows (`npm run e2e:run3`)

Each is a journey through the UI against the real API, with nothing done through an API shortcut (the flows read `GET /v1/jobs` and `/v1/session` only to check what the UI did). `e2e:run3` runs its four groups one after another and must exit 0 twice in a row; each group starts its own API, so the counts below start at zero.

1. **`flow-upload-analyze-result`** (upload → analyze → result). The composer's + opens the file picker and the real input uploads `dnipro-floodplain.tif`, a renamed copy of `resources/kahovka_data/hlsl_20230609.tif` in a temp dir (a 200 from `POST /v1/upload`, never `/v1/sample`). The chip reads `dnipro-floodplain.tif HLS L30 · B2–B6 · 30 m · 3.7 MB`. "Map vegetation health across the region" runs to done: the intent line "I'll compute NDVI from dnipro-floodplain.tif.", the done line "3 steps · … · NDVI", Show steps lists Read / NDVI / Interpretation all done, and the job's events go intent, planning, three steps, done. The result card (`NDVI`, kind index) has its map image loaded, the ramp legend `−0.2 0.8`, the source line and the stats `MEAN 0.17 MIN −1 MAX 1 VALID PIXELS 589 824`. Details › opens `/app/results/:id`: the title, the breadcrumb, the images loaded, the plan's three done steps, the statistics rows, the Asked card with the file, and the downloads `ndvi.tif`, `ndvi.png`, `ndvi_map.png`, `statistics.csv`. Clicking `ndvi.tif` downloads a TIFF (`II*\0`) saved as `ndvi.tif`, not the stored `1_NDVIMap_ndvi.tif`; the link is a 200 `image/tiff` with that Content-Disposition.
2. **`flow-map-to-chat`** (map draw → send to chat → analysis). In `/app/map` a rectangle is drawn with the toolbar by a real mouse drag (the stored region is a four-corner box), renamed "Dnipro floodplain · Kherson" (the region label and its area), set to HLS L30 and 2023-06-01 → 2023-06-09, then Send to chat puts the region chip in the composer. "Compare water before and after the dam breach" goes out with the region (name, dataset, dates, polygon) and no files, and ends `failed` with `EE_NOT_CONFIGURED`, its message and the fixes Use my own image / Try the Kakhovka sample. **There is no Earth Engine key here, so that failure is the honest end of a drawn-region analysis**; Try the Kakhovka sample re-sends the prompt with the sample pair and no region and reaches done with the Water change · NDWI card (+157 km² gained, 0.6 km² lost). Two jobs: one failed, one completed. Skips on `E2E_SERVER=demo`.
3. **`flow-follow-up`** (a follow-up turn). After NDWI change on the sample, the follow-up chip "Compare with NDVI change between the same dates" sends a second `/v1/chat` with the same two file paths and the first analysis in its history, and reaches done with the Vegetation change · NDVI card. The typed "Explain the last result in two sentences" gets a conversation reply and no job: with `E2E_LLM=fake` the scripted reply (`setLLMRules`), and the fake Ollama's request log (`llmRequests()`) shows that call's system message carried both questions and both analyses' outcomes; with `E2E_LLM=none` the offline catalog reply. Then `GET /v1/jobs` has the two completed jobs and the stored conversation is user, analysis, user, analysis, user, assistant.
4. **`flow-guest-quota`** (needs `E2E_GUEST_LIMIT=2`). The badge starts at 2 of 2; a typed NDWI change leaves 1 (the low-quota pill) and a follow-up chip leaves 0, with the composer's quota line giving the local reset time from `/v1/session`. A third request ("Map moisture change with NDMI…") is a 429 `GUEST_QUOTA` from `/v1/chat`, shown inline as "Guests can run 2 analyses per hour on this server. The next one is available at HH:MM UTC." (the same reset time), with no analysis message and still two jobs; the input stays editable. "hello" gets a reply that isn't an error and no job; the badge says "0 of 2 analyses left", and after a reload it still does, with the quota line. `workspace-quota` keeps the design screenshot of this state.
5. **`flow-signin`** (needs `E2E_SIGNIN=fake` and `E2E_GUEST_LIMIT=1`). A guest uses the hour's one analysis (badge 0 of 1, the quota line). The account menu's "Sign in with GitHub" (`/v1/auth/github/login`) goes through `fake-github.mjs` as `member` and back to a clean `/app`: the session is a member with no limit, the badge "Member. Local server · no analysis limit · assistant offline.", no quota pill or line, the menu's Member badge and Sign out, and the guest's job is the member's now. Two more analyses (a follow-up chip and a typed one) run with nothing refused: three jobs, still no limit. Sign out lands on `/` as a guest with no user and, from the same address, 0 of 1 left; `/app` shows the guest badge and "Sign in with GitHub", and the guest sees none of the member's jobs. `signin` (`e2e:signin:fake`) still covers the operator, the denied states and the menus.

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
| `E2E_SIGNIN` | `off` | `fake`: GitHub sign-in configured against `fake-github.mjs` (below). The runner sets `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` to the fake's, `GITHUB_OAUTH_BASE` and `GITHUB_API_BASE` to the fake, `OAUTH_REDIRECT_BASE` to the UI's origin (its port is picked before the API starts), `ALLOWED_EMAILS` to the fake member's and `OPERATOR_EMAILS` to the fake operator's, and blanks `DEV_ROLE`, so the role comes from sign-in. Refuses to run with `E2E_DEV_ROLE`. In every other run sign-in stays unconfigured, whatever `.env` says. |
| `E2E_SETTINGS` | unset | `1`: the Settings group's API env. `GEMINI_API_KEY` and `GROQ_API_KEY` get the fake values of `FAKE_PROVIDER_KEYS` (`harness.mjs`), which the flow looks for in the page; `LLM_PROVIDER_ORDER=ollama,mistral` (no Mistral key), so there are two providers to reorder and nothing calls Gemini or Groq. Not with `E2E_LLM=live`. |
| `PLAYWRIGHT_CHROMIUM_EXECUTABLE` | newest `~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell` | Browser binary. The cached revision does not have to match the `playwright-core` version. |

Flows receive these from the runner (don't set them yourself): `E2E_BASE_URL` (UI origin), `E2E_API_URL` (API origin, direct), `E2E_WS_ENDPOINT` (browser server), `E2E_OUT`, `E2E_LLM`, `E2E_FAKE_LLM_URL`, `E2E_FAKE_HF_URL`, `E2E_DEV_ROLE` (empty with `E2E_SIGNIN=fake`), `E2E_SERVER`, `E2E_SIGNIN`, `E2E_FAKE_GITHUB_URL` and `E2E_SETTINGS`.

Every run also points `DT4LC_SETTINGS_PATH` (the operator overrides file that `PUT /v1/settings` writes) into the temp dir.

## Fake HuggingFace Hub

`huggingface_hub` honours `HF_ENDPOINT`. For each file `hf_hub_download` sends `HEAD /{repo}/resolve/{revision}/{file}` and needs `X-Repo-Commit`, an `ETag` and `Content-Length` back, then `GET`s the same path. `fake-hf.mjs` answers every such path with 120 MB of zeros streamed at 8 MB/s, so a member's Preparing model step shows real `step_progress` (bytes, total, ETA) for about 15 s per file. `GET /__e2e/requests` lists the requests it saw. A flow cancels the job once it has its picture, before the model step would load the fake weights.

Delineate-Anything can't be faked this way: its download fetches a zip from `https://github.com/...` first (`ModelManager._download_github_repo`), so the preparing flow uses Prithvi on a six-band copy of the sample instead.

## Fake GitHub

`fake-github.mjs` (with `E2E_SIGNIN=fake`) answers what `server/github_oauth.py` calls: `GET /login/oauth/authorize` 302s to `redirect_uri?code&state`; `POST /login/oauth/access_token` (JSON, checks the client id and secret) returns `{access_token}`; `GET /user` and `GET /user/emails` read the bearer token. The scripted accounts are `FAKE_USERS`: `member` (allowlisted), `operator` (an operator email), `outsider` (on no list), `unverified` (primary email not verified) and `denied` (the authorize step returns `error=access_denied`). A flow picks who signs in next with `signInAs(user)` from `harness.mjs` (`PUT /__e2e/user`); the default is `member`. Avatars are `null`, so no image leaves the machine. `fake-github.test.mjs` is its vitest test.

## LLM modes

`server/app.py` calls `load_dotenv()` without `override`, so a variable the runner sets, even to `""`, beats `.env`.

- **`none`** (default, used in gates and CI): every provider key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `GPT_SECRET`) is blanked, `APERTUS_ENABLED=false`, `LLM_PROVIDER_ORDER=ollama` and `LLM_ENABLE_OLLAMA=false`. The router has no provider, so the backend is "assistant offline".
- **`fake`**: same blanking, plus `LLM_ENABLE_OLLAMA=true`, `OLLAMA_BASE_URL=http://127.0.0.1:<port>` and `OLLAMA_MODEL=fake`, all pointing at `fake-ollama.mjs`. It implements `/api/version`, `/api/tags` and `/api/chat` as `dta/dti/coe/llm/ollama.py` calls them, and logs every chat request: `GET /__e2e/requests` returns `[{messages, reply}]` (`llmRequests()` in `harness.mjs`), so a flow can check what the backend sent the LLM, e.g. the conversation history. A reply is the first rule whose `match` regex matches the **last user message or the last system message**; if none matches, it is `default`. A rule's `delay_ms` holds its reply back that long: `workspace-running` delays the agent-analysis call (system prompt "interpreting satellite imagery analysis results") by 6 s to catch the Interpretation row running.
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

`setLLMRules`, `resetLLMRules` and `llmRequests` throw unless `E2E_LLM=fake`. A flow that needs the fake LLM should skip itself otherwise (`test('…', { skip: process.env.E2E_LLM !== 'fake' }, …)`: the options are read before `before()` runs). The fake server exposes three control endpoints, `PUT /__e2e/rules` (replace the rules), `POST /__e2e/reset` (restore the file's rules) and `GET /__e2e/requests` (the chat requests so far); the helpers wrap them.

`fake-ollama.test.mjs` is a vitest unit test (runs in `npm test`) that checks the fake server's canned replies and control endpoints.

## Linux: missing shared libraries

On a bare Linux or WSL install, `chrome-headless-shell` may fail to start because `libnss3.so`, `libnspr4.so`, `libnssutil3.so` or `libasound.so.2` are missing, and installing them needs root. Without root:

```bash
tests/e2e/setup-libs.sh
```

The script runs `apt-get download libnss3 libnspr4 libasound2t64` (falling back to `libasound2`) and `dpkg -x` into `tests/e2e/.libroot/`, which is git-ignored. When `tests/e2e/.libroot/usr/lib/x86_64-linux-gnu` exists, the runner adds it to the browser's `LD_LIBRARY_PATH`. If the browser still fails with a missing `.so`, the runner names the library and points at the script.

CI installs the shell and its system dependencies with `npx playwright-core install --with-deps chromium-headless-shell` instead. Don't run that locally without `--no-remove`: it deletes cached browser revisions that `playwright-core` 1.64.1 doesn't use.
