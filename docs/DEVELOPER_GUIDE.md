# Developer Guide

Setup, layout, tests, adding algorithms, the COE flow, the job event stream, the frontend, the browser E2E harness, and the git workflow for DT4LC.

## Environment setup

**Prerequisites:** Python 3.10+, [uv](https://github.com/astral-sh/uv), optional [Task](https://taskfile.dev/).

```bash
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project
cp .env.example .env
```

With `VITE_API_URL` unset, the Vite dev server proxies `/v1`, `/docs` and `/redoc` to `http://localhost:8000` on the UI's own origin.

The backend loads `.env` via `load_dotenv()` in `server/app.py`. **Source of truth:** [`.env.example`](../.env.example) (commented defaults). Below is what matters for local dev.

### Backend `.env` (root)

| Group | Variables | Notes |
|-------|-----------|--------|
| **LLM routing** | `LLM_ENABLE_*`, `LLM_PROVIDER_ORDER`, `LLM_STRATEGY` | Enable/disable providers (an unset `LLM_ENABLE_*` counts as `true`); order is fallback chain (`fallback` strategy). Commercial providers (`anthropic`, `mistral`, `openai`) need `LLM_ENABLE_*=true`, a key, and an entry in `LLM_PROVIDER_ORDER`. The router moves to the next provider when one is unavailable or its call raises, and for context extraction also when the reply fails the context check; on an unusable reply, intent classification and planning fall back deterministically instead of asking the next provider; result interpretation uses the reply as is. |
| **API keys** | `GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY` (or `GPT_SECRET`) | Optional for local dev; without keys the router uses the next enabled provider (often Ollama). |
| **Models** | `GEMINI_MODELS`, `GROQ_MODEL`, `ANTHROPIC_MODELS`, `MISTRAL_MODELS`, `OPENAI_MODELS` | Comma-separated lists; one model per provider is picked when the router is built (once per process). Gemini tries the rest of its list on 404/429/503. Groq takes a single model. |
| **Ollama** | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Default `http://localhost:11434`. Under Docker Compose, `docker-compose.yml` defaults it to `http://ollama:11434` (the `ollama` service starts with `--profile ollama`); leave it unset in `.env`. An untagged `OLLAMA_MODEL` means `:latest`; a tagged one must match an installed tag. |
| **Apertus (GPU)** | `APERTUS_ENABLED`, `APERTUS_MODEL`, `APERTUS_DEVICE`, `APERTUS_DTYPE` | Off by default; needs GPU, `APERTUS_ENABLED=true`, `LLM_ENABLE_APERTUS=true` and `apertus` in `LLM_PROVIDER_ORDER`. |
| **Earth Engine** | `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_KEY` | Only for the GEE data fetch routes (`/v1/gee/*`). |
| **Jobs** | `JOB_STORAGE`, `DT4LC_JOBS_DB` | `sqlite` (default) or `memory` (lost on restart). DB path defaults to `resources/.cache/jobs.db`. |
| **Ports (Docker)** | `BACKEND_PORT`, `FRONTEND_PORT` | Host port mapping in `docker-compose.yml`; locally `task run:api` (`dt4lc-api`) listens on 8000 (`DT4LC_PORT`). |
| **Sessions and sign-in** | `SESSION_SECRET`, `SESSION_COOKIE_SECURE`, `DEV_ROLE`, `GUEST_ANALYSES_PER_HOUR`, `GUEST_SESSION_HOURS`, `GUEST_CLEANUP_MINUTES`, `MODEL_DOWNLOAD_ON_DEMAND`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `OAUTH_REDIRECT_BASE`, `GITHUB_OAUTH_BASE`, `GITHUB_API_BASE`, `ALLOWED_EMAILS`, `ALLOWED_DOMAINS`, `OPERATOR_EMAILS`, `FORWARDED_ALLOW_IPS` | Without the three GitHub variables sign-in is off and everyone is a guest; `DEV_ROLE=member` or `DEV_ROLE=operator` gives every request that role instead, the quickest way to see the member badge or Settings locally. See [API.md](API.md#sessions) and [API.md](API.md#sign-in). |
| **Paths** | `DT4LC_CACHE_PATH`, `DT4LC_TEMP_PATH`, `DT4LC_OUTPUTS_PATH`, `DT4LC_UPLOADS_PATH`, `DT4LC_MODEL_CACHE`, `DT4LC_JOBS_DB`, `DT4LC_SETTINGS_PATH` | All under `resources/.cache/` by default (`dta/config/__init__.py`); the others follow `DT4LC_CACHE_PATH`. The E2E runner points them all at a temp dir. |
| **Server** | `CORS_ORIGINS`, `DT4LC_DEPLOYMENT`, `DT4LC_HOST`, `DT4LC_PORT`, `DT4LC_RELOAD`, `DT4LC_LOG_LEVEL` | `DT4LC_DEPLOYMENT=demo` adds "public demo" to the Capabilities line. The last four are read by `dt4lc-api` (`server/cli.py`) only, not by `uvicorn` or the Docker image. |
| **Demo** | `COMPOSE_FILE`, `COMPOSE_PROFILES`, `DEMO_SUBNET`, `DEMO_SKIP_MODEL_DOWNLOAD` | Read by Docker Compose and `docker-compose.demo.yml`; see [DEPLOY_DEMO.md](DEPLOY_DEMO.md). |

**Minimal local COE (no Docker):** set at least one of `GEMINI_API_KEY` or `GROQ_API_KEY`, or run Ollama locally and set `LLM_PROVIDER_ORDER=ollama` with `LLM_ENABLE_OLLAMA=true`.

Full LLM options and examples are also in [README.md](../README.md#configuration).

### Frontend `cognitive_ui/.env`

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_API_URL` | unset: the UI's own origin | API origin for the API client (`src/api/client.ts`), the SSE streams, the upload/file hooks (`src/api/hooks/useUpload.ts`) and GeoTIFF tiles (`GeoTIFFLayer.tsx`) |

It is the only frontend variable. Leave it unset: sessions need the UI and the API on one origin (the Vite proxy in development, nginx under Docker, whose image builds with `VITE_API_URL=""`).

**Backend:**

```bash
task venv && task install          # or: uv sync --extra dev --extra server
task run:api                       # http://localhost:8000
uv run uvicorn server.app:app --reload --port 8000   # with hot reload
```

**Frontend:**

```bash
cd cognitive_ui && npm install && npm run dev   # http://localhost:5173
```

**Docker (full stack):** `docker compose --profile ollama up -d` (or `docker compose up -d` with API keys, without Ollama) → UI at http://localhost, Swagger at http://localhost/docs, API at :8000.

**Pre-commit (optional):** `pip install pre-commit && pre-commit install`

## Project structure

| Path | Role |
|------|------|
| `dta/dti/coe/` | Context Orchestration Engine (intent, context, plan, validate) |
| `dta/dti/algorithms/` | Raster algorithms (`run(RasterPath=...)`) |
| `dta/dti/models/` | ML models (Prithvi, Delineate-Anything) |
| `dta/registry.yaml` | Types, components, triggers, runners |
| `dta/dti/executor.py` | Runs validated execution plans |
| `server/` | FastAPI (`app.py`, `routes/`, job queue) |
| `cognitive_ui/` | React + TypeScript UI |
| `tests/` | pytest suite |

Bundled algorithms use dotted module paths as registry entrypoints (`dta.dti.algorithms.ndvi`). Filesystem paths (ending in `.py` or containing `/`) are also accepted and are used for the models: Prithvi's `${MODEL_PATH}/inference.py` is a file of the downloaded model (`${MODEL_PATH}` becomes the model's cache directory), and Delineate-Anything's `dta/dti/models/third_party/delineate_anything/inference.py` is a committed wrapper that runs the downloaded upstream code.

## Running tests

```bash
uv run pytest tests/ -m "not slow and not llm"   # what CI runs: no keys, network or models
task test              # all tests (uv run pytest); llm/GEE/model tests skip without keys or weights
task test:fast         # skip slow (-m "not slow")
uv run pytest tests/ -m llm             # opt-in: needs an available LLM provider
uv run pytest tests/ -m network         # opt-in: GEE_PROJECT_ID + Earth Engine credentials
uv run pytest tests/ -m "slow or gpu"   # opt-in: downloaded models + NVIDIA GPU
cd cognitive_ui && npm test             # frontend (vitest)
```

**By area:**

```bash
pytest tests/test_registry.py tests/test_algorithms.py -v
pytest tests/test_orchestrator.py tests/test_intent_classifier.py -v
pytest tests/test_executor.py -v
```

**Coverage:** `task coverage` or `pytest tests/ --cov=dta --cov=server --cov-report=html`

### Running everything

The gates every change must pass, backend then frontend:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy dta server
uv run pytest tests/ -m "not slow and not llm" -q
cd cognitive_ui
npm run lint                  # keep it at 0 errors and 0 warnings
npm run typecheck             # tsc -b --noEmit
npm run build                 # tsc -b, then vite build
npm test                      # vitest
npm run e2e:smoke             # Playwright, against dist/ and a real API
```

The pytest session writes uploads, temporary files, job outputs and the jobs DB into a temporary directory that it deletes at exit (`tests/conftest.py`; a `DT4LC_UPLOADS_PATH`, `DT4LC_TEMP_PATH`, `DT4LC_OUTPUTS_PATH` or `DT4LC_JOBS_DB` already set in your shell wins), so a gate run adds nothing to `resources/.cache`.

The type check is `npm run typecheck`, and CI runs the same `npx tsc -b --noEmit`. A plain `npx tsc --noEmit` checks nothing here: `tsconfig.json` has `files: []` and only references the app and node projects, which `-b` follows. `src/stockColours.test.ts` (part of `npm test`) fails on any stock Tailwind palette class in `src/`, or a hex value from that palette: take colours from the design tokens in `src/index.css` and, for MapLibre paint, from `components/map/mapColours.ts`.

The browser groups, each with its own API process (see [Browser E2E harness](#browser-e2e-harness)):

```bash
npm run e2e:workspace && npm run e2e:run2 && npm run e2e:run3
npm run e2e:signin && npm run e2e:settings
npm run e2e:responsive && npm run e2e:dark && npm run e2e:a11y
```

To use the app locally: `task run:api` (or `uv run uvicorn server.app:app --reload --port 8000`) and `cd cognitive_ui && npm run dev`, then open <http://localhost:5173>. Under Docker: `docker compose up -d --build` (add `--profile ollama` for the local LLM) and <http://localhost>. The public demo stack is in [DEPLOY_DEMO.md](DEPLOY_DEMO.md).

## Adding a new algorithm

Example: a custom spectral index `MYI` (same pattern as NDVI/EVI).

### 1. Algorithm module

Create `dta/dti/algorithms/my_index.py`:

```python
from typing import Any

from dta.dti.algorithms.spectral_index import run as _run_index
from dta.dti.registry import get_item, load_registry


def _config() -> dict[str, Any]:
    return get_item(load_registry(), "algorithms/my-index").config or {}


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    return _run_index(RasterPath, _config())
```

`spectral_index.run` writes the index as a GeoTIFF and its map as a PNG and returns references to them in `rasters["<formula>"]` and `visualizations["<formula>_map"]` (here `ndvi` and `ndvi_map`, the formula in the config below).

For non–spectral-index logic, implement `run()` returning a dict with results/statistics (see `dta/dti/algorithms/statistics.py`). A result must not hold pixel arrays or images ("Job artifacts" in [ARCHITECTURE.md](ARCHITECTURE.md#job-artifacts)). Pass them to `artifact_files.save`, which writes each 2-D array as a single-band float32 GeoTIFF (nodata NaN) and each PNG as a file, and puts `{kind, path, url, stats}` references into `result["rasters"]` and `result["visualizations"]` under the same keys:

```python
import rasterio

from dta.dti import artifact_files


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    with rasterio.open(RasterPath) as src:
        mask = ...  # 2-D numpy array, NaN where invalid
        png = ...   # PNG bytes, or None
        result = {"statistics": {...}}
        return artifact_files.save(result, {"my_mask": mask}, {"my_mask_map": png}, crs=src.crs, transform=src.transform)
```

In a pipeline the files go to the job's directory as `OUTPUTS_PATH/<job_id>/<step index>_<output type>_<key>.<tif|png>`; called directly, to a new `OUTPUTS_PATH/run-<uuid hex>/`. To compute a spectral index without writing files (e.g. to threshold it), call `spectral_index.calculate_index(path, config)`, which returns `(result, array, (crs, transform))`. A step that needs another step's pixels opens `rasters["<key>"]["path"]` with rasterio.

The executor calls `run(**inputs)` with one keyword argument per entry of the item's `inputs`, named after the type (`RasterPath=...`); a runner with `args_map:` instead calls `function:` (default `main`) with the mapped arguments. With a single output type the returned dict is stored as that output; with several, return a dict keyed by output type.

### 2. Registry entry

Add to `dta/registry.yaml` under `types` and `instances`:

```yaml
types:
  - MYIMap   # add to types list

instances:
  - id: algorithms/my-index
    kind: algorithm
    display_name: "MYI Calculation"
    keywords: [myi, my, index]
    inputs: [RasterPath]
    outputs: [MYIMap]
    runner:
      type: python
      entrypoint: "dta.dti.algorithms.my_index"
    triggers:
      keywords: [myi, "my index"]
      action_phrases:
        - "calculate myi"
        - "compute myi"
    user_guide:
      capability_response: "Yes, I can calculate MYI. Upload a GeoTIFF with the required bands."
      missing_file_response: "Please upload a GeoTIFF to calculate MYI."
      summary_template: "Calculating MYI"
    config:
      formula: ndvi          # a key of FORMULAS in spectral_index.py; a new formula needs an entry there (and in RANGES to be clipped)
      required_bands: [red, nir]
      colormap: ndvi
      vmin: -0.2
      vmax: 0.8
```

Triggers drive intent classification and context-agent hints without editing COE Python. `user_guide.capability_response` and `missing_file_response` are the intent classifier's answers to "can you ...?" and to an action without an upload; no code reads `summary_template` yet. The trigger index is built once per process (`dta/dti/coe/triggers.py`), so restart the backend after editing `registry.yaml`.

### 3. Tests

In `tests/test_algorithms.py` (or a new `tests/test_my_index.py`):

```python
def test_my_index_run_exists() -> None:
    from dta.dti.algorithms.my_index import run
    assert callable(run)

def test_registry_has_my_index() -> None:
    from dta.dti.registry import get_item, load_registry
    item = get_item(load_registry(), "algorithms/my-index")
    assert item.outputs == ["MYIMap"]
```

Add the module to the parametrized list of `test_single_raster_algorithm_results_hold_refs` in `tests/test_job_artifacts.py`, as `("my_index", {<rasters keys>}, {<visualizations keys>})`; the test fails if the result holds arrays or base64 images instead of references.

Run: `pytest tests/test_algorithms.py tests/test_job_artifacts.py -v` (and `tests/test_registry.py` if you added types).

## COE pipeline walkthrough: "Calculate NDVI"

End-to-end path when a user uploads a GeoTIFF and asks **"Calculate NDVI"**:

```text
ChatRequest
    → orchestrator.orchestrate()
        → intent_classifier.classify_intent()     # PIPELINE (attachment + "ndvi" keyword)
        → context_agent.analyze()                 # desired_outputs: ["NDVIMap"]
        → planner.plan()                          # input/file → algorithms/ndvi → post-processing/agent-analysis
        → plan_validator.validate()               # resource limits, then input/output types
    → executor.execute(plan)                      # runs each PlanStep
```

| Step | Module | What happens for NDVI |
|------|--------|------------------------|
| 1 | `orchestrator.py` | Classifies intent; loads registry; builds the plan, injects attachment paths into input steps (`_inject_file_bindings`) and validates it |
| 2 | `intent_classifier.py` | `registry.yaml` `triggers` on `algorithms/ndvi` match `"calculate ndvi"` → `PIPELINE` (with attachment) or `CONVERSATION` (asks for file if missing) |
| 3 | `context_agent.py` | LLM extracts `desired_outputs: ["NDVIMap"]`; domain hints rendered from registry. If no provider answers or every reply fails the context check, the context is built from the registry triggers instead (`algorithms/ndvi` gives `["NDVIMap"]`) |
| 4 | `planner.py` | Template planner when `estimate_plan_confidence()` (`llm_planner.py`) is at least 0.7, else the LLM planner; the template planner is also used when the LLM planner fails or returns invalid JSON or an unknown component. Template plan: `input/file` → `algorithms/ndvi` → `post-processing/agent-analysis` |
| 5 | `plan_validator.py` | `PlanValidator.check_resources` (`dta/dti/validation.py`): at most 50 steps, 600 s and 4096 MB estimated (per step: algorithm 5 s / 256 MB, model 30 s / 1024 MB, other 1 s / 50 MB); then every step's inputs must come from earlier outputs or binds. A failing plan is not executed; the orchestrator then tries the template plan once and uses it only if it passes and has an algorithm or model step |
| 6 | `executor.py` | `input/file` passes the bound `RasterPath` through; imports `dta.dti.algorithms.ndvi` and calls `run(RasterPath=...)` |

**API wiring:** in `server/routes/chat.py`, `/v1/execute` calls `orchestrate()` then `PipelineExecutor().execute()` on the validated plan, and `/v1/plan` stops after `orchestrate()`. Async jobs (`POST /v1/jobs`) use the same flow in `server/jobs.py`. `/v1/chat` classifies the message, answers conversation itself and submits an analysis as `POST /v1/jobs` does (see [API.md](API.md#post-v1chat)).

**Verify:** `pytest tests/test_orchestrator.py tests/test_intent_classifier.py -v`

## The job event stream

An analysis is a job whose progress is a log of events (`intent`, `planning`, `proposal`, `step_started`, `step_progress`, `step_finished`, then one of `done`, `failed`, `cancelled`). The payloads and the failure codes are in [API.md](API.md#get-v1jobsjob_idevents); the server side (`server/job_events.py`, `JobQueue._emit` and `subscribe`) is described in [ARCHITECTURE.md](ARCHITECTURE.md#job-event-stream-and-proposals). To emit a new event, add its type to `EVENT_TYPES` in `server/job_events.py`, emit it from `server/jobs.py`, document it in API.md, and teach the reducer below what it means.

**Consuming it from a script.** The stream is plain Server-Sent Events; keep the session cookie, as for any job route:

```bash
curl -N -b cookies.txt http://localhost:8000/v1/jobs/<job id>/events            # the whole log, then the live tail
curl -N -b cookies.txt "http://localhost:8000/v1/jobs/<job id>/events?after=8"  # resume after seq 8
```

**In the UI** the stream goes through four layers, each with its tests next to it:

1. `src/api/sse.ts`: `createSSEParser(onEvent)` turns text chunks into `{id, event, data}` frames (comments such as `: keep-alive` are dropped), and `readSSE(body)` is the async generator over a `fetch` response body. `POST /v1/chat` uses the same reader for its `reply_delta`, `reply_done` and `analysis` frames (`src/api/chat.ts`, `streamChat`).
2. `src/api/jobs.ts`: `streamJobEvents(jobId, after, onEvent, signal)` opens `GET /v1/jobs/{id}/events?after=<after>` and hands each frame to `onEvent` as the stored `{seq, type, data}`; a 404 throws `ChatError`.
3. `src/hooks/useJobEvents.ts`: `useJobEvents(jobId, terminal)` keeps one stream per job however many components follow it. Events are merged into the analysis message of the chat store (`applyJobEvents`, by `seq`, so a replay adds nothing twice); a dropped stream reconnects with `?after=<last seq>` after 0.5, 1, 2, 5 and then 10 s; the step into `done` fetches `GET /v1/jobs/{id}` for the result. A 404 marks the message "no longer on the server".
4. `src/components/chat/analysis/analysisReducer.ts`: `analysisReducer(events)` is a pure function from the event list to an `AnalysisView`: the phase (`starting`, `planning`, `proposal`, `running`, `done`, `failed`, `cancelled`), the step rows with their status and progress, the proposal, the failure with its fixes, the done summary and the follow-ups. `AnalysisMessage.tsx` renders the view, so a new event usually needs only a reducer case and a test in `analysisReducer.test.ts`.

## Frontend architecture

React 18, TypeScript, Tailwind, MapLibre (`react-map-gl`), TanStack Query for server state, Zustand for client state, `react-markdown` for replies, and no component library. The design tokens (colours for both themes, type scale, radii) are CSS variables in `src/index.css`, exposed as Tailwind theme values.

**Routes** (`src/routes.tsx`): `/` is the landing for a first-time guest (`pages/Landing.tsx`; `components/landing/landingDecision.ts` sends members, operators and returning guests to `/app`). Under `/app` the shell (`components/shell/Shell.tsx`) holds the pages: the Workspace (`/app`), Map mode (`/app/map`), the Library (`/app/library`), Capabilities (`/app/capabilities`), Result detail (`/app/results/:id`) and Settings (`/app/settings`); any other `/app/*` path, and Settings for a non-operator, is `pages/NotFound.tsx`. The v0.5 paths (`/chat`, `/jobs`, `/data`, `/models`, …) redirect.

| Area | Where | What it does |
|------|-------|--------------|
| Shell | `components/shell/`: `Shell`, `Rail`, `ConversationList`, `ChatColumn`, `DockedMap`, `SessionBadge`, `AccountMenu`, `PhoneMenu`, `navigation.ts` | The rail (Chat, Map, Library, Can do; theme and account at the bottom), the conversation list with the session badge, the chat column, and the map docked beside the chat at 1440 px and wider. `hooks/useMediaQuery.ts` (`useLayout()`: phone under 768 px, tablet under 1024 px) switches to the phone bar and menu drawer |
| Chat store | `store/useChatStore.ts` (persisted in `localStorage` as `dt4lc-chat-storage`, version 4) | Conversations and their messages: `user`, `assistant` (markdown) and `analysis` (one per job, with its event log). `addMessage`, `updateAnalysis`, `appendAnalysisEvent`, `findConversationByJobId`, the pending attachments and region of the composer. A guest's conversations follow the 24 h retention (`hooks/useConversationRetention.ts`) |
| App store | `store/useAppStore.ts` (`dt4lc-app-storage`) | Drawn regions, map layers (`addResultLayer`, ids `result-<jobId>-<view>`), the imagery panel, Compare, the map tool and view |
| Workspace | `pages/Workspace.tsx`, `components/chat/`: `MessageList`, `Composer`, `Chips`, `EmptyState`, `LibraryPicker`, `Markdown`, `analysis/AnalysisMessage`, `result/ResultCard` | One message per analysis, narrated from the event stream; the composer with +, From map, From library and drop; attachment and region chips. `hooks/useSendMessage.ts` sends a turn to `POST /v1/chat` with the history and the conversation's files; `hooks/useSample.ts` attaches the bundled pair (`?sample=1`) |
| Result cards | `components/chat/result/resultModel.ts`, `ResultCard.tsx` | `resultCardOf(job, context)` turns `result.execution.artifacts` into a `ResultCardModel` (index, change, classification, fields, statistics, image); `ResultCard` renders it with On map, GeoTIFF, PNG and Details. See [Adding a result card kind](#adding-a-result-card-kind) |
| Map mode | `pages/MapPage.tsx`, `components/map/mode/` (`ImageryPanel`, `LayersSection`, `MapToolbar`, `MapControls`, `MapLegend`, `RegionLabel`, `CompareSwipe`, region presets) | Draw or pick a region, choose dataset, dates, cloud and bands; Send to chat puts the region chip in the composer. Layers come from chat results, the sample (true colour, with its footprint) and Earth Engine. MapLibre can't read `oklch`, so map paint colours come from `components/map/mapColours.ts` |
| Library | `pages/Library.tsx`, `components/library/` (`libraryModel.ts`, `useLibraryActions.ts`), `api/library.ts` | One list of analyses (`GET /v1/jobs`, each with its `card`) and files (`GET /v1/files`, the sample pair as one row), filters and row actions (Map, Chat, Retry, download, CSV, Delete). It shares the `['files']` query with Map mode and invalidates it on delete |
| Result detail | `pages/ResultDetail.tsx`, `components/detail/` (`resultDetailModel.ts`, `useResultActions.ts`) | The maps, the class statistics with CSV, the plan steps with parameters and times, the interpretation and the downloads (`downloadName(ref)`) of one job, and Open in workspace, Re-run with changes, Open on map |
| Capabilities | `pages/Capabilities.tsx`, `api/hooks/useCapabilities.ts` | Renders `GET /v1/capabilities`; an example prompt opens `/app?prompt=…` with the prompt in the composer, unsent (`hooks/usePromptFromQuery.ts`) |
| Settings | `pages/Settings.tsx`, `components/settings/` (`AssistantSection`, `ModelsSection`, `AccessSection`, `ReadOnlySections`, `settingsModel.ts`, `form.ts`), `api/settings.ts` | Operators only. Reads `GET /v1/settings`, sends only the changed fields in `PUT`, `null` for Reset to .env, and runs the assistant test |
| Account menu | `components/shell/AccountMenu.tsx`, `api/auth.ts`, `hooks/useSigninNotice.ts` | The role badge, the theme, Settings for operators, "Sign in with GitHub" (a link to `/v1/auth/github/login`) or "Sign-in not configured on this server", and Sign out (`POST /v1/auth/logout`). `useSigninNotice` turns the callback's `?signin=denied&reason=…` or `?signin=error` into a notice and cleans the URL |

Server state goes through TanStack Query: the session (`api/hooks/useSession.ts`, `SESSION_QUERY_KEY`), health, capabilities (`['capabilities']`), the files list (`['files']`) and jobs (`['job', id]`). Every request goes to `API_BASE_URL` (`api/client.ts`), which is empty unless `VITE_API_URL` is set, so the UI's own origin; errors are `ApiError(message, status, body)`.

## Adding a result card kind

A card kind maps an analysis output to what the Workspace, Result detail and the docked map show. Example: a new algorithm whose output type `WetnessMap` has a raster `wetness`, its bare map `wetness_map_bare` and `statistics`.

1. **Backend first.** Save the raster with a style: `artifact_files.save(result, {"wetness": array}, {"wetness_map": png}, crs=…, transform=…, styles={"wetness": style}, bare={"wetness_map": "wetness"})` writes the `DT4LC_*` tags into its GeoTIFF and a `wetness_map_bare` PNG whose `stats.style` is that style (see [Job artifacts](ARCHITECTURE.md#bare-maps-and-style-tags)); then tiles and legends need no frontend colour table. If it has classes, return `class_statistics` with each class's `colour`. Add the output's registry id to `_KINDS` in `server/library.py`, so the job's `card.kind` (the Library's Type) is right, and to `_SINGLE_TITLES` or `_title` there for its card title.
2. **The model.** In `components/chat/result/resultModel.ts`, reuse a kind when the card looks the same: an index-like raster can go through `indexCard` by adding its output type to `INDEX_ARTIFACTS`. For a new look, add a member to the `ResultCardModel` union (`kind`, `views`, and what it shows), a builder that reads the artifact with `asRef` and `styleOf`, and a line in `resultCardOf`, before the generic `imageCard` fallback. `resultLayerOf` puts a view on the map.
3. **The view.** Add the `case` to the `switch (model.kind)` in `ResultCard.tsx`, using the token classes. If Result detail should show more than the card (statistics, a table), extend `components/detail/resultDetailModel.ts`.
4. **Tests.** Save a real `GET /v1/jobs/{id}` record from an isolated run in `src/test/fixtures/jobs/<name>.json` (as the existing fixtures are), then test the model in `resultModel.test.ts` and the render in `ResultCard.test.tsx`. Add a Playwright check of the card only if it is a new screen state.

## Browser E2E harness

`cognitive_ui/tests/e2e/` runs Playwright flows (`playwright-core` with the cached Chromium headless shell, driven by `node --test`) against the built `dist/` and a real API whose state lives in a temp dir. The full reference is [tests/e2e/README.md](../cognitive_ui/tests/e2e/README.md); the essentials:

- **Groups.** One API process serves a run, so flows that need another role, LLM mode or server state run in their own group, e.g. `e2e:workspace:guest`, `e2e:workspace:member` (`E2E_DEV_ROLE=member`), `e2e:capabilities:demo` (`E2E_SERVER=demo`), `e2e:run3:quota` (`E2E_GUEST_LIMIT=2`). The umbrella scripts are `e2e:workspace`, `e2e:run2`, `e2e:run3` (the five user flows: upload → analyze → result, map draw → send to chat → analysis, a follow-up turn, the guest quota, sign-in), `e2e:signin`, `e2e:settings`, `e2e:responsive`, `e2e:dark` and `e2e:a11y`. Screenshots and `api.log` go to `tests/e2e/out/` (git-ignored).
- **Fakes**, so no flow needs a key or the network: a fake Ollama (`fake-ollama.mjs`, with `E2E_LLM=fake`) answers from regex rules that a flow sets with `setLLMRules()` and logs what the backend sent (`llmRequests()`); a fake Hugging Face hub (`fake-hf.mjs`, always) serves zero-filled weights slowly enough to show download progress; a fake GitHub (`fake-github.mjs`, with `E2E_SIGNIN=fake`) has the scripted users `member`, `operator`, `outsider`, `unverified` and `denied`, picked with `signInAs()`. Earth Engine is never configured, so region flows end in `EE_NOT_CONFIGURED`.
- **Layout and accessibility.** `layout.mjs` exports `auditLayout(page, {scrollers, clippers, overlays, scope})`: it fails a screen whose document is wider than the viewport, whose elements leave an edge, or whose visible texts overlap; the responsive and dark groups run it on every screen. The `a11y` flow runs axe (`@axe-core/playwright`, WCAG 2.0/2.1 A and AA) on the seven screens in both themes and fails on any serious or critical violation; nothing is excluded. `screens.mjs` holds the seeding steps these groups share.
- **Writing a flow.** Add `flows/<name>.e2e.mjs`, start with `setup()` from `harness.mjs`, take screenshots with `outPath()`, assert `trackPageErrors(page)` is empty, and add the flow to its group's script in `package.json`. A flow that needs the fake LLM or another role skips itself otherwise.
- **Linux without root:** `tests/e2e/setup-libs.sh` fetches the browser's missing shared libraries into `tests/e2e/.libroot/`.

## Git workflow

1. Fork and branch from **`dev`** (not `master`).
2. Make focused changes; run `task test` and `task lint` (or `uv run ruff check .`).
3. Enable pre-commit hooks before pushing.
4. Open a PR to **`dev`** with description, test notes, and linked issue if any.
5. Address review feedback; maintainers merge after CI passes.

Releases merge `dev` into `master` and are tagged `vX.Y.Z`; see "Branches and Releases" in [CONTRIBUTING.md](../CONTRIBUTING.md) and the [pull request template](../.github/pull_request_template.md).
