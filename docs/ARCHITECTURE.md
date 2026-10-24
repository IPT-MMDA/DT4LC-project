# DT4LC Architecture

System design and component overview for the Digital Twin for Land Cover project.

## Overview

DT4LC is a cognitive digital twin framework for geospatial analysis, combining:

- **LLM-powered orchestration** for natural language interfaces
- **Pipeline execution** for data processing workflows
- **Multi-provider LLM support** with automatic fallback

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                       http://localhost:5173                      │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ /v1/* (nginx proxy)
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│                    http://localhost:8000                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Context Orchestration Engine (COE)                          ││
│  │  • Intent Classifier (routes pipeline vs conversation)      ││
│  │  • Context Agent (understands user intent)                  ││
│  │  • Planner (builds execution pipeline)                      ││
│  │  • Plan Validator (validates plan)                          ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Digital Twin Instance (DTI)                                 ││
│  │  • Pipeline Executor                                        ││
│  │  • Algorithm Registry (NDVI, Statistics, Change Detection)  ││
│  │  • Model Registry (Prithvi, Delineate-Anything)             ││
│  │  • Post-Processing (Visualization, Insights)                ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Job Queue                                                   ││
│  │  • Async job processing                                     ││
│  │  • Progress tracking                                        ││
│  │  • Result caching                                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Gemini API        │  │   Groq API          │  │   Ollama (Local)    │
│   (Cloud LLM)       │  │   (Cloud LLM)       │  │   (Local LLM)       │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

## Component Details

### Context Orchestration Engine (COE)

Located in `dta/dti/coe/`

#### Intent Classifier (`intent_classifier.py`)

Determines if a request needs pipeline execution or a conversational response.

- **PIPELINE**: Requests that need data processing (NDVI, change detection)
- **CONVERSATION**: Questions, guidance, or explanations

Uses pattern matching for clear action requests, falls back to LLM for ambiguous cases. If that call fails or returns no valid JSON, the request is treated as PIPELINE.

#### Context Agent (`context_agent.py`)

Extracts structured understanding from user requests:

- Goal identification
- Required inputs/outputs
- Keywords and hints

If no provider answers, or every reply fails the context check (for example, not valid JSON of the `ContextUnderstanding` shape, types outside the registry, or keywords that are not strings; extra keys are ignored), the context is built deterministically from the registry triggers (`find_matching_item` in `triggers.py`) and the number of attachments: two or more mean before/after change detection. A WARNING in the log names the reason.

#### Planner (`planner.py`)

Generates execution plans using:

- **Template planner**: Fast path for common patterns
- **LLM planner**: Smart path for complex requests
- Hybrid mode with confidence-based routing

The LLM planner runs only when `estimate_plan_confidence()` is below 0.7. If its call fails, or its answer is not valid JSON or names a component missing from the registry, the template planner is used.

#### Plan Validator (`plan_validator.py`)

Checks estimated resources (`PlanValidator.check_resources`: at most 50 steps, 600 s and 4096 MB), that every step's inputs come from earlier steps or binds, and that Python runners have an entrypoint. If a plan fails, the orchestrator builds the template plan once and uses it only if it differs from the rejected plan, passes the same checks and contains an algorithm or model step; otherwise it returns the original error and nothing runs.

### LLM Router (`dta/dti/coe/llm/`)

Multi-provider LLM support with automatic fallback:

| Provider | Type | Cost | Configuration |
|----------|------|------|---------------|
| Gemini | Cloud | Pay per token | `GEMINI_API_KEY` |
| Groq | Cloud | Pay per token | `GROQ_API_KEY` |
| Ollama | Local | Free | `OLLAMA_BASE_URL` |

**Fallback Strategy:**

```
Request → Gemini (if available)
            ↓ (on failure)
        → Groq (if available)
            ↓ (on failure)
        → Ollama (if available)
            ↓ (on failure)
        → Error
```

The router moves on when a provider is unavailable or its call raises. For the context-extraction call it also moves on when the reply fails the context check (see Context Agent above). Other calls are not retried with the next provider when the answer is unusable: the intent classifier then treats the request as PIPELINE and the planner uses the template plan; result interpretation has no such check. When every provider fails, the router raises "All LLM providers failed: …" with each provider's reason (or "No available LLM providers" when none is configured), and each stage falls back deterministically: intent to PIPELINE, context to the registry triggers, planning to the template plan and interpretation to a "summarization failed" notice with the error.

The order comes from `LLM_PROVIDER_ORDER` (default `gemini,groq,ollama`; `gemini,groq` under Docker Compose). For example, `LLM_PROVIDER_ORDER=gemini,anthropic,mistral`, with the models set in `GEMINI_MODELS`, `ANTHROPIC_MODELS` and `MISTRAL_MODELS`, gives the Gemini → Anthropic → Mistral chain described in Chernyatevich et al., "Hybrid-Deterministic Fallback for Dependable LLM Provider Composition" (DESSERT 2026, in press).

### Digital Twin Instance (DTI)

Located in `dta/dti/`

#### Pipeline Executor (`executor.py`)

Executes plans step-by-step:

1. Load data via input components
2. Process via algorithms/models
3. Post-process for visualization
4. Return artifacts

#### Algorithm Registry (`algorithms/`)

- **NDVI** (`ndvi.py`): Vegetation index calculation
- **Statistics** (`statistics.py`): Raster statistics
- **Change Detection** (`change_detection.py`): Before/after comparison

#### Model Registry (`models/`)

- **Prithvi** (`prithvi.py`): NASA/IBM foundation model for geospatial features
- **Delineate-Anything**: Field boundary detection

#### Post-Processing (`post_processing/`)

- **Visualization** (`visualization.py`): Map rendering, charts
- **Insights** (`insights.py`): LLM-powered analysis summaries

### Job Queue (`server/jobs.py`)

Async job processing:

- An asyncio queue (at most 100 waiting jobs) and a pool of 3 worker threads
- Job states: pending → (proposal →) running → completed / failed / cancelled
- Jobs, with their event logs, cards and owners, are stored in SQLite (`JOB_STORAGE=sqlite`, the default, at `DT4LC_JOBS_DB`) or in memory (`JOB_STORAGE=memory`, lost on restart); a job that was running when the server stopped fails with `INTERRUPTED` at the next start
- Each job's events are kept on the job and published to the open event streams (see [Job event stream and proposals](#job-event-stream-and-proposals))
- Cleanup every `GUEST_CLEANUP_MINUTES` (default 60): guest jobs older than `GUEST_SESSION_HOURS`, with their output directories, and guest proposals never confirmed. Members' and operators' jobs are kept

### Server (`server/app.py`)

FastAPI application with:

- The REST routes in `server/routes/` (health, session, auth, chat, jobs, files, tiles, settings, gee) and `server/model_routes.py`, documented in [API.md](API.md) and the generated reference at `/docs`
- The session middleware (`server/sessions.py`), which runs before routing on every request
- CORS for `CORS_ORIGINS`, without credentials, so the UI and the API share one origin (nginx, or the Vite dev proxy)
- The periodic cleanup above, started with the app

## Sessions, roles and owners

`server/sessions.py` puts a `Session` on every request: a signed `dt4lc_session` cookie (HMAC-SHA256 with `SESSION_SECRET`) holding the session id, the role, the issue and expiry times and, once signed in, the GitHub user. A request without a valid cookie gets a new guest session.

| Role | How | What it may do |
|------|-----|----------------|
| guest | Anyone without a sign-in | `GUEST_ANALYSES_PER_HOUR` analyses per rolling hour, counted per session and per client IP; jobs and uploads removed after `GUEST_SESSION_HOURS`; no model downloads by default |
| member | Signed in with an `ALLOWED_EMAILS` address or one in `ALLOWED_DOMAINS` | No analysis limit; jobs and uploads kept; model downloads on demand by default |
| operator | Signed in with an `OPERATOR_EMAILS` address | As a member, plus every owner's jobs and uploads, Settings, model deletion |

`DEV_ROLE` overrides the role of every request for development. The allowlists are read again on every request (they can change in Settings), so a changed role applies at once, and a signed-in user who is on no list any more gets a new guest session.

**Sign-in** (`server/routes/auth.py`, `server/github_oauth.py`) is GitHub OAuth: login redirects to GitHub with a `state` that a signed cookie also holds; the callback checks it, exchanges the code, reads the primary verified email and decides the role. An allowed user gets a new session cookie that carries the user; anyone else lands back on `/app?signin=denied&reason=…` as a guest. Logout issues a fresh guest session. The flow and its redirects are in [API.md](API.md#sign-in).

**Owners.** Jobs and uploads record `owner` = `Session.owner`: the session id for a guest, and `github:<numeric id>` once signed in, which is stable across logins and devices. Every route that takes a job id or a file path checks it: `Session.can_see(owner)` is true for the owner and for operators, and anything else answers 404, as for an unknown id. Rows from before v0.6.0 have no owner and are visible to operators only. Uploads keep their owner in a sidecar next to the file (`abc_x.tif` → `abc_x.owner.json`).

**Adoption.** When a guest signs in, its jobs and uploads get the user id as owner and the new role (`JobQueue.adopt`, `uploads.adopt_uploads`), so the guest cleanup no longer removes them and the member sees them from any browser.

**Client IP.** The per-IP quota counts the ASGI client address. uvicorn runs with `--proxy-headers`, so behind a proxy whose address is in `FORWARDED_ALLOW_IPS` that is the address in `X-Forwarded-For`; the quota counts live in memory and reset on restart.

## Capabilities document

`GET /v1/capabilities` (`server/capabilities.py`) is the Capabilities page, computed per request for the caller: six tiers (Ask, Composed, Built-in, Models, Data sources, Coming), a summary line and the offline banner. The built-in and coming items come from the registry entries that carry a `ui:` block (title, formula, inputs, outputs, time, example prompt); the Ask and Composed copy lives in the module. The statuses come from the server's state:

- the assistant: the cached status that `GET /v1/health` reports (30 s), the only part that may touch the network;
- models: file checks of the model cache (`model_states()`), combined with the caller's role and `MODEL_DOWNLOAD_ON_DEMAND` (installed, downloads on first use, not available for guests, not installed); a built-in analysis that needs a model takes the model's status;
- data sources: the bundled sample's files, and an Earth Engine configuration check (`GEE_PROJECT_ID` and an existing key file);
- `DT4LC_DEPLOYMENT`, which adds "public demo" to the summary line, and the session's role and quota.

## Job event stream and proposals

A job's progress is a log of events, each `{seq, type, data}` with `seq` from 1 (`server/job_events.py`). `JobQueue` appends an event to the job, saves it and publishes it to every subscriber of that job (an asyncio queue per open stream). `GET /v1/jobs/{id}/events` subscribes, sends the stored events after `?after` or `Last-Event-ID` and then the live tail, and closes after `done`, `failed` or `cancelled`. Because the log is stored, a reload or a reconnect replays it, and a finished job replays its whole history. `POST /v1/chat` streams its markdown replies in the same SSE frame format.

```text
POST /v1/jobs ─► intent (template planner, no LLM) ─► planning started
                                                        │
                    single built-in analysis ◄──────────┴──────────► composed plan
                               │                                        │
                    planning ready ─► step_* ─► done | failed     proposal {text, plan}, status "proposal"
                                                                        │ POST /v1/jobs/{id}/confirm
                                                                        ▼
                                                planning ready (composed) ─► step_* ─► done | failed
```

- **Proposals.** When planning composes a plan that is not a single built-in analysis (band math for a named index such as NDMI, or a new combination of steps), the job stops at `proposal` and its guest-quota count is released. Confirming counts it again and continues the same job and log. An unconfirmed guest proposal expires with the guest's jobs (`cancelled {reason: "expired"}`); a deleted one ends with `cancelled {reason: "deleted"}`.
- **Refusals at planning** (a band the file lacks, a region without Earth Engine, a model the role may not download) end with `failed` and no `ready` row; `failed` carries a stable `code` and 0–3 `fixes` that the UI shows as buttons.
- **Cancel** marks the job `cancelled` at once; a running job stops at the next step boundary and logs nothing after `cancelled`.

## Settings overrides

Operators change some settings at runtime through `PUT /v1/settings` (`server/settings.py`). The values are overrides of environment variables, kept in a JSON file at `DT4LC_SETTINGS_PATH` (default `<DT4LC_CACHE_PATH>/settings.json`), never in `.env` (`dta/config/overrides.py`):

- Readers of the overridable variables call `overrides.getenv(name, default)` instead of `os.environ.get`: an override wins over the environment, which wins over the default. These readers are the LLM router config (`LLM_PROVIDER_ORDER`, `LLM_ENABLE_*`, the provider model variables), the roles (`ALLOWED_EMAILS`, `ALLOWED_DOMAINS`, `OPERATOR_EMAILS`), the guest limits and `MODEL_DOWNLOAD_ON_DEMAND`. Nothing else is overridable, and no secret is.
- The file is re-read when its modification time or size changes, so a write, a restart or a hand edit applies at once; a write goes to a temporary file in the same directory and is renamed over the old one.
- After an assistant change, the shared LLM router is rebuilt and the cached assistant status dropped.
- `GET /v1/settings` reports, for each editable field, whether its value comes from an `override`, the `env` or the `default`. `null` in a `PUT` drops the override.

## Demo deployment

The public demo is the Docker stack plus `docker-compose.demo.yml` (profile `demo`); the runbook is [DEPLOY_DEMO.md](DEPLOY_DEMO.md).

```text
browser ─HTTPS─► TLS proxy on the host (Caddy or nginx; not part of the stack)
                   │ http://127.0.0.1:<FRONTEND_PORT>
                   ▼
                 frontend: nginx with deploy/nginx-demo.conf.template
                   • the UI, and the only way in to the API
                   • client IP: X-Forwarded-For trusted only from DEMO_SUBNET, then replaced
                   • per-IP limit_req → 429 RATE_LIMITED
                   │ http://backend:8000 (demo network DEMO_SUBNET, default 172.30.60.0/24)
                   ▼
                 backend: no published port
                   • uvicorn --proxy-headers, FORWARDED_ALLOW_IPS=DEMO_SUBNET
                   • DT4LC_DEPLOYMENT=demo, MODEL_DOWNLOAD_ON_DEMAND=operators
                   • guest quota and cleanup as above; sign-in from .env
                   ▲
                 models-init (one-shot, before the backend): installs the model weights
                 into the models_data volume (DEMO_SKIP_MODEL_DOWNLOAD=1 skips it)
```

## Job artifacts

Since v0.5.0 a job result never contains pixel data. Rasters are written as
GeoTIFF and visualizations as PNG under the job's output directory (model
steps: next to the model's own output files); the result carries a small
reference per file. This keeps `GET /v1/jobs/{id}` under 1 MB
for a 1000×1000 raster, `GET /v1/jobs?limit=20` under 200 KB, and the SQLite
`jobs` table free of arrays and base64 (#65, #67 bugs 2 and 3).

### Artifact reference

```json
{
  "kind": "geotiff",
  "path": "/app/resources/.cache/outputs/<job_id>/1_NDVIMap_ndvi.tif",
  "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F<job_id>%2F1_NDVIMap_ndvi.tif&name=ndvi_2023-06-01.tif",
  "stats": {"min": -1.0, "max": 1.0, "mean": 0.5451, "valid_pixels": 589824},
  "size_bytes": 2100449,
  "download_name": "ndvi_2023-06-01.tif"
}
```

(NDVI of the bundled sample `resources/kahovka_data/hlsl_20230601.tif`.)

| Field | Meaning |
|-------|---------|
| `kind` | `"geotiff"` (single-band float32 raster, nodata NaN, CRS and transform of the input) or `"png"` (rendered image) |
| `path` | Absolute server path. Used by later pipeline steps that need the pixels |
| `url` | Server-relative download URL: `/v1/download?path=` + the URL-encoded `path`, and `&name=` + `download_name` once the job has finished. Clients resolve it against the API base URL, like any other API call |
| `stats` | `geotiff`: `min`, `max`, `mean`, `valid_pixels` over finite, non-nodata pixels; with no valid pixel `min`, `max` and `mean` are `null` and `valid_pixels` is 0. `png`: `{}`, or for a bare map `{width, height, style}` (below) |
| `size_bytes` | The file's size when the reference was made, right after writing it |
| `download_name` | Added when the job completes (`server/library.py`, `name_downloads`): a readable name from the analysis and the analysed files' dates, unique within the job, e.g. `water-change_ndwi_2023-06-01_2023-06-09.tif`, its per-date index `ndwi_2023-06-01.tif`, `ndmi-change_2023-06-01_2023-06-09.tif`, `land-cover_2023-06-09.tif`; a bare map adds `_map`. The stored file keeps its name. Jobs finished before v0.6.0 have no `download_name` and no `size_bytes`: clients fall back to the base name of `path` |

`dta/dti/artifact_files.py` writes these files and builds the references.

### Bare maps and style tags

Besides the rendered PNG figure of each raster, the index, change, band-math, land-cover and snow algorithms write a map-only PNG under the same key with `_bare` appended (`ndvi_map_bare`, `change_map_bare`, `ndwi_before_bare`, `classification_map_bare`, …): the raster's pixels in its colours, nodata transparent. Result cards, the Result detail and the Library thumbnail (`card.thumbnail_url`) show these. A bare map's `stats` is `{width, height, style}`, and `style` is how the raster is coloured:

- `{"kind": "change" | "index", "range": [lo, hi], "ramp": ["#rrggbb", …]}`: values clipped to the range and interpolated between evenly spaced colour stops, e.g. NDVI `range [-0.2, 0.8]` and the NDWI change `range [-0.5, 0.5]` with `ramp ["#8c4a2f", "#c4cad6", "#2d5f9e"]` (lost · no change · gained);
- `{"kind": "class", "classes": {"<value>": "#rrggbb", …}}`: one colour per class value.

The GeoTIFF of a styled raster carries the same style in its tags, so `GET /v1/tiles` draws it in the colours of its bare map without reading the job (`artifact_files.read_style` and `colourize`):

| Tag | Value |
|-----|-------|
| `DT4LC_KIND` | `change`, `index` or `class` |
| `DT4LC_VMIN`, `DT4LC_VMAX` | The range (change and index) |
| `DT4LC_RAMP` | The colour stops, comma-separated (change and index) |
| `DT4LC_CLASSES` | `value:#rrggbb` pairs, comma-separated (class) |

A raster without these tags is drawn by the tile route's fallbacks: true colour for a multi-band raster with red, green and blue bands, else the older colour maps ([API.md](API.md#get-v1tileszxy)).

### Class statistics

Change detection, band-math change, land cover and snow results have `class_statistics`, the class table that legends, the Result detail and `GET /v1/jobs/{id}/statistics.csv` show, keyed by class name: `{id, name, colour, pixels, percent, area_km2}`, plus `mean_change` for change. `colour` is the class's colour in the bare map, so a legend needs no colour table of its own. For change the classes are gained / lost / no change of the index's cover class (Water for NDWI, Vegetation for NDVI), Wetter / Drier / No change for band-math NDMI, else Increase / Decrease / Stable. The NDWI example on the bundled crop is in [API.md](API.md#get-v1jobsjob_id).

### Where files go

- Job output directory: `OUTPUTS_PATH/<job_id>/`, with
  `OUTPUTS_PATH = resources/.cache/outputs` (`DT4LC_OUTPUTS_PATH`; inside the `./resources` volume
  in Docker, so files survive restarts). Executor runs outside the job queue
  (`POST /v1/execute`, direct `PipelineExecutor.execute()` calls) write to
  `OUTPUTS_PATH/run-<uuid hex>/`, one directory per run. `POST /v1/chat` runs
  no pipeline itself: it submits analyses to the job queue (see
  [API.md](API.md#post-v1chat)).
- File name: `<step index>_<output type>_<key>.<tif|png>`, where the step
  index is the step's 0-based position in the plan and the output type is the
  step's first registry output. The standard NDVI plan (`input/file`,
  `algorithms/ndvi`, `post-processing/agent-analysis`) gives
  `1_NDVIMap_ndvi.tif` and `1_NDVIMap_ndvi_map.png`. A step without outputs
  uses the bare index as prefix. These stored names never change; a download
  is named by the reference's `download_name` instead.
- An algorithm called directly, without the executor, writes to a new
  `OUTPUTS_PATH/run-<uuid hex>/` on each call, with files named
  `<key>.<tif|png>`.
- Model steps such as Prithvi write to `TEMP_PATH/<model id>_outputs/<run id>/`
  (`TEMP_PATH = resources/.cache/tmp`, `DT4LC_TEMP_PATH`), and the PNG of each output TIFF is
  written next to it. Delineate-Anything writes its GeoPackage and the
  `field_boundaries` PNG to `TEMP_PATH/delineate_outputs/` instead.
- Upload previews: `<uploads directory>/<id>_<file name stem>.png`, next to the
  upload.
- `GET /v1/download` and `GET /v1/tiles` serve only files under `OUTPUTS_PATH`,
  the uploads directory, all of `TEMP_PATH` (model outputs and
  `delineate_outputs/`), the Earth Engine exports (`<cache>/gee_exports`) and
  `<system temp>/dt4lc_delineate`. The path is resolved before the check;
  anything else is 403. Files under `OUTPUTS_PATH/<job_id>/` and uploads are
  then served to their owner and to operators only (404 for anyone else).

### Result shape per step output

Step outputs stay where they are (`result.execution.artifacts.<OutputType>`,
e.g. `NDVIMap`, `ChangeMap`) with their scalar fields (`statistics`,
`metadata`, `index_type`, `path`, …) unchanged. Two things change inside
each one:

- Every array field is removed and replaced by an entry in a new `rasters`
  object of `geotiff` references.
- `visualizations` keeps its keys, but each value becomes a `png` reference
  instead of a base64 string.

| Algorithm | Removed array fields → `rasters` key | `visualizations` keys (now `png` refs) |
|-----------|--------------------------------------|----------------------------------------|
| `spectral_index.run`; NDVI, NDWI, EVI modules | `<index>_array` → `<index>` | `<index>_map` |
| NDSI (`ndsi.py`) | `ndsi_array` → `ndsi`, `snow_mask` → `snow_mask` (1 snow, 0 no snow, NaN where the input is invalid) | `ndsi_map` |
| `spectral_index.run_change`, `ndvi.ndvi_change` | `change_array` → `change` | none (no `visualizations` key) |
| `change_detection` | `change_array` → `change`, `index_before` → `index_before`, `index_after` → `index_after`; the NDVI aliases `ndvi_before` and `ndvi_after` are removed (`ndvi_statistics` stays) | `change_map`, `<index>_before`, `<index>_after` |
| `snow_classifier` | `snow_mask` (NaN where the input is invalid), `ndsi`, `ndvi`, `brightness` (same keys) | `snow_classification`, `criteria_analysis` |
| `lulc_classifier` | `classification` (same key; class 0 "No Data" is stored as NaN) | `classification_map`, `distribution_chart` |
| Model steps (`_collect_model_outputs`, e.g. Prithvi) | — (`output_files` stay paths) | per output file, PNG written next to the TIFF |
| Delineate-Anything | — (`output_path` stays the GeoPackage path) | `field_boundaries`, PNG next to the GeoPackage |

Without matplotlib the algorithms return an empty `visualizations`;
`artifact_files.save` keeps an image that is `None` as `null` instead of a
reference.

A pipeline step that consumes another step's pixels opens
`rasters.<key>.path` instead of reading an in-memory array. The contract test
`tests/test_job_artifacts.py` walks a result for each row of the table and
fails on any `numpy.ndarray`, base64-looking string over 1 KB, nested list of
numbers or list of more than 256 numbers.

### API changes

- `GET /v1/jobs` returns job summaries: the `GET /v1/jobs/{id}` record
  without `result`. Clients that need a result fetch the job by id.
- `GET /v1/jobs/{id}` returns the full record, with references as above. In
  the OpenAPI schema `JobRecord` extends `JobSummary`, so `result` is its last
  field. Jobs stored before v0.5.0 are not rewritten: their stored result
  still holds arrays (as nested lists) and base64 images.
- `POST /v1/upload` additionally returns `preview_url`, the download URL of a
  PNG preview written to the uploads directory next to the file
  (`<id>_<file name stem>.png`).
  `preview_png_base64` is still returned for compatibility but clients must
  not persist it or send it back in job submissions.

### What the frontend reads

- Result cards and the Result detail: `resultCardOf()` in `cognitive_ui/src/components/chat/result/resultModel.ts` reads `result.execution.artifacts` into one card model per artifact kind. Images are the `*_bare` refs' `url`s, legends come from their `stats.style` and from `class_statistics[*].colour`, and the stats from `statistics` (and `water`, `vegetation`, `class_statistics` for change). No colour table is mirrored in the frontend.
- Map layers: a result raster is drawn from `/v1/tiles/{z}/{x}/{y}?path=<rasters.<key>.path>`, an attachment from its upload path (true colour by default), and an upload's outline from `metadata.footprint_wgs84`.
- Downloads: the ref's `url`, saved as `downloadName(ref)`: the `download_name`, else the base name of `path` for older jobs. The statistics CSV is `GET /v1/jobs/{id}/statistics.csv`.
- Library rows: `GET /v1/jobs` (each with its `card`) and `GET /v1/files`, without loading any result.
- Attachment previews: `preview_url` from the upload response.
- Persisted chat state (`localStorage`) holds URLs only, never image bytes.
- Jobs stored before v0.5.0 hold base64 strings where references now are, and `GET /v1/jobs/{id}` returns them unchanged. `asRef()` treats such a string as no reference, so these jobs show without an image.

## Data Flow

### Pipeline Execution Flow

```text
1. User Request
   ↓
2. Intent Classification (pipeline vs conversation)
   ↓
3. Context Understanding (extract goal, inputs, outputs)
   ↓
4. Plan Generation (template or LLM planner)
   ↓
5. Plan Validation (plan validator)
   ↓
6. Job Queuing (async processing)
   ↓
7. Pipeline Execution
   ├── Load data (input/file)
   ├── Process (algorithms/models)
   └── Post-process (visualization/insights)
   ↓
8. Return Results
```

### Conversation Flow

```text
1. User Question
   ↓
2. Intent Classification → CONVERSATION
   ↓
3. LLM generates helpful response
   ↓
4. Return response (no pipeline execution)
```

## Directory Structure

```
dta/
├── registry.yaml                 # Component registry
├── config/                       # Configuration
├── dti/                          # Digital Twin Instance
│   ├── coe/                      # Context Orchestration Engine
│   │   ├── llm/                  # LLM providers (Gemini, Groq, Ollama)
│   │   ├── intent_classifier.py  # Routes pipeline vs conversation
│   │   ├── context_agent.py      # Understands user intent
│   │   ├── planner.py            # Builds execution plans
│   │   ├── plan_validator.py     # Validates plans
│   │   └── orchestrator.py       # Main orchestration flow
│   ├── algorithms/               # NDVI, Statistics, Change Detection
│   ├── models/                   # Prithvi, Delineate-Anything
│   ├── post_processing/          # Visualization, Insights
│   └── executor.py               # Pipeline execution

server/
├── app.py                        # FastAPI application
├── routes/                       # REST routes (auth, chat, files, jobs, settings, tiles, ...)
├── jobs.py                       # Job queue system
├── job_events.py                 # Event names, payloads, SSE frames, failure codes
├── sessions.py                   # Session cookie, roles, owners, guest quota
├── github_oauth.py               # GitHub sign-in
├── capabilities.py               # The capabilities document
├── library.py                    # Job cards, statistics CSV, download names
├── settings.py                   # Operator settings document and updates
└── schemas.py                    # Request/response models

cognitive_ui/                     # React application
├── src/
│   ├── pages/                    # Landing, Workspace, Map, Library, Result detail, Capabilities, Settings
│   ├── components/               # shell, chat, map, library, detail, landing, settings
│   ├── hooks/                    # React hooks (job events, sample, theme, layout, ...)
│   ├── store/                    # Zustand stores (chat, app)
│   └── api/                      # API client, SSE reader
└── tests/e2e/                    # Playwright flows and fakes

tests/                            # Test suite
├── test_orchestrator.py          # Orchestration tests
├── test_intent_classifier.py     # Intent classification tests
├── test_algorithms.py            # Algorithm tests
└── test_jobs.py                  # Job queue tests
```

## Configuration

### Environment Variables

Every variable the code reads is listed and commented in [`.env.example`](../.env.example). The LLM ones:

```bash
# LLM Providers
GEMINI_API_KEY=...          # Google Gemini API key
GROQ_API_KEY=...            # Groq API key
OLLAMA_BASE_URL=http://localhost:11434

# LLM Selection
LLM_ENABLE_GEMINI=true
LLM_ENABLE_GROQ=true
LLM_ENABLE_OLLAMA=true
LLM_PROVIDER_ORDER=gemini,groq,ollama
LLM_STRATEGY=fallback

# Model Selection
GEMINI_MODELS=gemini-3.5-flash-lite
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_MODEL=llama3.2
```

### Registry (`dta/registry.yaml`)

Components are registered in YAML format:

```yaml
instances:
  - id: algorithms/ndvi
    kind: algorithm
    keywords: [ndvi, vegetation, index]
    inputs: [RasterPath]
    outputs: [NDVIMap, Statistics]
    runner:
      type: python
      entrypoint: "dta/dti/algorithms/ndvi.py"
```

## Extending the System

### Adding New Algorithms

1. Create algorithm file:

```python
# dta/dti/algorithms/my_algorithm.py
def run(RasterPath: str) -> dict:
    """Process raster and return results."""
    return {"result": ...}
```

2. Register in `dta/registry.yaml`:

```yaml
- id: algorithms/my_algorithm
  kind: algorithm
  keywords: [my, algorithm]
  inputs: [RasterPath]
  outputs: [MyOutput]
  runner:
    type: python
    entrypoint: "dta/dti/algorithms/my_algorithm.py"
```

### Adding New LLM Providers

1. Create provider class implementing `BaseLLMProvider`:

```python
# dta/dti/coe/llm/my_provider.py
class MyProvider(BaseLLMProvider):
    def generate(self, messages, temperature=0.7, max_tokens=1024):
        # Implementation
        return LLMResponse(text=..., provider="my_provider")
```

2. Add to router configuration in `config.py`

## Performance Considerations

- **LLM Calls**: 3-5 seconds per call (cloud), 2-10 seconds (local)
- **NDVI Calculation**: <1 second for typical rasters
- **Job Queue**: 3 concurrent workers by default
- **Caching**: Results cached for 30 minutes
