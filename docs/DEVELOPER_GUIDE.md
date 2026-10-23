# Developer Guide

Setup, layout, tests, adding algorithms, COE flow, and git workflow for DT4LC.

## Environment setup

**Prerequisites:** Python 3.10+, [uv](https://github.com/astral-sh/uv), optional [Task](https://taskfile.dev/).

```bash
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project
cp .env.example .env
cp cognitive_ui/.env.example cognitive_ui/.env   # frontend API URL
```

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

**Minimal local COE (no Docker):** set at least one of `GEMINI_API_KEY` or `GROQ_API_KEY`, or run Ollama locally and set `LLM_PROVIDER_ORDER=ollama` with `LLM_ENABLE_OLLAMA=true`.

**Also used in code but not in `.env.example`:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins (`server/app.py`) |
| `DT4LC_HOST` / `DT4LC_PORT` | `0.0.0.0` / `8000` | `server/cli.py` |
| `DT4LC_RELOAD` | off | Hot reload for CLI server |
| `DT4LC_LOG_LEVEL` | `info` | CLI log level |
| `DT4LC_MODEL_CACHE` | `resources/.cache/models` | Model download cache (set in Docker) |
| `DT4LC_UPLOADS_PATH` | `resources/.cache/uploads` | Uploaded rasters |
| `DT4LC_JOBS_DB` | `resources/.cache/jobs.db` | SQLite job store path |

Full LLM options and examples are also in [README.md](../README.md#configuration).

### Frontend `cognitive_ui/.env`

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_API_URL` | `http://localhost:8000` | Axios client (`src/api/client.ts`), upload/file hooks (`src/api/hooks/useUpload.ts`) and GeoTIFF tiles (`GeoTIFFLayer.tsx`) |

It is the only frontend variable; the Docker image builds with `VITE_API_URL=""`, so the UI calls the API on its own origin through nginx.

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

## Git workflow

1. Fork and branch from **`dev`** (not `master`).
2. Make focused changes; run `task test` and `task lint` (or `uv run ruff check .`).
3. Enable pre-commit hooks before pushing.
4. Open a PR to **`dev`** with description, test notes, and linked issue if any.
5. Address review feedback; maintainers merge after CI passes.

Releases merge `dev` into `master` and are tagged `vX.Y.Z`; see "Branches and Releases" in [CONTRIBUTING.md](../CONTRIBUTING.md) and the [pull request template](../.github/pull_request_template.md).
