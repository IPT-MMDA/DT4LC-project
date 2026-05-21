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
| **LLM routing** | `LLM_ENABLE_*`, `LLM_PROVIDER_ORDER`, `LLM_STRATEGY` | Enable/disable providers; order is fallback chain (`fallback` strategy). Commercial providers (`anthropic`, `mistral`) need both `LLM_ENABLE_*=true`, a key, and an entry in `LLM_PROVIDER_ORDER`. |
| **API keys** | `GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY` | Optional for local dev; without keys the router uses the next enabled provider (often Ollama). |
| **Models** | `GEMINI_MODELS`, `GROQ_MODEL`, `ANTHROPIC_MODELS`, `MISTRAL_MODELS` | Comma-separated lists rotate per provider (except Groq, single model). |
| **Ollama** | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Default `http://localhost:11434`. In Docker with `--profile ollama`, compose sets `OLLAMA_BASE_URL=http://ollama:11434`. |
| **Apertus (GPU)** | `APERTUS_ENABLED`, `APERTUS_MODEL`, `APERTUS_DEVICE`, `APERTUS_DTYPE` | Off by default; needs GPU + `LLM_ENABLE_APERTUS=true`. |
| **Earth Engine** | `GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_KEY` | Only for Sentinel-2 / GEE data fetch routes. |
| **Jobs** | `JOB_STORAGE`, `DT4LC_JOBS_DB` | `sqlite` (default) or `memory` (lost on restart). DB path defaults to `resources/.cache/jobs.db`. |
| **Ports (Docker)** | `BACKEND_PORT`, `FRONTEND_PORT` | Host port mapping in `docker-compose.yml`; local `task run:api:dev` uses uvicorn on 8000. |

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
| `VITE_API_URL` | `http://localhost:8000` | Axios client + upload/file hooks (`src/api/client.ts`) |

`GeoTIFFLayer.tsx` uses `VITE_API_BASE_URL` (same default); prefer setting **both** to the same backend URL if you customize.

**Backend:**

```bash
task venv && task install          # or: uv sync --extra dev --extra server
task run:api:dev                   # http://localhost:8000
```

**Frontend:**

```bash
cd cognitive_ui && npm install && npm run dev   # http://localhost:5173
```

**Docker (full stack):** `docker compose up -d` → UI at http://localhost, API at :8000.

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

Registry entrypoints use module paths like `dta.dti.algorithms.ndvi`, not file paths.

## Running tests

```bash
task test              # all tests
task test:fast         # skip slow
pytest tests/ -v       # direct
pytest -m "not llm" tests/ -v   # skip LLM-dependent tests
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

For non–spectral-index logic, implement `run()` returning a dict with results/statistics (see `dta/dti/algorithms/statistics.py`).

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
      formula: ndvi          # or custom; see spectral_index
      required_bands: [red, nir]
      colormap: ndvi
      vmin: -0.2
      vmax: 0.8
```

Triggers drive intent classification and context-agent hints without editing COE Python.

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

Run: `pytest tests/test_algorithms.py -v` (and `tests/test_registry.py` if you added types).

## COE pipeline walkthrough: "Calculate NDVI"

End-to-end path when a user uploads a GeoTIFF and asks **"Calculate NDVI"**:

```text
ChatRequest
    → orchestrator.orchestrate()
        → intent_classifier.classify_intent()     # PIPELINE (attachment + "ndvi" keyword)
        → context_agent.analyze()                 # desired_outputs: ["NDVIMap"]
        → planner.plan()                          # input/file → algorithms/ndvi
        → plan_validator.validate()
    → executor.execute(plan)                      # runs each PlanStep
```

| Step | Module | What happens for NDVI |
|------|--------|------------------------|
| 1 | `orchestrator.py` | Loads registry; classifies intent; builds validated plan |
| 2 | `intent_classifier.py` | `registry.yaml` `triggers` on `algorithms/ndvi` match `"calculate ndvi"` → `PIPELINE` (with attachment) or `CONVERSATION` (asks for file if missing) |
| 3 | `context_agent.py` | LLM extracts `desired_outputs: ["NDVIMap"]`; domain hints rendered from registry |
| 4 | `planner.py` | Template plan: `input/file` → `algorithms/ndvi` (+ postprocess if configured) |
| 5 | `executor.py` | Binds `RasterPath` from attachment; imports `dta.dti.algorithms.ndvi.run` |

**API wiring:** `server/routes/chat.py` calls `orchestrate()` then `executor.execute()` on the validated plan. Async jobs use the same flow in `server/jobs.py`.

**Verify:** `pytest tests/test_orchestrator.py tests/test_intent_classifier.py -v`

## Git workflow

1. Fork and branch from **`dev`** (not `master`).
2. Make focused changes; run `task test` and `task lint` (or `uv run ruff check .`).
3. Enable pre-commit hooks before pushing.
4. Open a PR to **`dev`** with description, test notes, and linked issue if any.
5. Address review feedback; maintainers merge after CI passes.

See [CONTRIBUTING.md](../CONTRIBUTING.md) and the [pull request template](../.github/pull_request_template.md).
