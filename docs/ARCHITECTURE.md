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

The router moves on when a provider is unavailable or its call raises. For the context-extraction call it also moves on when the reply fails the context check (see Context Agent above). Other calls are not retried with the next provider when the answer is unusable: the intent classifier then treats the request as PIPELINE and the planner uses the template plan; result interpretation has no such check. When every provider fails, the router raises "All LLM providers failed: …" with each provider's reason, and each stage falls back deterministically: intent to PIPELINE, context to the registry triggers, planning to the template plan and interpretation to a fixed "summarization failed" summary.

The order comes from `LLM_PROVIDER_ORDER` (default `gemini,groq,ollama`). For example, `LLM_PROVIDER_ORDER=gemini,anthropic,mistral`, with the models set in `GEMINI_MODELS`, `ANTHROPIC_MODELS` and `MISTRAL_MODELS`, gives the Gemini → Anthropic → Mistral chain described in Chernyatevich et al., "Hybrid-Deterministic Fallback for Dependable LLM Provider Composition" (DESSERT 2026, in press).

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

- In-memory asyncio.Queue
- Background worker pool (3 workers default)
- Job states: pending → running → completed/failed/cancelled
- Progress tracking (0.0 to 1.0)
- Automatic cleanup (1 hour retention)

### Server (`server/app.py`)

FastAPI application with:

- 12 REST endpoints
- CORS enabled
- File upload support
- Job lifecycle management
- Swagger/ReDoc documentation

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
  "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F<job_id>%2F1_NDVIMap_ndvi.tif",
  "stats": {"min": -1.0, "max": 1.0, "mean": 0.5451, "valid_pixels": 589824}
}
```

(NDVI of the bundled sample `resources/kahovka_data/hlsl_20230601.tif`.)

| Field | Meaning |
|-------|---------|
| `kind` | `"geotiff"` (single-band float32 raster, nodata NaN, CRS and transform of the input) or `"png"` (rendered image) |
| `path` | Absolute server path. Used by later pipeline steps that need the pixels |
| `url` | Server-relative download URL: `/v1/download?path=` + the URL-encoded `path`. Clients resolve it against the API base URL, like any other API call |
| `stats` | `geotiff`: `min`, `max`, `mean`, `valid_pixels` over finite, non-nodata pixels; with no valid pixel `min`, `max` and `mean` are `null` and `valid_pixels` is 0. `png`: `{}` |

`dta/dti/artifact_files.py` writes these files and builds the references.

### Where files go

- Job output directory: `OUTPUTS_PATH/<job_id>/`, with
  `OUTPUTS_PATH = resources/.cache/outputs` (inside the `./resources` volume
  in Docker, so files survive restarts). Executor runs outside the job queue
  (`POST /v1/execute`, `POST /v1/chat`, direct `PipelineExecutor.execute()`
  calls) write to `OUTPUTS_PATH/run-<uuid hex>/`, one directory per run.
- File name: `<step index>_<output type>_<key>.<tif|png>`, where the step
  index is the step's 0-based position in the plan and the output type is the
  step's first registry output. The standard NDVI plan (`input/file`,
  `algorithms/ndvi`, `post-processing/agent-analysis`) gives
  `1_NDVIMap_ndvi.tif` and `1_NDVIMap_ndvi_map.png`. A step without outputs
  uses the bare index as prefix.
- An algorithm called directly, without the executor, writes to a new
  `OUTPUTS_PATH/run-<uuid hex>/` on each call, with files named
  `<key>.<tif|png>`.
- Model steps such as Prithvi write to `TEMP_PATH/<model id>_outputs/<run id>/`
  (`TEMP_PATH = resources/.cache/tmp`), and the PNG of each output TIFF is
  written next to it. Delineate-Anything writes its GeoPackage and the
  `field_boundaries` PNG to `TEMP_PATH/delineate_outputs/` instead.
- Upload previews: `<uploads directory>/<id>_<file name stem>.png`, next to the
  upload.
- `GET /v1/download` serves only files under `OUTPUTS_PATH`, the uploads
  directory, all of `TEMP_PATH` (model outputs and `delineate_outputs/`) and
  `<system temp>/dt4lc_delineate`. The path is resolved before the check;
  anything else is 403.

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

- Result images: `visualizations.<key>.url` (also used for the image download
  link). `JobResultData.visualizations[]` holds `{type, label, url}`; no
  `base64` field.
- Statistics: `statistics.*` as before.
- Attachment previews: `preview_url` from the upload response.
- Persisted chat state (`localStorage`) holds URLs only, never image bytes.
  Messages persisted by older versions have no URL and render without an
  image.
- Jobs stored before v0.5.0 hold base64 strings where references now are,
  and `GET /v1/jobs/{id}` returns them unchanged. The Job Details page
  (`JobResult.tsx`, not persisted) renders such a string as a PNG data URL;
  the chat card reads only `url` and shows these jobs without an image.

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
├── jobs.py                       # Job queue system
└── schemas.py                    # Request/response models

cognitive_ui/                     # React application
├── src/
│   ├── components/               # UI components
│   ├── hooks/                    # React hooks
│   ├── stores/                   # Zustand stores
│   └── api/                      # API client

tests/                            # Test suite
├── test_orchestrator.py          # Orchestration tests
├── test_intent_classifier.py     # Intent classification tests
├── test_algorithms.py            # Algorithm tests
└── test_jobs.py                  # Job queue tests
```

## Configuration

### Environment Variables

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
