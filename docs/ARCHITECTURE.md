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

Uses pattern matching for clear action requests, falls back to LLM for ambiguous cases.

#### Context Agent (`context_agent.py`)

Extracts structured understanding from user requests:

- Goal identification
- Required inputs/outputs
- Keywords and hints

#### Planner (`planner.py`)

Generates execution plans using:

- **Template planner**: Fast path for common patterns
- **LLM planner**: Smart path for complex requests
- Hybrid mode with confidence-based routing

#### Plan Validator (`plan_validator.py`)

Validates generated plans against registry constraints.

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
GeoTIFF and visualizations as PNG under the job's output directory; the result
carries a small reference per file. This keeps `GET /v1/jobs/{id}` under 1 MB
for a 1000×1000 raster, `GET /v1/jobs?limit=20` under 200 KB, and the SQLite
`jobs` table free of arrays and base64 (#65, #67 bugs 2 and 3).

### Artifact reference

```json
{
  "kind": "geotiff",
  "path": "/app/resources/.cache/outputs/<job_id>/0_NDVIMap_ndvi.tif",
  "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F<job_id>%2F0_NDVIMap_ndvi.tif",
  "stats": {"min": -0.21, "max": 0.89, "mean": 0.41, "valid_pixels": 589824}
}
```

| Field | Meaning |
|-------|---------|
| `kind` | `"geotiff"` (single-band float32 raster, nodata NaN, CRS and transform of the input) or `"png"` (rendered image) |
| `path` | Absolute server path. Used by later pipeline steps that need the pixels |
| `url` | Server-relative download URL: `/v1/download?path=` + the URL-encoded `path`. Clients resolve it against the API base URL, like any other API call |
| `stats` | `geotiff`: `min`, `max`, `mean`, `valid_pixels` over finite, non-nodata pixels. `png`: `{}` |

### Where files go

- Job output directory: `OUTPUTS_PATH/<job_id>/`, with
  `OUTPUTS_PATH = resources/.cache/outputs` (inside the `./resources` volume
  in Docker, so files survive restarts). Runs outside the job queue (direct
  executor use, tests) write to `OUTPUTS_PATH/run-<uuid hex>/`.
- File name: `<step index>_<output type>_<key>.<tif|png>`, e.g.
  `0_NDVIMap_ndvi.tif`, `0_NDVIMap_ndvi_map.png`.
- `GET /v1/download` serves only files under `OUTPUTS_PATH`, the uploads
  directory, model output directories (`resources/.cache/tmp/<model>_outputs/`)
  and `<system temp>/dt4lc_delineate`. Anything else is 403.

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
| `spectral_index.run` (NDVI, NDWI, NDSI, EVI, …) | `<index>_array` → `<index>` | `<index>_map` |
| `spectral_index.run_change` | `change_array` → `change` | — |
| `change_detection` | `change_array` → `change`, `index_before` → `index_before`, `index_after` → `index_after` | `change_map`, `<index>_before`, `<index>_after` |
| `snow_classifier` | `snow_mask`, `ndsi`, `ndvi`, `brightness` (same keys) | `snow_classification`, `criteria_analysis` |
| `lulc_classifier` | `classification` (same key) | `classification_map`, `distribution_chart` |
| Model steps (`_collect_model_outputs`, e.g. Prithvi) | — (`output_files` stay paths) | per output file, PNG written next to the TIFF |

A pipeline step that consumes another step's pixels opens
`rasters.<key>.path` instead of reading an in-memory array. A contract test
walks every algorithm's result and fails on any `numpy.ndarray`, base64
string or nested list of numbers.

### API changes

- `GET /v1/jobs` returns job summaries: the `GET /v1/jobs/{id}` record
  without `result`. Clients that need a result fetch the job by id.
- `GET /v1/jobs/{id}` returns the full record, with references as above.
- `POST /v1/upload` additionally returns `preview_url`, the download URL of a
  PNG preview written to the uploads directory next to the file.
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
