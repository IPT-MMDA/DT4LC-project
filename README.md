# DT4LC - Digital Twin for Land Cover

A cognitive digital twin framework for land cover change detection and vegetation analysis using satellite imagery, machine learning, and LLM-powered orchestration.

It is aimed at remote sensing researchers, environmental scientists and disaster response teams who need to perform land cover analysis without building custom processing pipelines.

## Features

- **NDVI Analysis** - Calculate vegetation health indices from satellite imagery
- **Change Detection** - Compare before/after images to detect land cover changes
- **Field Boundary Detection** - Delineate-Anything model for agricultural field segmentation
- **LLM-Powered Planning** - Natural language interface with automatic pipeline generation
- **Intent Classification** - Smart routing between pipeline execution and conversational responses
- **Multi-Turn Conversations** - Follow-up requests using previous context and attachments
- **Multi-Provider LLM Support** - Gemini, Groq, OpenAI, Anthropic, Mistral, and local Ollama or Apertus with automatic fallback
- **Prithvi Model Integration** - NASA/IBM foundation model for geospatial features

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                    http://localhost (port 80)                    │
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
│  │  • Planner Agent (builds execution pipeline)                ││
│  │  • Decision Agent (validates plan)                          ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Digital Twin Instance (DTI)                                 ││
│  │  • Pipeline Executor                                        ││
│  │  • Algorithm Registry (NDVI, Statistics, Change Detection)  ││
│  │  • Model Registry (Prithvi, Delineate-Anything)             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Providers (fallback chain)                 │
│  Gemini (fast) → Groq (ultra-fast) → Ollama (local)             │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Docker (Recommended)

```bash
# 1. Clone and navigate to project
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project

# 2. Set up environment
cp .env.example .env
# Optional: Add GEMINI_API_KEY or GROQ_API_KEY for faster LLM responses

# 3. Start all services, including the local Ollama LLM (no API key needed)
docker compose --profile ollama up -d

# 4. Wait for the one-time llama3.2 pull and for the services to become healthy
docker compose logs ollama-pull    # ends with "Model pull complete!"
docker compose ps

# 5. Open application
open http://localhost
```

The UI is at <http://localhost>, and nginx also serves the API reference there: Swagger UI at <http://localhost/docs>, ReDoc at <http://localhost/redoc>. The backend alone listens on <http://localhost:8000> (Swagger UI at <http://localhost:8000/docs>). Under Docker Compose the backend reaches Ollama at `http://ollama:11434`; leave `OLLAMA_BASE_URL` unset in `.env` (bare-metal runs default to `http://localhost:11434`). With API keys in `.env`, `docker compose up -d` starts the stack without Ollama.

### Local Development

#### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Task](https://taskfile.dev/) (optional but recommended)

#### Backend

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using Task (recommended)
task venv           # Create virtual environment
task install        # Install all dependencies
task test           # Run tests
task run:api        # Start the API server on :8000

# Or manually with uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev,server]"
pytest tests/ -v
uvicorn server.app:app --reload --port 8000
```

#### Frontend

```bash
cd cognitive_ui

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Running Tests

### Without keys, network or models (what CI runs)

```bash
uv run pytest tests/ -m "not slow and not llm"
```

This needs no `.env`, API keys, network access or model weights. A plain `uv run pytest tests/` (or `task test`) also exits 0 without keys: `llm` tests skip ("no LLM provider available ..."), the Earth Engine test skips without `GEE_PROJECT_ID`, model tests skip when the weights are not downloaded, and the Delineate GPU test skips without the GDAL Python bindings. `tests/conftest.py` loads `.env`, so with keys there a plain run calls the live providers.

### Opt-in tests

| Marker | Command | Needs |
|--------|---------|-------|
| `llm` | `uv run pytest tests/ -m llm` | An available provider from `LLM_PROVIDER_ORDER`: its API key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY` or `GPT_SECRET`), or a running Ollama with `OLLAMA_MODEL` pulled |
| `network` | `uv run pytest tests/ -m network` | `GEE_PROJECT_ID` and Earth Engine credentials |
| `slow`, `gpu` | `uv run pytest tests/ -m "slow or gpu"` | Downloaded models (see [Model weights](#model-weights)) and an NVIDIA GPU; the Delineate test also needs GDAL (see [Field Boundary Detection](#field-boundary-detection-delineate-anything)) |

For Ollama, an untagged `OLLAMA_MODEL` such as `llama3.2` means `llama3.2:latest`; a tagged name such as `qwen2.5:7b` must match an installed tag exactly.

Frontend tests (vitest):

```bash
cd cognitive_ui && npm test
```

### Using Task

```bash
task test              # Run all tests (uv run pytest)
task test:fast         # Run tests excluding slow ones (-m "not slow")
task test:verbose      # Run with verbose output
task coverage          # Run with coverage report
```

### Test Categories

Tests are organized by component:

```bash
# Core components
pytest tests/test_registry.py -v      # Component registry
pytest tests/test_executor.py -v      # Pipeline executor
pytest tests/test_algorithms.py -v    # NDVI, Statistics, Change Detection

# LLM infrastructure
pytest tests/test_llm_providers.py -v # Gemini, Ollama, router
pytest tests/test_planner.py -v       # Template & LLM planners

# Infrastructure
pytest tests/test_cache.py -v         # Caching
pytest tests/test_metrics.py -v       # Metrics collection
pytest tests/test_validation.py -v    # Input/plan validation

# Application
pytest tests/test_orchestrator.py -v  # COE orchestration
pytest tests/test_jobs.py -v          # Async job queue
pytest tests/test_visualization.py -v # Visualization & insights

# Model integrations
pytest tests/test_models_prithvi.py -v    # Prithvi E2E
pytest tests/test_models_delineate.py -v  # Delineate-Anything E2E
```

### Test Coverage

```bash
task coverage
# Or manually:
pytest tests/ --cov=dta --cov=server --cov-report=html
open htmlcov/index.html
```

## Project Structure

```text
DT4LC-project/
├── Taskfile.yml                # Task runner commands
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Backend container
├── .env.example                # Environment template
│
├── dta/                        # Digital Twin Application
│   ├── dti/                    # Digital Twin Instance
│   │   ├── coe/                # Context Orchestration Engine
│   │   │   ├── llm/            # LLM providers (Gemini, Groq, Ollama)
│   │   │   ├── intent_classifier.py  # Routes pipeline vs conversation
│   │   │   ├── context_agent.py
│   │   │   ├── planner.py
│   │   │   └── orchestrator.py
│   │   ├── algorithms/         # NDVI, Statistics, Change Detection
│   │   ├── models/             # Prithvi, Delineate-Anything
│   │   └── executor.py         # Pipeline execution
│   └── registry.yaml           # Component registry
│
├── server/                     # FastAPI server
│   ├── app.py                  # Main application
│   └── schemas.py              # API schemas
│
├── cognitive_ui/               # React + TypeScript UI
│   ├── Dockerfile              # Frontend container
│   ├── nginx.conf              # Production nginx config
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── api/                # API client
│   │   └── store/              # Zustand state
│   └── package.json
│
├── tests/                      # Test suite (organized by component)
│   ├── test_registry.py        # Component registry tests
│   ├── test_executor.py        # Pipeline executor tests
│   ├── test_algorithms.py      # Algorithm tests
│   ├── test_llm_providers.py   # LLM provider tests
│   ├── test_planner.py         # Planner tests
│   ├── test_orchestrator.py    # Orchestration tests
│   └── ...
│
├── resources/                  # Sample data (kahovka_data/)
├── docs/                       # Development docs & architecture diagrams
└── scripts/                    # Utility scripts
```

## Configuration

### LLM Providers

The system supports multiple LLM providers with automatic fallback:

| Provider | Speed | Free Tier | Setup |
|----------|-------|-----------|-------|
| Gemini | Fast | Yes | `GEMINI_API_KEY` from [AI Studio](https://aistudio.google.com/apikey) |
| Groq | Ultra-fast (300+ tok/s) | Yes | `GROQ_API_KEY` from [Console](https://console.groq.com/) |
| Ollama | Local (5-10 tok/s) | N/A | Automatic in Docker |
| Apertus | Local (GPU) | N/A | Requires GPU (~16GB+ VRAM) |
| OpenAI | Cloud | No (commercial) | `OPENAI_API_KEY` (or `GPT_SECRET`) from [platform.openai.com](https://platform.openai.com/api-keys), plus `openai` in `LLM_PROVIDER_ORDER` |

Anthropic (`ANTHROPIC_API_KEY`) and Mistral (`MISTRAL_API_KEY`) are set up like OpenAI: the key, plus `anthropic` or `mistral` in `LLM_PROVIDER_ORDER`. See `.env.example`.

#### Basic Setup

Add API keys to `.env`:

```bash
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

#### Model Selection

Choose specific models for each provider:

```bash
# Gemini: one model, or a list. One model from the list is picked when the process
# starts; if it answers 404, 429 (quota) or 503, the others are tried in order.
GEMINI_MODELS=gemini-3.5-flash-lite
GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.5-flash

# Groq: Single model (already ultra-fast, no rotation needed)
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama: Local model
OLLAMA_MODEL=llama3.2

# OpenAI
OPENAI_MODELS=gpt-4o-mini
```

Available models:

- **Gemini**: model IDs that the Gemini API lists for `generateContent`, e.g. `gemini-3.5-flash-lite` (default), `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-2.5-flash`
- **Groq**: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- **Ollama**: Any installed model (e.g., `llama3.2`, `mistral`)
- **Apertus**: `swiss-ai/Apertus-8B`, `swiss-ai/Apertus-70B`
- **OpenAI**: `gpt-4o-mini` (default) or another chat model. Reasoning models (the o-series, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.5` and later, `gpt-6` and later) are sent without `temperature`, which they reject.

#### Apertus (Local GPU Model)

[Apertus](https://www.swiss-ai.org/apertus) is an open-source multilingual LLM from Swiss AI. Runs locally via HuggingFace Transformers.

```bash
# Enable Apertus (disabled by default)
APERTUS_ENABLED=true
LLM_ENABLE_APERTUS=true             # .env.example sets it to false
APERTUS_MODEL=swiss-ai/Apertus-8B  # or swiss-ai/Apertus-70B
APERTUS_DEVICE=0                    # GPU device index
APERTUS_DTYPE=bfloat16              # bfloat16, float16, float32

# Add to provider order
LLM_PROVIDER_ORDER=apertus,gemini,groq,ollama
```

**Requirements**: GPU with ~16GB VRAM (8B) or ~140GB VRAM (70B)

#### Advanced LLM Configuration

Control which providers are used and their priority:

```bash
# Enable/disable specific providers
LLM_ENABLE_GEMINI=true
LLM_ENABLE_GROQ=true
LLM_ENABLE_OLLAMA=true
LLM_ENABLE_OPENAI=true

# Set priority order (comma-separated)
# Available: gemini, groq, ollama, apertus, anthropic, mistral, openai
LLM_PROVIDER_ORDER=gemini,groq,ollama

# Routing strategy: fallback | cost | availability
LLM_STRATEGY=fallback
```

**Examples:**

```bash
# Use only Groq (fastest free option)
LLM_ENABLE_GEMINI=false
LLM_ENABLE_OLLAMA=false
LLM_PROVIDER_ORDER=groq

# Use only local Ollama (no external API calls)
LLM_ENABLE_GEMINI=false
LLM_ENABLE_GROQ=false
LLM_PROVIDER_ORDER=ollama

# Prefer Groq, fallback to Ollama
LLM_PROVIDER_ORDER=groq,ollama
```

### Intent Classification

The system automatically classifies user requests into two categories:

- **PIPELINE** - Requests that need data processing (triggers algorithm execution)
- **CONVERSATION** - Questions, guidance requests, or general conversation

Examples:

| Request | Intent | Action |
|---------|--------|--------|
| "Calculate NDVI" | PIPELINE | Executes NDVI algorithm |
| "Detect field boundaries" | PIPELINE | Runs Delineate-Anything model |
| "What can we do next?" | CONVERSATION | Returns helpful guidance |
| "Explain what NDVI means" | CONVERSATION | Provides explanation |

The classifier uses pattern matching for clear action requests and falls back to LLM-based classification for ambiguous cases.

### Multi-Turn Conversations

The system supports follow-up requests that reference previously uploaded data:

```text
User: [uploads image.tif] "Detect agricultural parcels in this area"
→ Runs field boundary detection

User: "What can we do with this data?"
→ Returns guidance about available analyses

User: "Calculate NDVI"
→ Uses the previously uploaded image.tif automatically
```

Context is maintained per chat session, including:

- Previous file attachments with their server paths
- Conversation history for context-aware responses
- Job results for reference in follow-up questions

### Adding New Algorithms

1. Create algorithm file:

   ```python
   # dta/dti/algorithms/my_algorithm.py
   def run(RasterPath: str) -> dict:
       """Process raster and return results."""
       return {"result": ..., "statistics": {...}}
   ```

2. Register in `dta/registry.yaml`:

   ```yaml
   - id: algorithms/my-algorithm
     kind: algorithm
     keywords: [my, algorithm, keywords]
     inputs: [RasterPath]
     outputs: [MyOutput]
     runner:
       type: python
       entrypoint: "dta/dti/algorithms/my_algorithm.py"
   ```

### Field Boundary Detection (Delineate-Anything)

The Delineate-Anything model ([arXiv:2504.02534](https://arxiv.org/abs/2504.02534)) detects agricultural field boundaries from satellite imagery. Its Python dependencies (`ultralytics`, `scipy`, `psutil`) are in the `server` extra. Its upstream code also imports the GDAL Python bindings (`osgeo.gdal`), which build against the system GDAL library, so install that first (Debian/Ubuntu: `libgdal-dev`, which provides `gdal-config`) and then the binding that matches it:

```bash
sudo apt-get install libgdal-dev
uv pip install "gdal==$(gdal-config --version)"   # or: pip install ...
```

The `delineate` extra declares the same dependency: `pip install -e ".[delineate]"` builds the newest `gdal` release, and `uv sync --extra delineate` builds the `gdal` 3.13.3 source distribution pinned in `uv.lock`; each works only where the system libgdal has that version. A later plain `uv sync` removes a `gdal` installed by hand; use `uv sync --inexact` to keep it. The Docker image installs `libgdal-dev` and the matching `gdal` itself.

The model weights (`delineate-anything-small`) are downloaded separately; see [Model weights](#model-weights).

Usage via natural language:

```text
"Detect field boundaries in this image"
"Delineate agricultural parcels from the satellite image"
```

The model outputs GeoPackage files with polygon geometries representing detected fields.

**Note**: Input must be a GeoTIFF with at least 3 bands; bands 1, 2 and 3 are passed to the model as its three image channels. DT4LC runs the small model variant (`DelineateAnything-S.pt`): the registry entry `models/delineate-anything` in `dta/registry.yaml` passes `delineate_model: "small"`.

### Model weights

The Prithvi (`prithvi-eo-v1-100m`) and Delineate-Anything (`delineate-anything-small`) weights are not in the repository. Download them from the **Models** page of the UI, with `POST /v1/ml-models/{model_id}/download`, or from Python:

```bash
uv run python - <<'EOF'
import time
from dta.dti.models import ModelStatus, get_model_manager

manager = get_model_manager()
for model_id in ("prithvi-eo-v1-100m", "delineate-anything-small"):
    if not manager.is_model_available(model_id):
        manager.start_download(model_id)  # returns at once; the download runs in a background thread
        while manager.get_model_status(model_id) == ModelStatus.DOWNLOADING:
            time.sleep(5)
    print(model_id, manager.get_model_status(model_id).value)
EOF
```

The weights come from Hugging Face; Delineate-Anything also fetches its upstream code from GitHub (`Lavreniuk/Delineate-Anything`). Files go to `resources/.cache/models/<model_id>/`; set `DT4LC_MODEL_CACHE` to use another directory (the Docker backend uses `/app/resources/.cache/models`, inside the mounted `./resources`). Licences: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Expected Band Order

The spectral indices (NDVI, NDWI, NDSI, EVI) and change detection find the blue, green, red, NIR and SWIR1 bands as follows (`band_indices` in `dta/dti/raster.py`):

1. **Semantic band descriptions**, case-insensitive and trimmed: `blue` or `B`, `green` or `G`, `red` or `R`, `nir` or `nir08`, `swir`, `swir1` or `swir16` (SWIR1). A role that no description names is taken from the band-count rule in step 3, with a warning, unless that band is already named as another role; otherwise the role is unavailable.
2. **Band codes** in the descriptions (`B2`, `B02`, `B8A`, ...). If `B8` or `B8A` is present, the Sentinel-2 scheme applies: B2 blue, B3 green, B4 red, B8 NIR (B8A if there is no B8), B11 SWIR1. Otherwise, if all codes are within B1-B7, the Landsat 8/9 and HLS L30 scheme applies: B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1. Other code sets go to step 3.
3. **Band count**, with a logged warning:

   | Bands | Assumed order |
   |-------|---------------|
   | 7 or more | Landsat 8/9: B1 coastal, B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1 |
   | 5 or 6 | blue, green, red, NIR, SWIR1 |
   | 4 | red, green, blue, NIR (no SWIR) |
   | fewer than 4 | band 1 red, band 2 NIR (no blue, green or SWIR) |

A 4-band file in blue, green, red, NIR order without descriptions is therefore read as red, green, blue, NIR. Set the descriptions to avoid the guess, e.g. with rasterio: open the file with `rasterio.open(path, "r+")` and call `set_band_description(1, "blue")` and so on for each band.

The spectral-index algorithms (`dta/dti/algorithms/spectral_index.py`) also normalise the values: if the maximum over the required bands exceeds 1000, the raster is taken as reflectance scaled by 10000 and divided by 10000; nodata and non-finite pixels are masked; index values are clipped to [-1, 1]; statistics such as `valid_pixels` count valid pixels only. The land cover and snow classifiers still choose bands by band count alone.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/upload` | POST | Upload a GeoTIFF |
| `/v1/jobs` | POST | Submit an analysis job (prompt and attachments) |
| `/v1/jobs` | GET | List jobs |
| `/v1/jobs/{id}` | GET | Get job status and result |
| `/v1/jobs/{id}/cancel` | POST | Cancel job |
| `/v1/chat` | POST | Legacy: plan and run the last of `messages` as a Server-Sent Events stream (no attachments) |

All routes, including file listing and download, map tiles, Earth Engine data and model management, are in the generated API reference: Swagger UI at <http://localhost:8000/docs> and ReDoc at <http://localhost:8000/redoc> (under Docker: <http://localhost/docs>, <http://localhost/redoc>). See also [docs/API.md](docs/API.md).

### Example Request

```bash
# 1. Upload a GeoTIFF; the response contains its server-side "path"
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@image.tif"
# {"id": "1a2b3c4d", "filename": "image.tif", "path": "/.../resources/.cache/uploads/1a2b3c4d_image.tif", ...}

# 2. Submit an analysis job with the uploaded file attached (202 Accepted)
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate NDVI for vegetation analysis",
    "attachments": [{"id": "1a2b3c4d", "filename": "image.tif", "path": "<path from step 1>", "mime_type": "image/tiff"}]
  }'
# {"id": "<job id>", "status": "pending", ...}

# 3. Poll the job until "status" is "completed" or "failed"
curl http://localhost:8000/v1/jobs/<job id>
```

## Docker Deployment

### Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 80 | React UI (nginx) |
| Backend | 8000 | FastAPI server |
| Ollama | 11434 | Local LLM server |

### Commands

#### With Task

```bash
# Start services (uses cloud LLM providers from .env)
task docker:up

# Build and start services
task docker:up:build

# Start with local Ollama LLM
task docker:up:ollama

# Build and start with Ollama
task docker:up:build:ollama

# View logs
task docker:logs
task docker:logs:backend

# Stop all services
task docker:down

# Stop and remove volumes
task docker:down:clean
```

#### With docker compose

```bash
# Start frontend and backend (add --profile ollama to also start the local Ollama LLM)
docker compose up -d

# View logs
docker compose logs -f

# View specific service
docker compose logs -f backend

# Rebuild after changes
docker compose build --no-cache
docker compose up -d

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Health Checks

```bash
# All services status
docker compose ps

# Backend health
curl http://localhost:8000/v1/health

# Ollama models
curl http://localhost:11434/api/tags
```

### Troubleshooting

**Backend won't start:**

```bash
docker compose logs backend
# Check for port conflicts or missing dependencies
```

**Ollama model download slow:**

```bash
docker compose logs ollama-pull
# Model is ~2GB, takes 2-5 minutes
```

**Frontend shows Network Error:**

```bash
curl http://localhost:8000/v1/health
# Verify backend is running
```

**"All LLM providers failed":**

The message lists every provider that was tried and why it failed, e.g.
`All LLM providers failed: gemini (gemini-3.5-flash-lite): <ErrorType>: <message>; ollama (llama3.2): not available`
(API-key fragments are redacted). Check the keys in `.env`, `LLM_PROVIDER_ORDER`, and for Ollama that `OLLAMA_MODEL` is pulled.

**LLM responses slow:**

```bash
# Add API keys to .env for faster providers:
GROQ_API_KEY=your_key  # 300+ tokens/sec
```

### Resource Requirements

| Service | RAM | CPU | Storage |
|---------|-----|-----|---------|
| Frontend | 128MB | 0.1 | 50MB |
| Backend | 512MB | 0.5 | 200MB |
| Ollama | 4GB+ | 2+ | 3GB |

**Total minimum**: 5GB RAM, 2 CPU cores, 4GB storage

## Sample Data

`resources/kahovka_data/` holds two crops of HLS L30 scenes (Harmonized Landsat Sentinel-2, the Landsat-derived product, not Sentinel-2) taken before and after the breach of the Kakhovka dam on 6 June 2023:

| File | Date | Size |
|------|------|------|
| `hlsl_20230601.tif` | 2023-06-01 (before) | 4,180,159 B |
| `hlsl_20230609.tif` | 2023-06-09 (after) | 3,839,983 B |

Both are 768 × 768 px at 30 m (23.04 km × 23.04 km) in EPSG:32636 with the upper-left corner at (453240, 5167860), about 32.39-32.69° E, 46.46-46.66° N: the lower Dnipro around Kherson, downstream of the dam. They have five bands, B2-B6 (blue, green, red, NIR, SWIR1), named in the band descriptions, stored as int16 reflectance × 10000 with nodata -9999 and DEFLATE compression. They are cut (rows 640-1408, columns 256-1024) from 2906 × 1479 px scenes that are not in the repository.

The pair shows the flooding after the breach: the share of valid pixels with NDWI = (B3 - B5) / (B3 + B5) above 0 grows from 9 % to 40 %, and NDWI changes by more than 0.3 on 41.2 % of them, almost all of it an increase. Upload both files and ask, for example:

```text
"Compare before and after images to detect changes"      (NDVI change, the default)
"Detect water changes between these images"               (NDWI change)
```

## Development

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for environment setup, project layout, tests, adding algorithms, COE flow, and git workflow.

### Task Commands

All development commands are available via [Task](https://taskfile.dev/):

```bash
task --list  # Show all available commands
```

| Command | Description |
|---------|-------------|
| **Environment** | |
| `task venv` | Create virtual environment |
| `task install` | Install with all dev dependencies |
| `task sync` | Sync environment with pyproject.toml |
| **Code Quality** | |
| `task lint` | Run ruff linter |
| `task lint:fix` | Run linter with auto-fix |
| `task format` | Format code with ruff |
| `task format:check` | Check formatting (no changes) |
| `task typecheck` | Run mypy type checker |
| **Testing** | |
| `task test` | Run all tests |
| `task test:fast` | Run tests excluding slow ones |
| `task test:verbose` | Run with verbose output |
| `task coverage` | Run with coverage report |
| **Application** | |
| `task run:api` | Run FastAPI server |
| **Build & Clean** | |
| `task build` | Build wheel and sdist |
| `task clean` | Remove build artifacts |
| `task check` | Run all quality checks |

### Code Style

- **Python**: ruff for linting/formatting, mypy for type checking
- **TypeScript**: ESLint with react-hooks plugin
- **Line length**: 119 characters

### Type Checking

```bash
task typecheck  # or: uv run mypy .

# TypeScript
cd cognitive_ui && npm run lint
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details. Licences of the third-party models it downloads are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Citation

```bibtex
@software{dt4lc,
  author = {Anton Chernyatevich},
  title = {DT4LC - Digital Twin for Land Cover},
  year = {2025},
  url = {https://github.com/IPT-MMDA/DT4LC-project}
}
```
