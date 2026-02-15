# DT4LC - Digital Twin for Land Cover

A cognitive digital twin framework for land cover change detection and vegetation analysis using satellite imagery, machine learning, and LLM-powered orchestration.

## Features

- **NDVI Analysis** - Calculate vegetation health indices from satellite imagery
- **Change Detection** - Compare before/after images to detect land cover changes
- **Field Boundary Detection** - Delineate-Anything model for agricultural field segmentation
- **LLM-Powered Planning** - Natural language interface with automatic pipeline generation
- **Intent Classification** - Smart routing between pipeline execution and conversational responses
- **Multi-Turn Conversations** - Follow-up requests using previous context and attachments
- **Multi-Provider LLM Support** - Gemini, Groq, and local Ollama with automatic fallback
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
git clone <repo-url>
cd DT4LC-project

# 2. Set up environment
cp .env.example .env
# Optional: Add GEMINI_API_KEY or GROQ_API_KEY for faster LLM responses

# 3. Start all services
docker compose up -d

# 4. Wait for services (1-2 min for Ollama model download)
docker compose ps

# 5. Open application
open http://localhost
```

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
task run:api:dev    # Start server with hot reload

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

### Using Task (Recommended)

```bash
task test              # Run all tests
task test:fast         # Run tests excluding slow ones
task test:verbose      # Run with verbose output
task coverage          # Run with coverage report
```

### Using pytest directly

```bash
source .venv/bin/activate
pytest tests/ -v
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
│   │   │   ├── planner_agent.py
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

#### Basic Setup

Add API keys to `.env`:

```bash
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

#### Model Selection

Choose specific models for each provider:

```bash
# Gemini: Single model or multiple (rotates for rate limit balancing)
GEMINI_MODELS=gemini-2.5-flash
GEMINI_MODELS=gemini-2.0-flash,gemini-2.5-flash  # rotates between models

# Groq: Single model (already ultra-fast, no rotation needed)
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama: Local model
OLLAMA_MODEL=llama3.2
```

Available models:

- **Gemini**: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash-lite`
- **Groq**: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- **Ollama**: Any installed model (e.g., `llama3.2`, `mistral`)
- **Apertus**: `swiss-ai/Apertus-8B`, `swiss-ai/Apertus-70B`

#### Apertus (Local GPU Model)

[Apertus](https://www.swiss-ai.org/apertus) is an open-source multilingual LLM from Swiss AI. Runs locally via HuggingFace Transformers.

```bash
# Enable Apertus (disabled by default)
APERTUS_ENABLED=true
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

# Set priority order (comma-separated)
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

The Delineate-Anything model detects agricultural field boundaries from satellite imagery:

```bash
# Install dependencies
pip install -e ".[delineate]"
```

Usage via natural language:

```text
"Detect field boundaries in this image"
"Delineate agricultural parcels from the satellite image"
```

The model outputs GeoPackage files with polygon geometries representing detected fields.

**Note**: Input must be GeoTIFF format with at least 3 bands (RGB). Model variants:

- `small`: Faster inference, good for initial testing
- `large`: Higher accuracy, better for production use

Set model via environment:

```bash
DELINEATE_MODEL=large  # or "small" (default)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/chat` | POST | Submit analysis request |
| `/v1/upload` | POST | Upload raster file |
| `/v1/jobs` | GET | List jobs |
| `/v1/jobs/{id}` | GET | Get job status |
| `/v1/jobs/{id}/cancel` | POST | Cancel job |

### Example Request

```bash
# Upload a file
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@image.tif"

# Submit analysis
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Calculate NDVI for vegetation analysis",
    "attachments": [{"id": "att-1", "filename": "image.tif", "path": "/tmp/image.tif", "mime_type": "image/tiff"}]
  }'
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
# Start all services
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

Sample Kahovka region data in `resources/kahovka_data/`:

- `hlsl_20230601.tif` - Before image
- `hlsl_20230609.tif` - After image

Test change detection:

```text
"Compare before and after images to detect changes"
```

## Development

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
| `task run:ui` | Run Streamlit UI |
| `task run:api` | Run FastAPI server |
| `task run:api:dev` | Run API with hot reload |
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

This project is licensed under the Research Use License - see the [LICENSE](LICENSE) file for details.

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
# Build Backend
docker build --network sagemaker -t dt4lc-backend .

# Run Backend
docker run -d --network sagemaker --name dt4lc-backend --env-file .env -v $(pwd)/resources:/app/resources dt4lc-backend

# Build Frontend (after updating nginx.conf)
docker build --network sagemaker -t dt4lc-frontend cognitive_ui

# Run Frontend
docker run -d --network sagemaker --name dt4# Build Backend
docker build --network sagemaker -t dt4lc-backend .

# Run Backend
docker run -d --network sagemaker --name dt4lc-backend --env-file .env -v $(pwd)/resources:/app/resources dt4lc-backend

# Build Frontend (after updating nginx.conf)
docker build --network sagemaker -t dt4lc-frontend cognitive_ui

# Run Frontend
docker run -d --network sagemaker --name dt4