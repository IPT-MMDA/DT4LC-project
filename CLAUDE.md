# DT4LC Project Guide

## Project Overview

Digital Twin for Land Cover (DT4LC) - A cognitive framework for land cover change detection using satellite imagery, ML models, and LLM-powered orchestration.

## Monorepo Structure

```text
DT4LC-project/
├── docker-compose.yml      # Docker orchestration (all services)
├── Dockerfile              # Backend container
├── .env.example            # Environment template
├── pyproject.toml          # Python dependencies
│
├── dta/                    # Digital Twin Application (Python)
│   ├── config.py           # Configuration & paths
│   ├── registry.yaml       # Component registry
│   └── dti/                # Digital Twin Instance
│       ├── coe/            # Context Orchestration Engine
│       │   ├── llm/        # LLM providers (Gemini, Groq, Ollama)
│       │   ├── context_agent.py    # Intent understanding
│       │   ├── planner_agent.py    # Pipeline planning
│       │   ├── decision_agent.py   # Plan validation
│       │   └── orchestrator.py     # Main orchestration
│       ├── algorithms/     # NDVI, Statistics, Change Detection
│       ├── models/         # Prithvi and ML models
│       └── executor.py     # Pipeline execution
│
├── server/                 # FastAPI backend
│   ├── app.py              # Main application & routes
│   └── schemas.py          # API schemas
│
├── frontend/               # React + TypeScript UI
│   ├── Dockerfile          # Frontend container
│   ├── nginx.conf          # Production proxy config
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── api/            # API client (axios)
│   │   └── store/          # Zustand state management
│   └── package.json
│
├── tests/                  # Python test suite
├── resources/              # Sample data (kahovka_data/)
├── docs/                   # Development docs & architecture diagrams
└── scripts/                # Utility scripts
```

## Quick Commands

```bash
# Docker deployment (from DT4LC-project/)
cp .env.example .env
docker compose up -d

# Backend development
source .venv/bin/activate
pytest tests/ -v
uvicorn server.app:app --reload --port 8000

# Frontend development
cd frontend
npm install
npm run dev
```

## Key Files

- **Pipeline orchestration**: `dta/dti/coe/orchestrator.py`
- **LLM routing**: `dta/dti/coe/llm/router.py`
- **Component registry**: `dta/registry.yaml`
- **API server**: `server/app.py`
- **Change detection**: `dta/dti/algorithms/change_detection.py`
- **NDVI analysis**: `dta/dti/algorithms/ndvi.py`

## LLM Configuration

The system supports multiple LLM providers with automatic fallback:

```bash
# .env file

# Provider selection (enable/disable)
LLM_ENABLE_GEMINI=true
LLM_ENABLE_GROQ=true
LLM_ENABLE_OLLAMA=true

# Priority order (comma-separated)
LLM_PROVIDER_ORDER=gemini,groq,ollama

# Strategy: fallback | cost | availability
LLM_STRATEGY=fallback

# API Keys
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Default fallback chain**: Gemini → Groq → Ollama

Get free API keys:

- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/ (ultra-fast, recommended)

## Adding New Algorithms

1. Create `dta/dti/algorithms/your_algo.py`:

```python
def run(RasterPath: str) -> dict:
    return {"result": ...}
```

2. Register in `dta/registry.yaml`:

```yaml
- id: algorithms/your-algo
  kind: algorithm
  keywords: [your, keywords]
  inputs: [RasterPath]
  outputs: [YourOutput]
  runner:
    type: python
    entrypoint: "dta/dti/algorithms/your_algo.py"
```

## Code Style

- Python: ruff for linting/formatting, mypy for type checking (strict mode)
- Line length: 119 characters
- TypeScript: ESLint with react-hooks and react-refresh plugins
