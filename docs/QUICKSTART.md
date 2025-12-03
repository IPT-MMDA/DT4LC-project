# DT4LC Quick Start Guide

Get up and running with DT4LC in 5 minutes.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- LLM provider (one of):
  - Gemini API key (cloud)
  - Groq API key (cloud)
  - Ollama installed locally (free)

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and navigate
cd /path/to/DT4LC-project

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev,server]"
```

## Configuration

### Option 1: Cloud LLMs (Gemini/Groq)

```bash
# Create .env file
cp .env.example .env

# Add your API keys
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env
```

### Option 2: Local LLM (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama server
ollama serve

# Configure .env
echo "LLM_PROVIDER_ORDER=ollama" >> .env
echo "OLLAMA_MODEL=llama3.2" >> .env
```

## Run Server

### Option A: Docker (Recommended)

```bash
# Build and start all services
task docker:up:build

# Or with Ollama for local LLM
task docker:up:build:ollama

# View logs
task docker:logs

# Application available at http://localhost
```

### Option B: Local Development

```bash
# Start the API server
dt4lc-api

# Or with uvicorn directly
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Server available at http://localhost:8000
```

## Test Endpoints

### Health Check

```bash
curl http://localhost:8000/v1/health
# {"ok":true,"service":"DT4LC","version":"1.0.0"}
```

### Submit Job

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "calculate ndvi"}'
```

### Check Job Status

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

### Upload File and Analyze

```bash
# 1. Upload GeoTIFF
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@/path/to/data.tif"

# 2. Submit job with attachment
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "calculate ndvi",
    "attachments": [{
      "id": "file_id",
      "filename": "data.tif",
      "path": "/tmp/dt4lc_uploads/file_id_data.tif"
    }]
  }'
```

## Run Tests

```bash
# All tests
task test

# Fast tests (skip slow ones)
task test:fast

# Specific test file
pytest tests/test_orchestrator.py -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Health check |
| `/v1/jobs` | POST | Submit async job |
| `/v1/jobs/{id}` | GET | Get job status/results |
| `/v1/jobs/{id}/cancel` | POST | Cancel job |
| `/v1/jobs` | GET | List jobs (paginated) |
| `/v1/upload` | POST | Upload GeoTIFF file |
| `/v1/capabilities` | GET | List available algorithms |
| `/v1/models` | GET | List available models |
| `/v1/metrics` | GET | System metrics |
| `/v1/queue/stats` | GET | Queue statistics |

## Interactive API Docs

Once server is running:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## Troubleshooting

### "All LLM providers failed"

- Check that at least one LLM provider is configured
- For Ollama: ensure `ollama serve` is running
- For cloud APIs: verify API keys in `.env`

### "Module not found"

```bash
uv pip install -e ".[dev,server]"
```

### Port 8000 in use

```bash
# Find and kill process
lsof -ti:8000 | xargs kill

# Or use different port
dt4lc-api --port 8001
```

## Next Steps

- See [API.md](API.md) for complete API reference
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See main [README.md](../README.md) for full documentation
