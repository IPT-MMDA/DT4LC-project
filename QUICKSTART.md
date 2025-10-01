# DT4LC Quick Start Guide

Get up and running with the DT4LC MVP in 5 minutes.

---

## Prerequisites

- Python 3.10+
- UV package manager (or pip)
- Gemini API key (optional, for LLM features)

---

## Installation

```bash
# Clone and navigate
cd /path/to/DT4LC-project

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev,ui,models,api,agents,server]"
```

---

## Configuration

### Set Gemini API Key (Optional)
```bash
export GEMINI_API_KEY=your_key_here
```

Without this key:
- ✅ Server will start
- ✅ Registry will load
- ✅ Algorithms will execute
- ❌ LLM planning will fail gracefully
- ❌ Agent summarization unavailable

---

## Run Server

```bash
# Start the API server
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload

# Server will be available at:
# http://localhost:8000
```

---

## Test Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/v1/health
```

**Response:**
```json
{"ok":true,"service":"DT4LC","version":"1.0.0"}
```

### 2. List Capabilities
```bash
curl http://localhost:8000/v1/capabilities | python -m json.tool
```

**Response:** Full registry with algorithms, models, types

### 3. Generate Execution Plan
```bash
curl -X POST http://localhost:8000/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Calculate NDVI for vegetation analysis"}
    ]
  }' | python -m json.tool
```

**Response:**
```json
{
  "ok": true,
  "plan": {
    "flow": "auto",
    "steps": [
      {"uses": "input/file", "binds": {}},
      {"uses": "algorithms/ndvi", "binds": {}},
      ...
    ],
    "outputs": ["publish: chat"]
  }
}
```

### 4. Execute Pipeline (requires Gemini API key)
```bash
curl -X POST http://localhost:8000/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Calculate NDVI"}
    ]
  }' | python -m json.tool
```

---

## Run Tests

```bash
# All Phase 1 tests
pytest tests/test_phase1_integration.py -v

# Specific test
pytest tests/test_phase1_integration.py::test_ndvi_algorithm_direct -v

# Without coverage reports
pytest tests/test_phase1_integration.py -v --no-cov
```

**Expected:** 7/7 tests passing (some require Gemini API key)

---

## Project Structure

```
DT4LC-project/
├── PLAN.md                      # Full implementation plan
├── PHASE1_COMPLETE.md           # Phase 1 summary
├── QUICKSTART.md                # This file
├── README.md                    # Original project README
├── dta/                         # Digital Twin Aggregator
│   ├── registry.yaml            # Component registry
│   ├── dti/
│   │   ├── executor.py          # Pipeline executor
│   │   ├── algorithms/          # NDVI, statistics
│   │   ├── assets/              # Data management
│   │   ├── models/              # ML models
│   │   └── coe/                 # Orchestration agents
│   └── config/                  # Configuration
├── server/                      # FastAPI server
│   ├── app.py                   # API endpoints
│   └── schemas.py               # Request/response types
├── tests/                       # Test suite
└── resources/                   # Data files
```

---

## Common Tasks

### Add New Algorithm

1. Create file in `dta/dti/algorithms/your_algo.py`:
```python
def run(RasterPath: str) -> dict:
    # Your algorithm here
    return {"result": ...}
```

2. Register in `dta/registry.yaml`:
```yaml
- id: algorithms/your_algo
  kind: algorithm
  keywords: [your, keywords]
  inputs: [RasterPath]
  outputs: [YourOutputType]
  runner:
    type: python
    entrypoint: "dta/dti/algorithms/your_algo.py"
```

3. Test it:
```bash
curl -X POST http://localhost:8000/v1/plan \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "run your_algo"}]}'
```

### Debug Execution

Check server logs:
```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --log-level debug
```

Test executor directly:
```python
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ExecutionPlan, PlanStep

executor = PipelineExecutor()
plan = ExecutionPlan(steps=[...])
result = executor.execute(plan)
print(result)
```

### View Available Components

```bash
# Via API
curl http://localhost:8000/v1/capabilities

# Via Python
python -c "
from dta.dti.registry import load_registry
reg = load_registry()
for item in reg.instances:
    print(f'{item.id} ({item.kind}): {item.keywords}')
"
```

---

## Troubleshooting

### "GEMINI_API_KEY not set"
- **Cause**: LLM features require API key
- **Fix**: `export GEMINI_API_KEY=your_key` or use without LLM

### "Asset not found"
- **Cause**: Data file missing
- **Fix**: Check `resources/` directory or provide full path

### "Module not found"
- **Cause**: Dependencies not installed
- **Fix**: `uv pip install -e ".[dev,api,agents]"`

### Server won't start
- **Cause**: Port 8000 already in use
- **Fix**: `lsof -ti:8000 | xargs kill` or use different port

### Tests failing
- **Cause**: Missing test data or API key
- **Fix**: Run with `--no-cov` and check individual test output

---

## API Documentation

Once server is running, visit:
```
http://localhost:8000/docs         # Swagger UI
http://localhost:8000/redoc        # ReDoc
```

---

## Next Steps

1. ✅ Complete Phase 1 (DONE!)
2. 📋 Review PLAN.md for Phase 2 roadmap
3. 🚀 Start Phase 2: Multi-LLM integration
4. 📊 Add more algorithms and models
5. 🔧 Implement async job queue (Phase 4)

---

## Getting Help

- **Architecture**: See `PLAN.md` and `PHASE1_COMPLETE.md`
- **API Reference**: Server docs at `/docs`
- **Code Examples**: Check `tests/test_phase1_integration.py`
- **Registry Format**: See `dta/registry.yaml` with comments

---

## Success Checklist

- [ ] Server starts without errors
- [ ] `/v1/health` returns `{"ok": true}`
- [ ] `/v1/capabilities` lists registry
- [ ] Tests pass: `pytest tests/test_phase1_integration.py -v`
- [ ] Can generate plans via `/v1/plan`
- [ ] (Optional) Can execute with Gemini API key

**All checked?** 🎉 You're ready to build!
