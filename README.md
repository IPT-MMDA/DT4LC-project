# Cognitive Digital Twin for Land Cover Change Detection (DT4LC Project)

Scalable Digital Twin Models for Land Cover Change Detection Using Machine Learning

**📋 [Quick Start Guide](QUICKSTART.md)** | **📖 [Implementation Plan](PLAN.md)** | **✅ [Phase 1](PHASE1_COMPLETE.md)** | **✅ [Phase 2](PHASE2_COMPLETE.md)**

---

## 🚀 Status: MVP Phase 2 Complete!

**Latest**: Intelligent LLM-powered planner with hybrid mode (Oct 2, 2025)

### Phase 1 ✅
- ✅ Pipeline executor with algorithm support
- ✅ NDVI and statistical analysis
- ✅ FastAPI server with 6 endpoints
- ✅ Test coverage (7/7 passing)

### Phase 2 ✅
- ✅ LLM backend abstraction (Gemini + Ollama)
- ✅ Automatic fallback routing
- ✅ Configuration management
- ✅ Context agent with router
- ✅ LLM-powered intelligent planner
- ✅ Hybrid planning (template + LLM)
- ✅ Confidence scoring
- ✅ Test coverage (29/29 passing)
- ⏳ Phase 3: Production hardening (next)

---

## Overview

This project implements a cognitive digital twin framework with a modular architecture:

- **Context Orchestration Engine (COE)**: Agentic layer using LLMs to understand requests and generate pipeline plans
- **Digital Twin Aggregator (DTA)**: Execution engine with registry-based algorithms, models, and data management
- **Server API**: FastAPI-based HTTP interface for frontend communication
- **Registry System**: YAML-based component registry for dynamic capability discovery

**Current MVP**: The system supports NDVI calculation, statistical analysis, and LLM-powered insights via HTTP API. Frontend integration ready.

## Features

- Interactive visualization of satellite imagery
- Temporal comparison of land cover changes
- AI-powered analysis and interpretation of environmental patterns
- Query interface for exploring specific aspects of detected changes
- Synthetic data generation for historical comparisons

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip

### Installing UV

UV is a fast, reliable Python package installer and resolver. To install UV:

```bash
# macOS/Linux
curl -sSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

For more installation options, see the [UV documentation](https://github.com/astral-sh/uv).

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/your-org/dt4lc-project.git
cd dt4lc-project

# Create and activate a virtual environment with UV
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install the package with development dependencies
# Optional extras:
#   ui      – Streamlit and rasterio
#   models  – model-related libs
#   api     – FastAPI + Uvicorn for the HTTP server (optional)
uv pip install -e ".[dev,ui,models]"
```

## Data Management

### Downloading Model Weights and Sample Data (optional)

You can optionally download the Prithvi model weights and sample data. The current orchestration uses a lightweight Prithvi features stub so weights are not strictly required to run the basic flow.

Use the provided script:

```bash
# Download Prithvi model weights and sample data
python scripts/fetch_prithvi_v1_weights.py
```

This script will:

1. Download the pre-trained Prithvi model weights
2. Place them in the correct directory structure
3. Download sample satellite imagery datasets for testing

After running this script, you can experiment with model-backed features when enabled.

## Running the Application

Once installed, you can run the application in several ways:

### Using the CLI Command (recommended)

```bash
# Run with default UI (Streamlit)
cdt --ui streamlit

# Check version
cdt --version
```

### Using Streamlit Directly

```bash
# Run the Streamlit app directly
streamlit run cognitive_ui/app.py
```

### Using UV

```bash
# Run with UV
uv run cdt
```

The application will be available at [localhost](http://localhost:8501) by default.

### Optional: Run the HTTP API server

The server exposes a minimal POST `/flow` endpoint that accepts a request and returns the planned pipeline and execution output (JSON).

```bash
# Install API extras if not installed yet
uv pip install -e ".[api]"

# Start the API (factory pattern)
dt4lc-api

# Then POST a request (example)
curl -X POST http://127.0.0.1:8000/flow -H 'Content-Type: application/json' \
  -d '{"prompt":"ndvi on kahovka data"}'
```

## Configuration

### Streamlit Configuration

Streamlit settings are configured in `.streamlit/config.toml`. You can modify this file to change server behavior, themes, and other Streamlit-specific settings.

### Application Configuration

Application-specific settings are in `cognitive_ui/config.py`. This includes:

- Path definitions
- Visualization parameters
- UI settings
- Default query templates

## Project Structure

```text
dt4lc-project/
├── .streamlit/                 # Streamlit configuration
├── capabilities/
│   └── capabilities.yaml       # Registry of tools (ids, inputs, outputs, tags)
├── orchestrator/               # Context Orchestration Engine
│   ├── agents.py               # ContextUnderstanding, DecisionMaking, Planner
│   ├── registry.py             # Loads capabilities.yaml
│   └── types.py                # Plan and step data structures
├── dta/                        # Digital Twin Aggregator (runtime services)
│   ├── assets.py               # DataAssetManager (e.g., Kahovka raster loader)
│   ├── algorithms.py           # NDVI, NDVI change (minimal)
│   ├── executor.py             # PipelineExecutor
│   ├── models.py               # ModelRegistry (Prithvi features stub)
│   └── post.py                 # PostProcessor (summary only)
├── server/                     # Optional HTTP API server (FastAPI)
│   ├── app.py                  # /flow endpoint
│   └── cli.py                  # dt4lc-api entry point
├── cognitive_ui/               # UI and interface layer
│   ├── app.py                  # Streamlit app entry
│   ├── cli.py                  # cdt --ui streamlit
│   ├── interface/              # UI-neutral controller (run_flow)
│   ├── ui_streamlit/           # Streamlit implementation placeholder
│   └── ui/                     # Streamlit UI components
├── digital_twin/
│   └── models/prithvi_v1/      # Prithvi model implementation
├── resources/                  # Data resources (e.g., kahovka_data/*.tif)
├── scripts/                    # Utility scripts
└── pyproject.toml              # Project configuration
```

## How it Works (Architecture at a Glance)

1) The UI (Streamlit) collects a natural-language request. Use the Problem Solving tab and click “Run Orchestrated Flow (Planner)”.
2) The Interface layer calls the orchestrator in-process.
3) The Orchestrator reads `capabilities.yaml`, interprets the intent, and drafts a plan (sequence of steps with inputs/outputs).
4) The DTA `PipelineExecutor` runs the plan by invoking loaders/models/algorithms and returns artifacts.
5) Post-processing returns a concise textual summary for now (WMS/static visualization can be added later).

Example intents supported today:
- “ndvi on kahovka data” → load Kahovka raster → compute NDVI → summarize
- “ndvi change on kahovka data” with two uploaded images (future UI hook) → compute NDVI on both → change map → summarize

## Citation

If you use this software in your research, please cite:

```bibtex
@software{cognitive_digital_twin,
  author = {Anton Chernyatevich},
  title = {Cognitive Digital Twin for Land Cover Change Detection (DT4LC Project)},
  year = {2025},
  url = {https://github.com/IPT-MMDA/DT4LC-project}
}
```

## License

This project is licensed under the Research Use License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.
