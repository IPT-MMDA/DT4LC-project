# DT4LC Implementation Plan - MVP Phase

**Project**: Digital Twin for Land Cover (DT4LC)
**Version**: 1.0 MVP
**Status**: In Progress
**Last Updated**: 2025-10-02

---

## Executive Summary

This plan outlines the implementation of a modular Digital Twin architecture with:
- **COE (Context Orchestration Engine)**: Agentic layer using LLMs to understand requests and draft pipeline plans
- **DTA (Digital Twin Aggregator)**: Runtime execution engine with registry-based model/algorithm orchestration
- **Server API**: FastAPI-based backend for frontend communication
- **Registry System**: YAML-based component registry for dynamic pipeline construction

---

## Current Architecture Status

### ✅ **Working Components**
- Registry system (`dta/registry.yaml`)
- COE agents (Context Understanding, Decision Making, Planner)
- Server infrastructure (FastAPI with SSE streaming)
- Prithvi model integration (stub)
- Comprehensive test suite structure

### ❌ **Missing Critical Components**
- Pipeline Executor (no execution engine)
- Data Asset Manager (no data loading)
- Algorithm library (NDVI, change detection, etc.)
- Model Registry wrapper
- Post-processing pipeline
- Server ↔ DTA integration

### 🔙 **Legacy Files** (Reference Only)
- `dta.bak/` - Old DTA implementation
- `orchestrator.bak/` - Old orchestrator (replaced by COE)
- `cognitive_ui/` - POC Streamlit UI (not part of MVP)

---

## Implementation Phases

## 🔴 Phase 1: Critical Path - Foundation (Week 1-2)

**Goal**: Minimal viable pipeline execution from user request to results

### 1.1 Pipeline Executor (`dta/dti/executor.py`)
- Create `PipelineExecutor` class
- Dynamic runner dispatch (python/agent/passthrough)
- Step-by-step execution with artifact passing
- Progress callbacks/events
- Error handling and rollback
- Integration with registry items

**Key Methods**:
```python
class PipelineExecutor:
    def execute(plan: ExecutionPlan) -> ExecutionResult
    def _run_step(step: PlanStep, artifacts: dict) -> Any
    def _dispatch_runner(runner: Runner, step: PlanStep) -> Any
```

### 1.2 Algorithm Library (`dta/dti/algorithms/`)
Port essential algorithms from `dta.bak/`:
- `ndvi.py` - NDVI calculation
- `statistics.py` - Basic statistical analysis
- Base algorithm interface

### 1.3 Data Asset Manager (`dta/dti/assets/`)
Basic data loading:
- `manager.py` - Main DataAssetManager class
- Support for local file:// paths
- GeoTIFF loading with rasterio
- Metadata extraction (CRS, bounds, bands)

### 1.4 Server Integration (`server/app.py`)
- Update imports to use new COE
- Wire COE → Executor flow
- Add `/v1/execute` endpoint
- Update schemas for new architecture

### 1.5 End-to-End Testing
- Validate full flow: Request → COE → DTA → Response
- Update `test_full_flow_prithvi.py` for new architecture
- Basic integration tests

**Deliverable**: Working demo with Prithvi test passing

---

## 🟡 Phase 2: Agent Intelligence (Week 3-4)

**Goal**: Multi-LLM support with intelligent planning

### 2.1 LLM Backend Abstraction (`dta/dti/coe/llm/`)
```
dta/dti/coe/llm/
├── base.py              # BaseLLMProvider protocol
├── gemini.py            # Google Gemini (primary, existing)
├── local_llama.py       # Ollama/Llama.cpp
├── router.py            # Smart routing/fallback
└── config.py            # Provider configuration
```

### 2.2 Enhanced Context Agent
- Refactor to use LLMRouter
- Enhanced structured context extraction:
  - Temporal range
  - Geographic bounds
  - Required data types
  - Analysis goals

### 2.3 Intelligent Planner
- LLM-powered pipeline generation
- Constraint satisfaction (type matching, resource checks)
- Multi-path planning
- Plan optimization

### 2.4 Hybrid Planning Mode
- Template-based plans for common scenarios (fast)
- LLM for complex/ambiguous requests (slow)
- Confidence scoring to choose mode
- Plan validation and safety checks

**Deliverable**: System can handle complex ambiguous requests with fallback to local models

---

## 🟢 Phase 3: Model Expansion (Week 5-6)

**Goal**: Multiple models and rich capabilities

### 3.1 Model Registry (`dta/dti/models/registry.py`)
- `BaseModel` protocol
- Model loader/wrapper classes
- Third-party model adapters
- Model versioning and metadata

### 3.2 Prithvi Full Integration
- Complete wrapper class for inference
- Batch processing support
- GPU/CPU dispatch
- Feature extraction outputs

### 3.3 Climate Model Integration (`dta/dti/models/climate/`)
- DestinE model support
- Temporal interpolation
- Multi-variable predictions
- Uncertainty quantification

### 3.4 VI Prediction Models (`dta/dti/models/vi_prediction/`)
- Time series models (LSTM, Transformer)
- Multi-spectral band prediction
- Ensemble methods

### 3.5 Post-Processing (`dta/dti/post_processing/`)
- Raster visualization (colormaps, hillshade)
- Chart generation (matplotlib, plotly)
- Statistics formatting
- GeoJSON output for web maps
- LLM-powered insight generation

**Deliverable**: Multi-model pipelines with rich visualizations

---

## 🔵 Phase 4: Production Ready (Week 7-8)

**Goal**: Scalability, robustness, and production deployment

### 4.1 Async Job Queue (`server/jobs.py`)
- In-memory job queue
- Background worker pool
- Progress tracking
- Result caching
- Cleanup policies

### 4.2 Enhanced Server API
**New Endpoints**:
```python
POST /v1/chat           # Chat interface (existing)
POST /v1/plan           # Generate plan only
POST /v1/execute        # Execute a plan
GET  /v1/jobs/{id}      # Job status
POST /v1/upload         # Upload data (existing)
GET  /v1/models         # List available models
GET  /v1/capabilities   # List registry items
```

### 4.3 Comprehensive Testing
- Unit tests for all algorithms
- Integration tests for pipelines
- Performance tests (large rasters, concurrent requests)
- Mock LLM providers for CI

### 4.4 Enhanced Registry (`dta/registry.yaml`)
- Model metadata (GPU requirements, memory, latency)
- Cost estimates (API calls, compute time)
- Version constraints
- Conditional availability (feature flags)
- Tags and categories

### 4.5 Configuration Management (`dta/dti/config/`)
- Environment-based config (dev/prod)
- LLM provider credentials
- Model paths and URLs
- Resource limits (memory, timeout)
- Feature flags

### 4.6 API Documentation
- OpenAPI/Swagger schema
- Example requests/responses
- Postman collection
- Architecture diagrams

**Deliverable**: Production-ready system with SLA guarantees

---

## Technical Decisions

### LLM Strategy
- **Primary**: Google Gemini API (2.5-flash model)
- **Future**: Local Llama via Ollama for privacy/offline scenarios
- **Routing**: Simple primary/fallback (no cost optimization yet)

### Execution Model
- **Phase 1**: Synchronous in-process execution
- **Phase 4**: Python asyncio + in-memory queue for async jobs
- **Future**: Consider Celery + Redis if scaling needed

### Model Storage
- **Current**: Hugging Face Hub (Prithvi)
- **Local**: `dta/dti/models/third_party/` for downloaded models
- **Future**: S3/cloud storage for large models

### Visualization
- **Phase 1**: Server-side rendering (matplotlib → PNG)
- **Phase 3**: GeoJSON + metadata for client-side rendering
- **Future**: WMS server if needed

### Database
- **Phase 1-3**: No persistent storage (stateless)
- **Phase 4**: SQLite for job history (optional)
- **Future**: PostgreSQL + PostGIS if geodata storage needed

### Deployment
- **Target**: Local development only
- **Docker**: Not required for MVP
- **CI/CD**: GitHub Actions for tests (future)

---

## Directory Structure (Target)

```
dta/
├── registry.yaml                    # Component registry
├── config/                          # Configuration
│   ├── __init__.py
│   └── settings.py
├── dti/                             # Digital Twin Instance
│   ├── __init__.py
│   ├── schemas.py                   # Data models
│   ├── registry.py                  # Registry loader
│   ├── executor.py                  # Pipeline executor ⭐ NEW
│   ├── algorithms/                  # Algorithm library ⭐ NEW
│   │   ├── __init__.py
│   │   ├── ndvi.py
│   │   ├── statistics.py
│   │   └── change_detection.py
│   ├── assets/                      # Data management ⭐ NEW
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── models/                      # Model wrappers
│   │   ├── __init__.py
│   │   ├── registry.py              # ⭐ NEW
│   │   └── third_party/
│   │       └── prithvi_eo_v1_100m/
│   ├── post_processing/             # ⭐ NEW
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── insights.py
│   └── coe/                         # Context Orchestration Engine
│       ├── __init__.py
│       ├── orchestrator.py          # ✅ EXISTS
│       ├── context_agent.py         # ✅ EXISTS
│       ├── decision_agent.py        # ✅ EXISTS
│       ├── planner_agent.py         # ✅ EXISTS
│       └── llm/                     # ⭐ PHASE 2
│           ├── base.py
│           ├── gemini.py
│           ├── router.py
│           └── config.py

server/
├── app.py                           # FastAPI server (update)
├── schemas.py                       # API schemas (update)
└── jobs.py                          # Job queue ⭐ PHASE 4

tests/
├── conftest.py
├── test_full_flow_prithvi.py        # Update for new arch
├── test_executor.py                 # ⭐ NEW
├── test_algorithms.py               # ⭐ NEW
└── test_assets.py                   # ⭐ NEW
```

---

## Success Criteria

### Phase 1 (MVP Foundation)
- [ ] User can send chat request to server
- [ ] COE generates valid execution plan
- [ ] Executor runs plan and produces results
- [ ] Server returns results to client
- [ ] Test suite passes (Prithvi flow)

### Phase 2 (Intelligence)
- [ ] System handles ambiguous requests
- [ ] LLM fallback works (Gemini → local Llama)
- [ ] Plan quality improves with LLM reasoning
- [ ] Context extraction is rich and accurate

### Phase 3 (Capabilities)
- [ ] 3+ models integrated (Prithvi, climate, VI)
- [ ] Multi-step pipelines work
- [ ] Visualizations are rich and accurate
- [ ] Post-processing generates insights

### Phase 4 (Production)
- [ ] Async job execution works
- [ ] API documentation complete
- [ ] Test coverage >80%
- [ ] Performance benchmarks met (e.g., <30s for NDVI)
- [ ] Error handling is robust

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API costs | High | Use flash models, implement caching |
| Model inference slow | Medium | Add timeout limits, queue system |
| Memory exhaustion (large rasters) | High | Chunked processing, resource limits |
| Plan validation failures | Medium | Extensive testing, fallback to simple plans |
| Third-party model integration | Medium | Keep wrappers thin, document assumptions |

---

## Next Steps (Today)

1. ✅ Create this PLAN.md
2. Create Pipeline Executor (`dta/dti/executor.py`)
3. Port NDVI algorithm (`dta/dti/algorithms/ndvi.py`)
4. Create Data Asset Manager (`dta/dti/assets/manager.py`)
5. Update server integration
6. Test full flow

**Estimated Time**: 4-6 hours for Phase 1 foundation

---

## Notes

- **Legacy files** (`*.bak/`) are kept for reference but not used directly
- **CognitiveUI** was a POC; MVP focuses on server/API
- **Testing strategy**: Write tests incrementally as components are built
- **Target environment**: Local development only (no Docker/K8s yet)
- **LLM priority**: Gemini-only for Phase 1, multi-LLM in Phase 2
