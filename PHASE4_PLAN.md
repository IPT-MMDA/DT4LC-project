# Phase 4 Implementation Plan - Production Ready

## Overview

Phase 4 focuses on making the system production-ready with async processing, enhanced APIs, configuration management, and comprehensive testing.

**Goal**: Scalability, robustness, and deployment readiness

**Estimated Time**: 12-18 hours

---

## Components to Implement

### 1. Async Job Queue System ⭐ Priority 1
**Time Estimate**: 3-4 hours

**Files to Create**:
- `server/jobs.py` - Job queue and worker pool
- `server/models.py` - Job models and schemas
- `tests/test_phase4_jobs.py` - Job queue tests

**Features**:
- In-memory job queue (asyncio.Queue)
- Background worker pool
- Job status tracking (pending, running, completed, failed)
- Progress updates via callbacks
- Result caching (30min TTL)
- Automatic cleanup of old jobs (1 hour retention)

**Implementation Details**:
```python
# Job States
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Job Model
@dataclass
class Job:
    id: str
    status: JobStatus
    prompt: str
    plan: ExecutionPlan | None
    result: dict | None
    progress: float  # 0.0 to 1.0
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

# JobQueue
class JobQueue:
    - submit_job(prompt: str) -> str  # Returns job_id
    - get_job(job_id: str) -> Job
    - cancel_job(job_id: str) -> None
    - list_jobs() -> list[Job]
    - cleanup_old_jobs() -> int
```

---

### 2. Enhanced Server API ⭐ Priority 1
**Time Estimate**: 3-4 hours

**Files to Modify**:
- `server/app.py` - Add new endpoints

**New Endpoints**:

#### POST /v1/jobs
Submit async job
```json
Request: {"prompt": "calculate ndvi on kahovka data"}
Response: {
  "job_id": "abc123",
  "status": "pending",
  "created_at": "2025-10-02T..."
}
```

#### GET /v1/jobs/{job_id}
Get job status and results
```json
Response: {
  "id": "abc123",
  "status": "completed",
  "progress": 1.0,
  "result": {...},
  "plan": {...},
  "created_at": "...",
  "completed_at": "..."
}
```

#### POST /v1/jobs/{job_id}/cancel
Cancel running job

#### GET /v1/jobs
List all jobs (with pagination)
```json
Response: {
  "jobs": [...],
  "total": 10,
  "page": 1,
  "page_size": 20
}
```

#### POST /v1/plan
Generate plan only (no execution)
```json
Request: {"prompt": "calculate ndvi"}
Response: {
  "plan": {...},
  "confidence": 0.95,
  "mode": "llm"
}
```

#### GET /v1/models
List available models
```json
Response: {
  "models": [
    {
      "id": "prithvi:v1.0",
      "name": "prithvi",
      "version": "v1.0",
      "available": true,
      "inputs": ["Raster"],
      "outputs": ["Features", "Embeddings"],
      "gpu_required": false,
      "memory_mb": 1024
    }
  ]
}
```

#### GET /v1/capabilities
List registry capabilities
```json
Response: {
  "capabilities": [
    {
      "id": "algorithms/ndvi",
      "name": "NDVI Calculator",
      "inputs": ["Raster"],
      "outputs": ["NDVI"],
      "tags": ["vegetation", "index"]
    }
  ]
}
```

#### GET /v1/metrics
Get system metrics
```json
Response: {
  "total_executions": 100,
  "successful_executions": 95,
  "total_llm_calls": 200,
  "total_llm_cost": 0.15,
  "cache_hit_rate": 0.65
}
```

**Keep Existing**:
- POST /flow (legacy, but keep for backward compatibility)
- POST /upload (file upload)

---

### 3. Enhanced Registry Metadata ⭐ Priority 2
**Time Estimate**: 1-2 hours

**Files to Modify**:
- `dta/registry.yaml` - Add metadata fields

**New Metadata Fields**:
```yaml
- id: algorithms/ndvi
  # ... existing fields ...
  metadata:
    cost_estimate: 0.001  # USD per execution
    avg_duration_ms: 500
    memory_mb: 256
    gpu_required: false
    version: "1.0.0"
    tags: [vegetation, index, satellite]
    category: "algorithm"
    stable: true
```

**Categories**:
- `input` - Data loaders
- `algorithm` - Processing algorithms
- `model` - ML models
- `post-processing` - Visualization/insights

---

### 4. Configuration Management ⭐ Priority 2
**Time Estimate**: 2-3 hours

**Files to Create**:
- `dta/dti/config/settings.py` - Configuration classes
- `dta/dti/config/dev.yaml` - Dev config
- `dta/dti/config/prod.yaml` - Production config
- `tests/test_phase4_config.py` - Config tests

**Configuration Structure**:
```python
@dataclass
class LLMConfig:
    gemini_api_key: str | None
    ollama_base_url: str
    default_provider: str
    temperature: float
    max_tokens: int

@dataclass
class ExecutionConfig:
    max_concurrent_jobs: int
    job_timeout_seconds: int
    max_steps_per_plan: int
    max_memory_mb: int

@dataclass
class CacheConfig:
    enabled: bool
    max_size: int
    default_ttl_seconds: int

@dataclass
class Settings:
    environment: str  # dev/prod
    llm: LLMConfig
    execution: ExecutionConfig
    cache: CacheConfig
    registry_path: Path
    data_dir: Path
    log_level: str
    log_format: str  # standard/json
```

**Features**:
- Environment variable override
- YAML file loading
- Validation on load
- Hot reload support (optional)

---

### 5. Comprehensive Testing ⭐ Priority 3
**Time Estimate**: 3-4 hours

**Test Files to Create**:
- `tests/test_phase4_jobs.py` - Job queue tests
- `tests/test_phase4_api.py` - API endpoint tests
- `tests/test_phase4_config.py` - Configuration tests
- `tests/test_phase4_integration.py` - Full flow tests

**Test Coverage**:

#### Job Queue Tests (12 tests):
- Job submission and retrieval
- Job status transitions
- Progress tracking
- Cancellation
- Result caching
- Cleanup policies
- Concurrent job execution
- Error handling

#### API Tests (15 tests):
- All new endpoints
- Request/response validation
- Error cases (404, 400, 500)
- Pagination
- Job lifecycle via API
- Legacy endpoint compatibility

#### Config Tests (6 tests):
- YAML loading
- Environment override
- Validation
- Default values
- Invalid config handling

#### Integration Tests (5 tests):
- Full async job flow
- Concurrent requests
- Large raster processing
- LLM fallback during execution
- Cache effectiveness

**Target**: 38 new tests for Phase 4

---

### 6. API Documentation ⭐ Priority 3
**Time Estimate**: 2-3 hours

**Files to Create**:
- `server/openapi.py` - OpenAPI schema generation
- `docs/API.md` - API documentation
- `docs/postman_collection.json` - Postman collection

**Documentation Includes**:
- OpenAPI 3.0 schema
- Request/response examples
- Authentication (if added)
- Rate limiting (if added)
- Error codes and messages
- Architecture diagrams

---

## Implementation Order

### Stage 1: Async Infrastructure (4-5 hours)
1. Create job queue system (`server/jobs.py`)
2. Add job models and schemas
3. Write basic job queue tests
4. Integrate with existing executor

### Stage 2: API Enhancement (3-4 hours)
1. Add new endpoints to `server/app.py`
2. Update response models
3. Add pagination support
4. Write API tests

### Stage 3: Configuration & Registry (3-4 hours)
1. Enhance registry metadata
2. Create configuration management
3. Add environment-based settings
4. Write config tests

### Stage 4: Testing & Documentation (3-4 hours)
1. Write comprehensive tests
2. Generate OpenAPI schema
3. Create API documentation
4. Create Postman collection

---

## What Will NOT Be Implemented

Based on PLAN.md Phase 4 scope:

❌ **Persistent Database** (SQLite/PostgreSQL)
- Keeping in-memory only for MVP
- Can add in future if needed

❌ **External Message Queue** (Celery/Redis)
- Using asyncio.Queue for simplicity
- Sufficient for single-instance deployment

❌ **WebSocket Streaming**
- Mentioned in PLAN.md but not critical
- Polling via GET /v1/jobs/{id} is sufficient

❌ **Advanced Security**
- No authentication/authorization
- No rate limiting
- Local development only

---

## Success Criteria

✅ Async job processing working
✅ All new API endpoints functional
✅ Configuration management complete
✅ Registry enhanced with metadata
✅ 38+ new tests passing (total: 112+)
✅ API documentation generated
✅ Backward compatibility maintained

---

## Dependencies

**New Python Packages**:
- None (using stdlib asyncio)

**Environment Variables**:
```bash
DTA_ENV=dev|prod
DTA_LOG_LEVEL=INFO
DTA_LOG_FORMAT=standard|json
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Risks and Mitigations

**Risk 1**: Memory limits with in-memory job queue
- Mitigation: Implement job cleanup (1 hour retention)
- Mitigation: Max queue size limit

**Risk 2**: Long-running jobs blocking workers
- Mitigation: Job timeout configuration
- Mitigation: Worker pool sizing

**Risk 3**: Breaking changes to existing API
- Mitigation: Keep `/flow` endpoint for backward compatibility
- Mitigation: Version API endpoints with `/v1/` prefix

---

## Phase 4 Timeline

**Total Estimate**: 12-18 hours

1. **Day 1** (6-8 hours): Job queue + API endpoints
2. **Day 2** (6-8 hours): Config + Registry + Tests
3. **Day 3** (2-3 hours): Documentation + polish

**Milestone**: Production-ready system with async processing
