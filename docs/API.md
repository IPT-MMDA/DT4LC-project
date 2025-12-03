# DT4LC API Reference

Complete REST API documentation for the DT4LC backend.

**Base URL:** `http://localhost:8000`

## Authentication

Currently no authentication required (local development only).

## Endpoints

### Health & Status

#### GET /v1/health

Health check endpoint.

**Response:**

```json
{
  "ok": true,
  "service": "DT4LC",
  "version": "1.0.0"
}
```

#### GET /v1/capabilities

List available algorithms, models, and components from the registry.

**Response:**

```json
{
  "instances": [
    {
      "id": "algorithms/ndvi",
      "kind": "algorithm",
      "keywords": ["ndvi", "vegetation", "index"],
      "inputs": ["RasterPath"],
      "outputs": ["NDVIMap", "Statistics"]
    }
  ]
}
```

#### GET /v1/models

List available ML models.

**Response:**

```json
{
  "models": [
    {
      "name": "prithvi",
      "version": "v1.0",
      "available": true,
      "inputs": ["Raster"],
      "outputs": ["Features", "Embeddings"],
      "gpu_required": false,
      "memory_mb": 1024
    }
  ],
  "count": 1
}
```

#### GET /v1/metrics

System-wide execution metrics.

**Response:**

```json
{
  "total_executions": 100,
  "successful_executions": 95,
  "total_llm_calls": 200,
  "total_llm_tokens": 15000,
  "total_llm_cost": 0.15,
  "llm_by_provider": {
    "gemini": {"calls": 150, "tokens": 10000},
    "ollama": {"calls": 50, "tokens": 5000}
  }
}
```

#### GET /v1/queue/stats

Job queue statistics.

**Response:**

```json
{
  "total_jobs": 50,
  "queue_size": 3,
  "workers": 3,
  "max_workers": 3,
  "by_status": {
    "completed": 45,
    "running": 2,
    "pending": 3
  }
}
```

### Job Management

#### POST /v1/jobs

Submit an async job for processing.

**Request:**

```json
{
  "prompt": "calculate ndvi on the uploaded image",
  "attachments": [
    {
      "id": "abc123",
      "filename": "image.tif",
      "path": "/tmp/dt4lc_uploads/abc123_image.tif"
    }
  ]
}
```

**Response:** (202 Accepted)

```json
{
  "id": "job_xyz789",
  "state": "pending",
  "prompt": "calculate ndvi on the uploaded image",
  "progress": 0.0,
  "created_at": "2025-01-15T10:30:00Z"
}
```

#### GET /v1/jobs/{job_id}

Get job status and results.

**Response (completed):**

```json
{
  "id": "job_xyz789",
  "state": "completed",
  "prompt": "calculate ndvi",
  "progress": 1.0,
  "intent": "pipeline",
  "result": {
    "plan": {
      "steps": [
        {"uses": "input/file", "binds": {"RasterPath": "/path/to/file.tif"}},
        {"uses": "algorithms/ndvi", "binds": {}},
        {"uses": "post-processing/agent-analysis", "binds": {}}
      ]
    },
    "execution": {
      "ok": true,
      "artifacts": {
        "ndvi_result": {"mean": 0.45, "min": -0.2, "max": 0.85},
        "agent_result": {"summary": "Vegetation health is moderate..."}
      }
    }
  },
  "created_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:05Z"
}
```

**Response (conversation intent):**

```json
{
  "id": "job_abc123",
  "state": "completed",
  "prompt": "what can we do next?",
  "intent": "conversation",
  "result": {
    "response": "Based on your uploaded data, you can: 1) Calculate NDVI..."
  }
}
```

#### POST /v1/jobs/{job_id}/cancel

Cancel a pending or running job.

**Response:** (200 OK)

```json
{
  "id": "job_xyz789",
  "state": "cancelled"
}
```

#### GET /v1/jobs

List jobs with optional filtering and pagination.

**Query Parameters:**

- `status`: Filter by status (pending, running, completed, failed, cancelled)
- `limit`: Max results (default: 20)
- `offset`: Skip first N results (default: 0)

**Example:** `GET /v1/jobs?status=completed&limit=10`

**Response:**

```json
{
  "jobs": [...],
  "total": 50,
  "limit": 10,
  "offset": 0
}
```

### File Upload

#### POST /v1/upload

Upload a GeoTIFF file for analysis.

**Request:** `multipart/form-data` with `file` field

```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@/path/to/image.tif"
```

**Response:**

```json
{
  "id": "abc123",
  "filename": "image.tif",
  "path": "/tmp/dt4lc_uploads/abc123_image.tif",
  "size": [1024, 1024],
  "crs": "EPSG:4326",
  "bounds": [30.0, 45.0, 31.0, 46.0],
  "preview_png_base64": "iVBORw0KGgo..."
}
```

**Supported Formats:**

- GeoTIFF (.tif, .tiff)

### Legacy/Sync Endpoints

#### POST /v1/plan

Generate execution plan without executing (sync).

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "calculate ndvi"}
  ]
}
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
      {"uses": "post-processing/agent-analysis", "binds": {}}
    ],
    "outputs": ["publish: chat"]
  }
}
```

#### POST /v1/execute

Generate plan and execute synchronously.

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "calculate ndvi"}
  ]
}
```

**Response:**

```json
{
  "ok": true,
  "plan": {...},
  "execution": {
    "ok": true,
    "artifacts": {...}
  }
}
```

## Error Responses

### 400 Bad Request

Invalid request format or parameters.

```json
{
  "detail": "Invalid request: missing required field 'prompt'"
}
```

### 404 Not Found

Resource not found.

```json
{
  "detail": "Job not found: xyz123"
}
```

### 429 Too Many Requests

Queue is full.

```json
{
  "detail": "Job queue is full, try again later"
}
```

### 500 Internal Server Error

Server-side error.

```json
{
  "detail": "Internal error: [error message]"
}
```

## Intent Classification

The system automatically classifies requests:

| Intent | Description | Returns |
|--------|-------------|---------|
| `pipeline` | Data processing request | Plan + execution results |
| `conversation` | Question or guidance | Text response |

**Pipeline examples:**

- "calculate ndvi"
- "detect field boundaries"
- "run change detection"

**Conversation examples:**

- "what can we do next?"
- "explain what NDVI means"
- "help me understand this result"

## WebSocket (Future)

Real-time progress updates via WebSocket are planned for future releases.

## Rate Limits

No rate limits enforced (local development mode).

## CORS

CORS is enabled for all origins in development mode.
