# DT4LC API Reference

Complete REST API documentation for the DT4LC backend.

**Base URL:** `http://localhost:8000`

This page describes the main routes. The generated reference covers every route, including those not described here (`POST /v1/chat`, `GET /v1/files`, `GET /v1/tiles/{z}/{x}/{y}`, `/v1/gee/*` and `/v1/ml-models/*`): Swagger UI at `/docs`, ReDoc at `/redoc` and the OpenAPI schema at `/openapi.json` on the backend (<http://localhost:8000/docs>), or through nginx under Docker Compose (<http://localhost/docs>).

## Authentication

Currently no authentication required (local development only).

## Endpoints

### Health & Status

#### GET /v1/health

Health check endpoint.

**Query Parameters:**

| Parameter  | Type    | Default | Description                                                     |
|------------|---------|---------|----------------------------------------------------------------|
| `detailed` | boolean | false   | Return extended diagnostics (restricted to localhost requests) |

**Response (simple):**

```json
{
  "ok": true,
  "service": "DT4LC",
  "version": "0.5.0",
  "llm_providers": null,
  "gee": null,
  "models": null,
  "disk": null
}
```

**Response (`?detailed=true`, localhost only — returns 403 from remote):**

```json
{
  "ok": true,
  "service": "DT4LC",
  "version": "0.5.0",
  "llm_providers": [
    { "name": "gemini", "model": "gemini-2.5-flash", "available": true },
    { "name": "groq",   "model": "llama-3.3-70b-versatile", "available": true },
    { "name": "ollama", "model": "llama3.2", "available": true }
  ],
  "gee": {
    "initialized": false,
    "service_account_configured": false
  },
  "models": {
    "total": 3,
    "available": 2,
    "models": [
      { "id": "prithvi:v1.0", "available": true },
      { "id": "algorithms/ndvi", "available": true }
    ]
  },
  "disk": {
    "uploads": { "bytes": 1048576, "human": "1.0 MB" },
    "cache":   { "bytes": 5242880, "human": "5.0 MB" },
    "models":  { "bytes": 0,       "human": "0.0 B"  },
    "exports": { "bytes": 0,       "human": "0.0 B"  }
  }
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
  "prompt": "calculate ndvi",
  "attachments": [
    {
      "id": "983694a1",
      "filename": "hlsl_20230601.tif",
      "path": "/app/resources/.cache/uploads/983694a1_hlsl_20230601.tif",
      "mime_type": "image/tiff"
    }
  ]
}
```

`id`, `filename` and `path` come from the `POST /v1/upload` response.

**Response:** (202 Accepted, the job record)

```json
{
  "id": "34351192",
  "status": "pending",
  "prompt": "calculate ndvi",
  "plan": null,
  "progress": 0.0,
  "error": null,
  "created_at": "2026-10-21T08:40:19.964349",
  "started_at": null,
  "completed_at": null,
  "result": null
}
```

#### GET /v1/jobs/{job_id}

Get the full job record: status, plan and, once the job has finished, its `result` (`null` while the job is pending or running).

**Response (completed NDVI job on the bundled sample, trimmed; paths as under Docker Compose):**

```json
{
  "id": "c360c0df",
  "status": "completed",
  "prompt": "calculate ndvi",
  "plan": {
    "flow": "calculate ndvi",
    "steps": [
      {"uses": "input/file", "binds": {"RasterPath": "/app/resources/.cache/uploads/983694a1_hlsl_20230601.tif"}},
      {"uses": "algorithms/ndvi", "binds": {}},
      {"uses": "post-processing/agent-analysis", "binds": {}}
    ],
    "outputs": ["publish: chat"]
  },
  "progress": 1.0,
  "error": null,
  "created_at": "2026-10-21T08:36:07.125036",
  "started_at": "2026-10-21T08:36:07.128291",
  "completed_at": "2026-10-21T08:36:07.762783",
  "result": {
    "intent": "pipeline",
    "plan": {...},
    "execution": {
      "flow": "calculate ndvi",
      "steps": ["input/file", "algorithms/ndvi", "post-processing/agent-analysis"],
      "artifacts": {
        "RasterPath": "/app/resources/.cache/uploads/983694a1_hlsl_20230601.tif",
        "NDVIMap": {
          "metadata": {"crs": "EPSG:32636", "width": 768, "height": 768, "count": 5, ...},
          "statistics": {"min": -1.0, "max": 1.0, "mean": 0.5451, "std": 0.2704, "median": 0.6149, "valid_pixels": 589824, "total_pixels": 589824},
          "visualizations": {
            "ndvi_map": {
              "kind": "png",
              "path": "/app/resources/.cache/outputs/c360c0df/1_NDVIMap_ndvi_map.png",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2Fc360c0df%2F1_NDVIMap_ndvi_map.png",
              "stats": {}
            }
          },
          "path": "/app/resources/.cache/uploads/983694a1_hlsl_20230601.tif",
          "index_type": "ndvi",
          "rasters": {
            "ndvi": {
              "kind": "geotiff",
              "path": "/app/resources/.cache/outputs/c360c0df/1_NDVIMap_ndvi.tif",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2Fc360c0df%2F1_NDVIMap_ndvi.tif",
              "stats": {"min": -1.0, "max": 1.0, "mean": 0.5451, "valid_pixels": 589824}
            }
          }
        },
        "agent_result": {"summary": "...", "inputs": {...}, "llm_provider": "..."}
      },
      "outputs": {}
    }
  }
}
```

The result holds no pixel data. Each raster is a `geotiff` reference in `rasters` and each image a `png` reference in `visualizations`, both `{kind, path, url, stats}`; fetch the file with a `GET` on its `url` (see [GET /v1/download](#get-v1download)). The reference fields, file names and the keys each algorithm returns are in [Job artifacts](ARCHITECTURE.md#job-artifacts). Jobs stored before v0.5.0 are returned as they were stored, with arrays (as nested lists) and base64 images in `result`.

**Response (conversation intent, trimmed):**

```json
{
  "id": "job_abc123",
  "status": "completed",
  "prompt": "what can we do next?",
  "result": {
    "intent": "conversation",
    "response": "Based on your uploaded data, you can: 1) Calculate NDVI...",
    "reason": "..."
  }
}
```

#### POST /v1/jobs/{job_id}/cancel

Cancel a pending or running job.

**Response:** (200 OK, the job record)

```json
{
  "id": "34351192",
  "status": "cancelled",
  "prompt": "calculate ndvi",
  "plan": null,
  "result": null,
  "progress": 0.2,
  "error": null,
  "created_at": "2026-10-21T08:40:19.964349",
  "started_at": "2026-10-21T08:40:19.967340",
  "completed_at": "2026-10-21T08:40:20.485006"
}
```

#### GET /v1/jobs

List job summaries with optional filtering and pagination. A summary is the `GET /v1/jobs/{job_id}` record without `result`; fetch a job by id for its result.

**Query Parameters:**

- `status`: Filter by status (pending, running, completed, failed, cancelled)
- `limit`: Max results (default: 20)
- `offset`: Skip first N results (default: 0)

**Example:** `GET /v1/jobs?status=completed&limit=10`

**Response:**

```json
{
  "jobs": [
    {
      "id": "c360c0df",
      "status": "completed",
      "prompt": "calculate ndvi",
      "plan": {...},
      "progress": 1.0,
      "error": null,
      "created_at": "2026-10-21T08:36:07.125036",
      "started_at": "2026-10-21T08:36:07.128291",
      "completed_at": "2026-10-21T08:36:07.762783"
    }
  ],
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

**Response (the bundled sample; paths as under Docker Compose):**

```json
{
  "id": "983694a1",
  "filename": "hlsl_20230601.tif",
  "path": "/app/resources/.cache/uploads/983694a1_hlsl_20230601.tif",
  "size": [768, 768],
  "crs": "EPSG:32636",
  "bounds": [453240.0, 5144820.0, 476280.0, 5167860.0],
  "preview_png_base64": "iVBORw0KGgoA...",
  "preview_url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Fuploads%2F983694a1_hlsl_20230601.png"
}
```

The grayscale preview is written next to the upload as `<id>_<file name stem>.png`; `preview_url` is its download URL. `preview_png_base64` holds the same PNG and is kept for compatibility: do not persist it or send it back in job submissions.

**Supported Formats:**

- GeoTIFF (.tif, .tiff)

#### GET /v1/download

Download a job artifact, upload preview or other generated file by its server path. The `url` of an artifact reference and the `preview_url` of an upload already have this form: `/v1/download?path=` + the URL-encoded absolute path.

```bash
curl -o ndvi.tif "http://localhost:8000/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2Fc360c0df%2F1_NDVIMap_ndvi.tif"
```

Only files under these directories are served: job outputs (`resources/.cache/outputs/`), uploads (`resources/.cache/uploads/`, or `DT4LC_UPLOADS_PATH`), temporary and model outputs (`resources/.cache/tmp/`) and `<system temp>/dt4lc_delineate`. The path is resolved first, so `..` cannot leave them. Any other path returns 403 (`error.code` `unauthorized`); a missing file returns 404. The content type follows the extension: `.tif`/`.tiff` `image/tiff`, `.png` `image/png`, `.gpkg` `application/geopackage+sqlite3`, `.json` `application/json`, anything else `application/octet-stream`.

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

HTTP errors return an `ErrorResponse` body:

```json
{
  "ok": false,
  "error": {
    "code": "not_found",
    "message": "Job xyz123 not found",
    "details": {}
  }
}
```

`error.code` follows the HTTP status:

| Status | `error.code` |
|--------|--------------|
| 400 | `bad_request` |
| 403 | `unauthorized` |
| 404 | `not_found` |
| 413 | `file_too_large` |
| 405, 409, 429 and other 4xx | `bad_request` |
| 5xx | `internal_error` |

When planning fails on `POST /v1/plan` or `POST /v1/execute`, the response is 400 with code `planning_failed`; the rejected plan, if any, is in `error.details.candidate`.

Request validation errors (422) keep FastAPI's format, a `detail` array:

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "prompt"], "msg": "Field required", "input": {}}
  ]
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
