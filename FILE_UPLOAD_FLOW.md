# File Upload Flow

## Overview

The DT4LC API now supports file uploads from the frontend. Users can upload GeoTIFF files and submit jobs that process those files.

## Flow

### 1. Upload File

**Endpoint:** `POST /v1/upload`

**Request:**
```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@/path/to/data.tif"
```

**Response:**
```json
{
  "id": "a1b2c3d4",
  "filename": "data.tif",
  "path": "/tmp/dt4lc_uploads/a1b2c3d4_data.tif",
  "size": [1024, 1024],
  "crs": "EPSG:4326",
  "bounds": [30.0, 45.0, 31.0, 46.0],
  "preview_png_base64": "iVBORw0KGgo..."
}
```

**Key Fields:**
- `id`: Unique file identifier
- `path`: Server-side file path (use this in job submission)
- `preview_png_base64`: Base64-encoded PNG preview for display

### 2. Submit Job with File

**Endpoint:** `POST /v1/jobs`

**Request:**
```json
{
  "prompt": "calculate ndvi",
  "attachments": [
    {
      "id": "a1b2c3d4",
      "filename": "data.tif",
      "path": "/tmp/dt4lc_uploads/a1b2c3d4_data.tif"
    }
  ]
}
```

**Response:**
```json
{
  "id": "job123",
  "status": "pending",
  "prompt": "calculate ndvi",
  "attachments": [...],
  "progress": 0.0,
  "created_at": "2025-10-02T..."
}
```

### 3. Check Job Status

**Endpoint:** `GET /v1/jobs/{job_id}`

**Response (when completed):**
```json
{
  "id": "job123",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "plan": {
      "steps": [
        {"uses": "input/file", "binds": {"RasterPath": "/tmp/dt4lc_uploads/a1b2c3d4_data.tif"}},
        {"uses": "algorithms/ndvi"},
        {"uses": "post-processing/agent-analysis"}
      ]
    },
    "execution": { ... }
  },
  "completed_at": "2025-10-02T..."
}
```

## Architecture

### Components

1. **Upload Endpoint** (`/v1/upload`)
   - Validates GeoTIFF format
   - Saves file to temp directory (`UPLOAD_DIR`)
   - Generates preview image
   - Returns file metadata including `path`

2. **Job Submission** (`/v1/jobs`)
   - Accepts `attachments` array with file metadata
   - Stores attachments with job

3. **Orchestrator** (`orchestrate()`)
   - Receives `ChatRequest` with attachments
   - Injects attachment paths into `input/file` step binds
   - Example: `{"uses": "input/file", "binds": {"RasterPath": "/tmp/..."}}`

4. **Executor** (`PipelineExecutor`)
   - Reads `RasterPath` from step binds
   - Passes path to algorithms/models
   - Processes uploaded file

### Data Flow

```
Frontend Upload
    ↓
POST /v1/upload
    ↓
Save to /tmp/dt4lc_uploads/
    ↓
Return {id, path, ...}
    ↓
Frontend stores path
    ↓
POST /v1/jobs with attachments: [{path: "..."}]
    ↓
Job Queue stores attachments
    ↓
Worker calls orchestrate(req)
    ↓
Orchestrator injects path into input/file.binds
    ↓
Executor reads path from binds
    ↓
Algorithms process file
    ↓
Return results
```

## Frontend Integration

### Example React Code

```typescript
// 1. Upload file
const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/v1/upload', {
    method: 'POST',
    body: formData,
  });

  return await response.json();  // {id, path, preview_png_base64, ...}
};

// 2. Submit job with uploaded file
const submitJob = async (prompt: string, uploadedFile: UploadResponse) => {
  const response = await fetch('http://localhost:8000/v1/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      attachments: [{
        id: uploadedFile.id,
        filename: uploadedFile.filename,
        path: uploadedFile.path,
      }],
    }),
  });

  return await response.json();  // {id, status, ...}
};

// 3. Poll for results
const checkJobStatus = async (jobId: string) => {
  const response = await fetch(`http://localhost:8000/v1/jobs/${jobId}`);
  return await response.json();
};
```

## File Storage

- **Location:** `/tmp/dt4lc_uploads/`
- **Format:** `{file_id}_{original_filename}`
- **Cleanup:** Manual (no automatic cleanup yet - could add TTL in future)

## Security Considerations

1. **File Validation:**
   - Only `.tif` and `.tiff` extensions allowed
   - Validates GeoTIFF format with rasterio
   - Rejects empty files

2. **Path Injection:**
   - Server controls file paths (frontend can't specify arbitrary paths)
   - Files saved to controlled temp directory

3. **Future Improvements:**
   - Add file size limits
   - Implement automatic cleanup (e.g., delete files after 24 hours)
   - Add authentication/authorization
   - Rate limiting on uploads

## Testing

### Manual Test

```bash
# 1. Upload a file
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@test.tif" \
  | jq .

# Save the "path" from response

# 2. Submit job
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "calculate ndvi",
    "attachments": [{
      "id": "test123",
      "filename": "test.tif",
      "path": "/tmp/dt4lc_uploads/test123_test.tif"
    }]
  }' \
  | jq .

# 3. Check status (use job ID from response)
curl http://localhost:8000/v1/jobs/{job_id} | jq .
```

## Date

October 2, 2025
