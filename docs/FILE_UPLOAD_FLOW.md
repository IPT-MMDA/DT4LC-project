# File Upload Flow

How files are uploaded and used in DT4LC analysis pipelines.

## Overview

DT4LC accepts GeoTIFF file uploads via the `/v1/upload` endpoint. Uploaded files can then be used in analysis jobs.

## Upload Flow

```text
1. Frontend uploads file
   POST /v1/upload (multipart/form-data)
   ↓
2. Backend validates and saves file
   → /tmp/dt4lc_uploads/{id}_{filename}
   ↓
3. Backend returns file metadata
   {id, path, size, crs, bounds, preview}
   ↓
4. Frontend submits job with attachment
   POST /v1/jobs {prompt, attachments: [{id, path}]}
   ↓
5. Orchestrator injects path into plan
   input/file.binds.RasterPath = path
   ↓
6. Executor loads and processes file
```

## API Usage

### 1. Upload File

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

### 2. Submit Job with Attachment

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "calculate ndvi",
    "attachments": [{
      "id": "a1b2c3d4",
      "filename": "data.tif",
      "path": "/tmp/dt4lc_uploads/a1b2c3d4_data.tif"
    }]
  }'
```

### 3. Check Job Status

```bash
curl http://localhost:8000/v1/jobs/{job_id}
```

## Multi-Turn Context

When a file is uploaded in a chat session, subsequent messages can reference it without re-uploading:

```text
User: [uploads image.tif] "Detect agricultural parcels"
→ Job runs with image.tif

User: "Now calculate NDVI on this data"
→ Job automatically uses image.tif from previous context
```

The frontend maintains chat session context including:

- Previous file attachments with server paths
- Conversation history
- Job results for reference

## File Storage

- **Location:** `/tmp/dt4lc_uploads/`
- **Format:** `{file_id}_{original_filename}`
- **Supported:** GeoTIFF (.tif, .tiff)

## Validation

Files are validated on upload:

- Extension must be `.tif` or `.tiff`
- Must be valid GeoTIFF (rasterio can open)
- Must not be empty

## Security Notes

- Server controls file paths (frontend cannot specify arbitrary paths)
- Files saved to controlled temp directory
- File size limits can be configured

## Future Improvements

- Automatic cleanup (TTL-based deletion)
- Additional format support (GeoJSON, Shapefile)
- File size limits
- Rate limiting on uploads
