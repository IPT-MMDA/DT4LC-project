# DT4LC API Reference

Complete REST API documentation for the DT4LC backend.

**Base URL:** `http://localhost:8000`

This page describes the main routes. The generated reference covers every route, including those not described here (`GET /v1/files`, `GET /v1/tiles/{z}/{x}/{y}`, `/v1/gee/*` and `/v1/ml-models/*`): Swagger UI at `/docs`, ReDoc at `/redoc` and the OpenAPI schema at `/openapi.json` on the backend (<http://localhost:8000/docs>), or through nginx under Docker Compose (<http://localhost/docs>).

## Sessions

There is no sign-in yet. The first request without a valid `dt4lc_session` cookie gets one: signed with `SESSION_SECRET` (a random secret per server start if it is unset), `HttpOnly`, `SameSite=Lax`, and `Secure` with `SESSION_COOKIE_SECURE=true`. It lasts `GUEST_SESSION_HOURS` (24). Every session is a guest, unless the server sets `DEV_ROLE=member` or `DEV_ROLE=operator` for development.

Jobs and uploads record the session id as `owner`. A session sees only its own: another session's job, its events, cancel and confirm, downloads and tiles answer 404, as for an unknown id. `GET /v1/jobs` lists only the caller's jobs, and `GET /v1/files` only the caller's uploads (Earth Engine exports are listed to everyone). Operators see everything, including jobs and uploads from before v0.6.0, which have no owner. Guest jobs, their outputs and guest uploads are removed by an hourly cleanup once they are older than `GUEST_SESSION_HOURS`.

The cookie only works same-origin (see [CORS](#cors)). With `curl`, keep it in a cookie jar; otherwise each call is a new guest that cannot see the job the previous call submitted:

```bash
curl -c cookies.txt -b cookies.txt http://localhost:8000/v1/session
```

## Endpoints

### Health & Status

#### GET /v1/health

Health check endpoint.

**Query Parameters:**

| Parameter  | Type    | Default | Description                                                     |
|------------|---------|---------|----------------------------------------------------------------|
| `detailed` | boolean | false   | Return extended diagnostics (restricted to localhost requests) |

**Response (simple, no LLM provider available):**

```json
{
  "ok": true,
  "service": "DT4LC",
  "version": "0.5.1",
  "assistant": {"connected": false, "offline": true, "provider": null, "model": null},
  "llm_providers": null,
  "gee": null,
  "models": null,
  "disk": null
}
```

`assistant` is in every response. It names the first available provider in `LLM_PROVIDER_ORDER` (later ones are not probed), e.g. `{"connected": true, "offline": false, "provider": "gemini", "model": "gemini-3.5-flash-lite"}`. `offline` is always the opposite of `connected`. A cloud provider counts as available when its API key is set; the key itself is not tested. The answer is cached for 30 s.

**Response (`?detailed=true`, localhost only — returns 403 from remote):**

```json
{
  "ok": true,
  "service": "DT4LC",
  "version": "0.5.1",
  "assistant": {"connected": true, "offline": false, "provider": "gemini", "model": "gemini-3.5-flash-lite"},
  "llm_providers": [
    { "name": "gemini", "model": "gemini-3.5-flash-lite", "available": true },
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

#### GET /v1/session

The caller's role, analyses in the last hour, retention and model-download permission, for the UI badge. Issues the cookie if the request has none, like any route.

**Response (guest, after one analysis):**

```json
{
  "role": "guest",
  "guest": true,
  "analyses": {"limit": 5, "used": 1, "remaining": 4, "resets_at": "2026-10-23T06:58:27+00:00"},
  "kept_hours": 24,
  "expires_at": "2026-10-24T05:58:02+00:00",
  "can_download_models": false
}
```

**Response (`DEV_ROLE=member`):**

```json
{
  "role": "member",
  "guest": false,
  "analyses": {"limit": null, "used": 0, "remaining": null, "resets_at": null},
  "kept_hours": null,
  "expires_at": "2026-10-24T06:00:18+00:00",
  "can_download_models": true
}
```

| Field | Meaning |
|-------|---------|
| `role` | `guest`, `member` or `operator` |
| `analyses.limit` | `GUEST_ANALYSES_PER_HOUR` for guests; `null` (no limit) for members and operators |
| `analyses.used` | Analyses started in the rolling last hour. For a guest, the larger of the session's count and the client IP's |
| `analyses.resets_at` | When `used` next drops by one (UTC); `null` if nothing was used or there is no limit |
| `kept_hours` | Hours a guest's jobs and uploads are kept (`GUEST_SESSION_HOURS`); `null` for members and operators |
| `expires_at` | When the session cookie expires (UTC) |
| `can_download_models` | Whether a job of this role may download missing model weights (`MODEL_DOWNLOAD_ON_DEMAND`: `members` by default, or `operators`, `all`, `none`) |

#### GET /v1/capabilities

What this server can do for the caller, tier by tier: the document the Capabilities page renders. Apart from the assistant probe it shares with `GET /v1/health` (cached for 30 s), it makes no network call: model states are file checks, and Earth Engine is a configuration check.

**Response (guest, assistant offline, no models installed, no Earth Engine; trimmed):**

```json
{
  "summary": {
    "deployment": "local",
    "parts": {
      "deployment": "local server",
      "role": "guest access",
      "assistant": "assistant offline",
      "models": "no models installed",
      "earth_engine": "no Earth Engine"
    },
    "line": "guest access · assistant offline · no models installed · no Earth Engine"
  },
  "assistant": {"connected": false, "provider": null, "model": null, "display": null},
  "offline": true,
  "banner": "The assistant is offline on this server, so tiers 1 and 2 are unavailable. Built-in analyses run as usual; answers are limited to the catalog.",
  "tiers": [
    {
      "id": "ask", "number": 1, "title": "Ask",
      "subtitle": "Questions, clarifications, explanations of methods and results, follow-ups.",
      "status": "offline", "status_label": "Assistant offline — limited answers",
      "needs_assistant": true, "badge": null, "collapsed": false, "groups": null, "note": null,
      "items": [{"id": "ndwi-vs-ndvi", "prompt": "Why is NDWI better than NDVI for water?"}, ...]
    },
    {"id": "composed", "number": 2, "status": "offline", "badge": "Composed", ...},
    {
      "id": "builtin", "number": 3, "title": "Built-in analyses",
      "status": "available", "status_label": "Runs without the assistant", "needs_assistant": false,
      "groups": [
        {"id": "spectral", "title": "Spectral indices"},
        {"id": "change", "title": "Change, classification, statistics"},
        {"id": "models", "title": "Model-based"}
      ],
      "items": [
        {
          "id": "algorithms/ndvi", "group": "spectral", "title": "NDVI", "formula": "(NIR−Red)/(NIR+Red)",
          "description": "Vegetation density and health.", "inputs": ["Red", "NIR"],
          "outputs": ["Index map", "legend", "stats"], "time_estimate": "~10 s",
          "runs_without_assistant": true, "prompt": "Map vegetation health across the region",
          "model": null, "status": "available", "status_label": "Runs without the assistant"
        },
        ...,
        {
          "id": "models/prithvi-reconstruction", "group": "models", "title": "Prithvi reconstruction", ...,
          "model": {"id": "prithvi-eo-v1-100m", "title": "Prithvi EO v1 (100M)", "short": "Prithvi EO v1"},
          "status": "not_available_for_guests", "status_label": "Not available for guests — ask an operator"
        }
      ],
      ...
    },
    {
      "id": "models", "number": 4, "status": "available", ...,
      "items": [
        ...,
        {
          "id": "prithvi-eo-v1-100m", "title": "Prithvi EO v1 (100M)",
          "description": "NASA/IBM foundation model; reconstructs masked or cloudy pixels.",
          "size_bytes": 471859200, "size_label": "450 MB", "download_time": "~75 s",
          "licence": "Apache-2.0", "licence_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
          "used_by": ["models/prithvi-reconstruction"],
          "status": "not_available_for_guests", "status_label": "Not available for guests — ask an operator"
        }
      ]
    },
    {
      "id": "data", "number": 5, "status": "available",
      "note": "Guests: up to 5 analyses per hour, results kept 24 h in this browser, no account needed.", ...,
      "items": [
        {"id": "uploads", "title": "Uploads", "detail": "GeoTIFF, up to 200 MB per file · any projection", "status": "always", "status_label": "Always"},
        {"id": "sample", "title": "Bundled sample", "detail": "kahovka_data · HLS L30 · 2023-06-01 → 06-09", "status": "always", "status_label": "Always"},
        {"id": "earth-engine", "title": "Earth Engine · Sentinel-2 / Landsat / MODIS", "detail": "Fetch by region, dates, bands and cloud filter from the Map", "status": "not_configured", "status_label": "Not configured"}
      ]
    },
    {"id": "coming", "number": 6, "status": "coming", "collapsed": true, "items": [{"id": "models/ndvi-convlstm", "title": "ConvLSTM NDVI forecast", "description": "..."}, ...], ...}
  ],
  "guest_limits": {"analyses_per_hour": 5, "kept_hours": 24},
  "session": {"role": "guest", "guest": true, "analyses": {...}, ...},
  "example_prompts": [
    {"tier": "ask", "id": "ndwi-vs-ndvi", "prompt": "Why is NDWI better than NDVI for water?"},
    {"tier": "composed", "id": "any-index", "prompt": "Map moisture change with NDMI between the two dates"},
    {"tier": "builtin", "id": "algorithms/ndvi", "prompt": "Map vegetation health across the region"},
    ...
  ]
}
```

- `summary.line` joins `parts`; the deployment appears only when `DT4LC_DEPLOYMENT=demo` ("public demo"). With an assistant connected, `parts.assistant` reads e.g. "Gemini 3.5 Flash-Lite connected" and `assistant.display` is "Gemini 3.5 Flash-Lite".
- `tiers` always has six entries, in this order. Each tier is `{id, number, title, subtitle, status, status_label, needs_assistant, badge, collapsed, groups, note, items}`. `status` is `connected` or `offline` for `ask`, `available` or `offline` for `composed`, `available` for `builtin`, `models` and `data`, and `coming` for `coming`.
- Items per tier: `ask` `{id, prompt}`; `composed` `{id, title, description, recipe, badge, needs_assistant, needs_label, prompt}`; `builtin` as above; `models` as above; `data` `{id, title, detail, status, status_label}`; `coming` `{id, title, description}`.
- Model status: `installed`; `downloads_on_first_use` (this role may download it, `status_label` "Downloads on first use · 450 MB · ~75 s · Apache-2.0"); `not_available_for_guests` (a guest, where members could download it); `not_installed` (any other case where this role may not download it). A built-in analysis that needs a model takes the model's status, or `installed` when the model is installed.
- Data status: `always`, or `not_bundled` when the sample is missing, for uploads and the sample; `configured` or `not_configured` for Earth Engine.
- `example_prompts` lists every item's `prompt` with its tier and id. The composed example is NDMI, which the bundled sample (bands B2–B6) supports.

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

### Conversation

#### POST /v1/chat

One Workspace message. The answer is either a markdown reply, which creates no job, or an analysis, submitted as by [POST /v1/jobs](#post-v1jobs). It replaces the earlier `{messages}` endpoint, which planned and ran the last message.

**Request:**

```json
{
  "message": "here",
  "history": [
    {"role": "user", "text": "Map water change between the two dates"},
    {"role": "assistant", "text": "I can perform change detection! Please upload two GeoTIFF images\n(before and after) and I'll analyze the differences."}
  ],
  "attachments": [
    {"id": "6774daf7", "filename": "hlsl_20230601.tif", "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif", "mime_type": "image/tiff", "metadata": {...}},
    {"id": "6774daf7", "filename": "hlsl_20230609.tif", "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230609.tif", "mime_type": "image/tiff", "metadata": {...}}
  ],
  "pending": {
    "prompt": "Map water change between the two dates",
    "item_id": "algorithms/change-detection",
    "needs": ["RasterPathBefore", "RasterPathAfter"],
    "asked_at": "2026-10-23T06:00:18.947035Z"
  }
}
```

| Field | Meaning |
|-------|---------|
| `message` | The user's text; may be empty when only files are attached |
| `history` | Earlier turns, oldest first, `{role: "user" \| "assistant", text}`. The server keeps the last 10, cuts each to 2000 characters and drops the oldest until the total is at most 8000. Send an analysis as an assistant turn with its summary, so a question about "the last result" can be answered |
| `attachments` | As in `POST /v1/jobs`, with `metadata`. The server does not remember earlier attachments: send the conversation's files again with each message |
| `region` | As in `POST /v1/jobs` |
| `pending` | The `pending` of the previous reply, if it had one |

**Response (JSON, a reply to "hello"):**

```json
{
  "kind": "reply",
  "text": "Hello! I'm the DT4LC assistant. I run land-cover analyses on satellite imagery: spectral indices, change detection between two dates, classification and field boundaries.\n\n- Attach a GeoTIFF (or send a region from the map) and describe the analysis, for example *Calculate NDVI*.\n- Ask **What can you do?** for the full catalog."
}
```

**Response (JSON, a reply that waits for a file):** the answer to "Calculate NDVI" without an attachment.

```json
{
  "kind": "reply",
  "text": "I'd be happy to calculate NDVI for you! Please upload a GeoTIFF image\nwith NIR and Red bands, and I'll analyze the vegetation health.",
  "pending": {"prompt": "Calculate NDVI", "item_id": "algorithms/ndvi", "needs": ["RasterPath"], "asked_at": "2026-10-23T05:59:32.917653Z"}
}
```

**Response (JSON, an analysis):** `job` is the record `POST /v1/jobs` returns.

```json
{
  "kind": "analysis",
  "job_id": "7e6e0538",
  "job": {
    "id": "7e6e0538",
    "status": "pending",
    "prompt": "Map water change between the two dates",
    "events": [{"seq": 1, "type": "intent", "data": {"text": "I'll compute NDWI for 1 June and 9 June from the sample, then difference them to map where water changed.", "t": 1792735218.984}}],
    ...
  }
}
```

**Response (`Accept: text/event-stream`):** the same answer as Server-Sent Events, in the frame format of [GET /v1/jobs/{job_id}/events](#get-v1jobsjob_idevents). A reply is a series of `reply_delta {text}` frames of six words each, then `reply_done {text, pending?}` with the whole text. An analysis is a single `analysis {job_id, job}` frame; follow `GET /v1/jobs/{job_id}/events` from there. The stream closes after the last frame.

```text
id: 1
event: reply_delta
data: {"text":"Hello! I'm the DT4LC assistant. I "}

id: 2
event: reply_delta
data: {"text":"run land-cover analyses on satellite imagery: "}

...

id: 10
event: reply_done
data: {"text":"Hello! I'm the DT4LC assistant. I run land-cover analyses on satellite imagery: ..."}
```

How a message is answered:

- Greetings, thanks, a bare "here" and capability questions ("What can you do?") get fixed replies from the server and the registry, even when an LLM provider is connected.
- Other questions about methods or results, without an analysis verb, file or region, go to the LLM with the history. Without a provider the reply starts "The assistant is offline on this server, so I can only describe the catalog." and lists the catalog.
- An analysis request without a file or region gets the registry's request for the file and a `pending` `{prompt, item_id, needs, asked_at}`. `needs` is `["RasterPath"]`, or `["RasterPathBefore", "RasterPathAfter"]` for two dates. Keep the latest reply's `pending` and send it with the next message; a reply without `pending` clears it. A `pending` older than 30 minutes is ignored.
- With a `pending`, a message that only points at the files ("here", "this one", none at all, or up to four words without an analysis verb, registry keyword or index name) runs the pending prompt once enough files (two when `needs` has two) or a region come with it, and returns `analysis`. With one file for a two-date analysis, the reply asks for both and keeps the `pending`.
- Conversation never creates a job and does not count against the guest quota. An analysis counts, as through `POST /v1/jobs`.

**Errors:** plain HTTP, before any stream starts: 422 for an invalid body (FastAPI's `detail` array), and 429 `GUEST_QUOTA` when the message would start an analysis beyond the guest limit (see [POST /v1/jobs](#post-v1jobs)).

### Job Management

#### POST /v1/jobs

Submit an async job for processing.

**Request:**

```json
{
  "prompt": "calculate ndvi",
  "attachments": [
    {
      "id": "6774daf7",
      "filename": "hlsl_20230601.tif",
      "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif",
      "mime_type": "image/tiff",
      "metadata": {"sensor": "HLS L30", "date": "2023-06-01", "line": "HLS L30 · B2–B6 · 30 m · 2023-06-01", ...}
    }
  ]
}
```

`id`, `filename`, `path` and `metadata` come from the `POST /v1/upload` or `POST /v1/sample` response; `mime_type` is required. Send `metadata` as it came: the `intent` sentence takes the dates from it ("for 1 June").

**Response:** (202 Accepted, the job record)

```json
{
  "id": "501f9478",
  "status": "pending",
  "prompt": "calculate ndvi",
  "plan": null,
  "progress": 0.0,
  "error": null,
  "created_at": "2026-10-23T05:58:27.874641",
  "started_at": null,
  "completed_at": null,
  "owner": "8b5280fd5cdf0e24c6c6075dc45aff25",
  "result": null,
  "events": [{"seq": 1, "type": "intent", "data": {"text": "I'll compute NDVI for 1 June from the sample.", "t": 1792735107.916}}],
  "proposal": null,
  "region": null
}
```

Follow the job with [GET /v1/jobs/{job_id}/events](#get-v1jobsjob_idevents), or poll `GET /v1/jobs/{job_id}`.

**Proposals.** When planning composes a plan that is not a single built-in analysis (band math for an index such as NDMI, or a new combination of steps), the job does not run. After `planning` `started` the stream emits `proposal {text, plan}`, and the job waits at `status: "proposal"` with `proposal: {text, plan}` until [POST /v1/jobs/{job_id}/confirm](#post-v1jobsjob_idconfirm). The plan carries `composed: true`. For "Map moisture change with NDMI between the two dates" on the bundled sample:

```json
{
  "text": "I don't have a moisture-change algorithm. I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) from B5 and B6 for both dates and difference them — run that?",
  "plan": {
    "flow": "Map moisture change with NDMI between the two dates",
    "steps": [
      {"uses": "input/file-before", "binds": {"RasterPathBefore": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif"}, "config": {}},
      {"uses": "input/file-after", "binds": {"RasterPathAfter": "/app/resources/.cache/uploads/6774daf7_hlsl_20230609.tif"}, "config": {}},
      {"uses": "algorithms/band-math-change", "binds": {}, "config": {"formula": "(B5 - B6) / (B5 + B6)", "name": "NDMI", "threshold": 0.2}},
      {"uses": "post-processing/agent-analysis", "binds": {}, "config": {}}
    ],
    "outputs": ["publish: chat"],
    "composed": true
  }
}
```

The named indices NDMI, NBR, SAVI and NDBI are composed from the attached files' bands without the LLM. An index the file cannot support fails at planning instead: on the sample (B2–B6), "Map burn severity" fails with `BAND_MISSING`, because NBR needs B7. Cancelling a proposal ends its stream with `cancelled`; a guest's unconfirmed proposal is removed with the guest's other jobs and ends with `cancelled {reason: "expired"}`.

**Regions.** `region` is a region sent from the map. It is fetched from Earth Engine when no file is attached:

```json
{
  "prompt": "Calculate NDVI",
  "attachments": [],
  "region": {
    "bbox": [32.39, 46.46, 32.69, 46.66],
    "name": "Dnipro floodplain",
    "place": "Kherson",
    "dataset": "HLS L30",
    "start_date": "2023-06-01",
    "end_date": "2023-06-09"
  }
}
```

| Field | Meaning |
|-------|---------|
| `geometry` or `bbox` | Exactly one: a GeoJSON Polygon or MultiPolygon in WGS84, or `[west, south, east, north]` in degrees |
| `name`, `place` | Optional, at most 80 characters each; shown in the chip line |
| `dataset` | `HLS L30`, `HLS S30`, `Sentinel-2`, `Landsat 8/9` or `MODIS` |
| `start_date`, `end_date` | `YYYY-MM-DD`, start not after end. A single-date analysis fetches one median composite over the range; a change analysis fetches one around each date |
| `bands` | Band codes, e.g. `["B2", "B5"]`; default: the dataset's default bands |
| `cloud_max` | Maximum cloud cover in percent, 0–100, default 20 |

A region over 2,500 km², an unknown field or band, or a malformed geometry is a 422. The job echoes the region with `bands` resolved, `bounds`, `area_km2` and the chip `line`:

```json
{
  "geometry": null, "bbox": [32.39, 46.46, 32.69, 46.66], "name": "Dnipro floodplain", "place": "Kherson",
  "dataset": "HLS L30", "start_date": "2023-06-01", "end_date": "2023-06-09",
  "bands": ["B2", "B3", "B4", "B5", "B6", "B7"], "cloud_max": 20,
  "bounds": [32.39, 46.46, 32.69, 46.66], "area_km2": 511.4,
  "line": "Region · Dnipro floodplain · Kherson · 511 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud"
}
```

When files are attached as well, the files are analysed, the region is not fetched, and the echo has `note`: "Files are attached, so the analysis reads them and doesn't fetch this region." The fetch steps are `input` steps named like "Fetch HLS L30 · 2023-05-29 → 06-04". Without Earth Engine credentials a region job fails at planning with `EE_NOT_CONFIGURED`.

**Guest quota.** A guest may start `GUEST_ANALYSES_PER_HOUR` (5) analyses per rolling hour, counted per session and per client IP. Beyond that, `POST /v1/jobs` returns 429 with a `Retry-After` header (seconds) and no job is created:

```json
{
  "ok": false,
  "error": {
    "code": "GUEST_QUOTA",
    "message": "Guests can run 5 analyses per hour on this server. The next one is available at 06:58 UTC.",
    "details": {"limit": 5, "used": 5, "resets_at": "2026-10-23T06:58:27+00:00"}
  }
}
```

Every submission counts, including one that fails. One that ends in a proposal is not counted; its confirmation is. A guest who has used up the quota gets the 429 at submission, before planning. Members and operators are not limited. The counts are kept in memory, so a server restart resets them.

#### POST /v1/jobs/{job_id}/confirm

Run a job's proposal (status `proposal`) on the same job and event stream. The request has no body, or `{"plan": {...}}` with an edited plan, whose file inputs are replaced by the job's own attachments.

**Response:** 202 Accepted with the job record, `status: "pending"`. The stream continues with `planning` `ready` (`composed: true`), the steps, and `done` or `failed`. For the NDMI proposal above: `done` "4 steps · 1 s · NDMI", and a result whose `BandMathChangeMap` has `name: "NDMI"` and `formula: "(B5 - B6) / (B5 + B6)"`.

**Errors:**

| Status | When |
|--------|------|
| 404 | Unknown job, or another session's |
| 409 | The job is not a proposal (confirmed already, running, finished or cancelled), e.g. "Job d9252b95 is not a proposal waiting for confirmation (it is running)" |
| 422 | The edited plan is invalid: `error.code` `validation_error`, with the validator's message, e.g. "Step algorithms/band-math-change: Formula not allowed: \`abs(B5)\`. …" |
| 429 | `GUEST_QUOTA`: the confirmation counts against the guest quota |

#### GET /v1/jobs/{job_id}/events

The job's events as Server-Sent Events (`text/event-stream`): the log so far, then the live tail. Each frame has `id: <seq>`, `event: <type>` and `data: <JSON>`. An idle stream gets a `: keep-alive` comment every 15 s. The stream closes after a terminal event: `done`, `failed` or `cancelled`. The response sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so nginx passes frames through without buffering.

**Query Parameters and headers:**

| Parameter | Description |
|-----------|-------------|
| `after` | Replay only events with a larger `seq`; wins over `Last-Event-ID` |
| `Last-Event-ID` (header) | The same, as browsers send it when an `EventSource` reconnects |

A finished job replays its whole log (or the part after `after`) and closes. An unknown job, or another session's, is 404.

**Stream (NDVI on the bundled sample, 2023-06-01):**

```text
id: 1
event: intent
data: {"text":"I'll compute NDVI for 1 June from the sample.","t":1792735107.916}

id: 2
event: planning
data: {"phase":"started","t":1792735107.923}

id: 3
event: planning
data: {"phase":"ready","steps":[{"index":1,"id":"input/file","name":"Read hlsl_20230601.tif","kind":"input"},{"index":2,"id":"algorithms/ndvi","name":"NDVI","kind":"algorithm"},{"index":3,"id":"post-processing/agent-analysis","name":"Interpretation","kind":"postprocess"}],"composed":false,"t":1792735108.038}

id: 4
event: step_started
data: {"index":1,"id":"input/file","name":"Read hlsl_20230601.tif","t":1792735108.038}

id: 5
event: step_finished
data: {"index":1,"id":"input/file","elapsed_s":0.0,"t":1792735108.038}

id: 6
event: step_started
data: {"index":2,"id":"algorithms/ndvi","name":"NDVI","t":1792735108.038}

id: 7
event: step_finished
data: {"index":2,"id":"algorithms/ndvi","elapsed_s":0.54,"t":1792735108.573}

id: 8
event: step_started
data: {"index":3,"id":"post-processing/agent-analysis","name":"Interpretation","t":1792735108.573}

id: 9
event: step_finished
data: {"index":3,"id":"post-processing/agent-analysis","elapsed_s":0.0,"t":1792735108.575}

id: 10
event: done
data: {"elapsed_s":0.7,"summary":"3 steps · <1 s · NDVI","followups":["Use EVI for dense vegetation","Map open water with NDWI","Classify land cover in this image"],"t":1792735108.575}
```

Reconnecting with `?after=8` (or `Last-Event-ID: 8`) replays frames 9 and 10, then closes.

**Events.** Every `data` has `t` (Unix time in seconds) besides the fields below; an optional field is left out when it is unknown.

| Event | `data` | Meaning |
|-------|--------|---------|
| `intent` | `text` | One sentence of what will happen, written at submission from the registry and the template planner (no LLM call) |
| `planning` | `phase: "started"` | Planning has started |
| `proposal` | `text`, `plan` | Planning composed a plan; the job waits for [confirm](#post-v1jobsjob_idconfirm). Not terminal: the stream stays open |
| `planning` | `phase: "ready"`, `steps`, `composed` | The plan is ready. `steps` is `[{index, id, name, kind}]`, `index` 1-based, `kind` the registry kind (`input`, `algorithm`, `model`, `postprocess`, `preprocessor`) or `prepare` |
| `step_started` | `index`, `id`, `name` | A step has started |
| `step_progress` | `index`, `fraction?`, `message?`, `bytes?`, `total_bytes?`, `eta_s?` | Progress within a step |
| `step_finished` | `index`, `id`, `elapsed_s` | A step has finished |
| `done` | `elapsed_s`, `summary`, `followups` | Terminal. `summary` is one line, e.g. "3 steps · <1 s · NDVI"; `followups` are two or three full prompts for the chips under the result. Fetch the result with `GET /v1/jobs/{job_id}` |
| `failed` | `code`, `message`, `fixes`, `step_index?`, `raw` | Terminal; see below |
| `cancelled` | `at_step?`, `reason?` | Terminal. `at_step` is the step that was running when the job was cancelled; `reason: "expired"` for a proposal that was never confirmed |

When a model's weights must be downloaded first (members and operators, by default), a row `{index, id: "prepare/<model id>", name: "Preparing model · Prithvi EO v1 (100M)", kind: "prepare", model_id}` comes before the step that needs it, and the later rows' indices shift by one. Its `step_progress` carries `fraction`, `bytes`, `total_bytes`, `eta_s` and `message` ("Downloading Prithvi EO v1 (100M) · 450 MB"). `done.summary` counts plan steps only, not the preparation row.

Other sequences:

- A composed plan: `intent` → `planning` started → `proposal`; after confirm, `planning` ready (`composed: true`) → steps → `done`. The NDMI example ends with `done` "4 steps · 1 s · NDMI".
- A refusal at planning (a missing band for a named index, a region without Earth Engine, a model a guest may not download): `intent` → `planning` started → `failed`, with no `ready` row list.
- A cancel mid-run: the step in progress runs to completion, but nothing more is logged after `cancelled`, e.g. `intent` → `planning` ×2 → `step_started` → `step_finished` → `step_started` → `cancelled {at_step: 2}`.

**Failures.** `failed` data, e.g. "Map burn severity between the two dates" on the sample:

```json
{
  "code": "BAND_MISSING",
  "message": "Burn severity uses NBR = (NIR − SWIR2)/(NIR + SWIR2), which needs band B7 (SWIR2). The sample has B2–B6, so I can't compute it here.",
  "fixes": [
    {"label": "Map moisture change with NDMI instead", "kind": "prompt", "prompt": "Map moisture change with NDMI between the two dates"},
    {"label": "Use my own image", "kind": "upload"}
  ],
  "raw": "Burn severity uses NBR = (NIR − SWIR2)/(NIR + SWIR2), which needs band B7 (SWIR2). The sample has B2–B6, so I can't compute it here.",
  "t": 1792735153.055
}
```

`message` is at most 500 characters, without the executor's "Step … failed:" prefix. `raw` is the error text, up to 2000 characters, with API-key-like strings redacted. `step_index` is the step that failed, if one was running.

`fixes` is a list of 0–3 buttons, `{label, kind, prompt?, params?}`:

| `kind` | What the client does |
|--------|----------------------|
| `prompt` | Puts `prompt` in the composer, or sends it, as a new request |
| `retry` | Sends the same prompt and attachments again, with `params` merged into the request |
| `upload` | Opens the file picker, then retries with the new file |
| `open_map` | Opens the map to pick a region |
| `sample` | Attaches the bundled Kakhovka pair instead of the region (see [POST /v1/sample](#post-v1sample)) and sends the same prompt again |

| `code` | When | `fixes` |
|--------|------|---------|
| `INPUT_MISSING` | No input file for the analysis | "Upload a GeoTIFF" (`upload`), "Pick a region on the map" (`open_map`) |
| `EE_NOT_CONFIGURED` | A region job on a server without Earth Engine credentials: "Earth Engine isn't configured on this server, so I can't fetch imagery for this region." | "Use my own image" (`upload`), "Try the Kakhovka sample" (`sample`) |
| `BAND_MISSING` | The file lacks a band the analysis needs. Prithvi needs six HLS bands ("Prithvi needs six HLS bands (B2–B7); this file has five (B2–B6, no B7)."); NBR needs B7 (above); a band-math formula names a band the file lacks | Prithvi: "Use my own image" (`upload`), "Run NDWI change instead" (`prompt`). NBR: as above. Band math: "Use my own image" (`upload`), "What can this file do?" (`prompt`) |
| `MODEL_GATED` | The model is not installed and the job owner's role may not download it: "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator." | "See what I can do" (`prompt` "What can you do?"), "Use a built-in analysis instead" (`prompt`) |
| `MODEL_NOT_INSTALLED` | The model is not installed and no role may download it (`MODEL_DOWNLOAD_ON_DEMAND=none`): "… isn't installed on this server (450 MB) — ask an operator to install it." | As for `MODEL_GATED` |
| `INTERRUPTED` | The server restarted while the job was running | "Retry" (`retry`) |
| `PLAN_FAILED` | Planning failed. "I can't map this request to an analysis I can run here: …" when no step of the plan is an analysis | "See what I can do", then "Use a built-in analysis instead" for "I can't map this request…", else "Retry" |
| `INTERNAL` | Anything else | "Retry" (`retry`) |

`GUEST_QUOTA` is not an event code: the quota refuses with HTTP 429 before a job exists.

#### GET /v1/jobs/{job_id}

Get the full job record: status, plan, the event log (`events`, the list the event stream replays) and, once the job has finished, its `result` (`null` while the job is pending or running). `status` is `pending`, `proposal`, `running`, `completed`, `failed` or `cancelled`.

**Response (completed NDVI job on the bundled sample, trimmed; paths as under Docker Compose):**

```json
{
  "id": "501f9478",
  "status": "completed",
  "prompt": "calculate ndvi",
  "plan": {
    "flow": "calculate ndvi",
    "steps": [
      {"uses": "input/file", "binds": {"RasterPath": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif"}, "config": {}},
      {"uses": "algorithms/ndvi", "binds": {}, "config": {}},
      {"uses": "post-processing/agent-analysis", "binds": {}, "config": {}}
    ],
    "outputs": ["publish: chat"],
    "composed": false
  },
  "progress": 1.0,
  "error": null,
  "created_at": "2026-10-23T05:58:27.874641",
  "started_at": "2026-10-23T05:58:27.920396",
  "completed_at": "2026-10-23T05:58:28.574690",
  "owner": "8b5280fd5cdf0e24c6c6075dc45aff25",
  "result": {
    "intent": "pipeline",
    "plan": {...},
    "execution": {
      "flow": "calculate ndvi",
      "steps": ["input/file", "algorithms/ndvi", "post-processing/agent-analysis"],
      "artifacts": {
        "RasterPath": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif",
        "NDVIMap": {
          "metadata": {"crs": "EPSG:32636", "width": 768, "height": 768, "count": 5, ...},
          "statistics": {"min": -1.0, "max": 1.0, "mean": 0.5451, "std": 0.2704, "median": 0.6149, "valid_pixels": 589824, "total_pixels": 589824},
          "visualizations": {
            "ndvi_map": {
              "kind": "png",
              "path": "/app/resources/.cache/outputs/501f9478/1_NDVIMap_ndvi_map.png",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi_map.png",
              "stats": {}
            }
          },
          "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif",
          "index_type": "ndvi",
          "rasters": {
            "ndvi": {
              "kind": "geotiff",
              "path": "/app/resources/.cache/outputs/501f9478/1_NDVIMap_ndvi.tif",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi.tif",
              "stats": {"min": -1.0, "max": 1.0, "mean": 0.5451, "valid_pixels": 589824}
            }
          }
        },
        "agent_result": {"summary": "...", "inputs": {...}, "llm_provider": "..."}
      },
      "outputs": {}
    },
    "followups": ["Use EVI for dense vegetation", "Map open water with NDWI", "Classify land cover in this image"]
  },
  "events": [
    {"seq": 1, "type": "intent", "data": {"text": "I'll compute NDVI for 1 June from the sample.", "t": 1792735107.916}},
    ...,
    {"seq": 10, "type": "done", "data": {"elapsed_s": 0.7, "summary": "3 steps · <1 s · NDVI", "followups": [...], "t": 1792735108.575}}
  ],
  "proposal": null,
  "region": null
}
```

The result holds no pixel data. Each raster is a `geotiff` reference in `rasters` and each image a `png` reference in `visualizations`, both `{kind, path, url, stats}`; fetch the file with a `GET` on its `url` (see [GET /v1/download](#get-v1download)). The reference fields, file names and the keys each algorithm returns are in [Job artifacts](ARCHITECTURE.md#job-artifacts); band math is described [below](#band-math). Jobs stored before v0.5.0 are returned as they were stored, with arrays (as nested lists) and base64 images in `result`.

`result.followups` repeats the `done` event's follow-up prompts. `proposal` is `{text, plan}` once planning has proposed a composed plan, and `region` is the region echo of a region job.

Change results report areas in km² when the raster's CRS is projected; otherwise `areas_km2`, `water` and `vegetation` are `null`. `ChangeMap.statistics` (and `BandMathChangeMap.statistics`) has `areas_km2: {increase, decrease, stable}`. For NDWI, `ChangeMap.statistics.water` is `{before_km2, after_km2, gained_km2, lost_km2, threshold: 0.2}`; for NDVI, `vegetation` has the same fields with `threshold: 0.3`. NDWI change on the bundled sample gives water 13.881 km² before and 170.039 km² after, 156.761 km² gained and 0.603 km² lost.

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

`POST /v1/jobs` still answers a conversational prompt with such a job. [POST /v1/chat](#post-v1chat) answers it without one.

#### POST /v1/jobs/{job_id}/cancel

Cancel a pending, proposed or running job. The status becomes `cancelled` at once, and the event stream ends with `cancelled`; a running job stops at the next step boundary.

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
  "completed_at": "2026-10-21T08:40:20.485006",
  "owner": "8b5280fd5cdf0e24c6c6075dc45aff25",
  "events": [..., {"seq": 3, "type": "cancelled", "data": {"t": 1792572020.485}}],
  "proposal": null,
  "region": null
}
```

#### GET /v1/jobs

List the caller's job summaries with optional filtering and pagination. A summary is the `GET /v1/jobs/{job_id}` record without `result`, `events`, `proposal` and `region`; fetch a job by id for those. Operators see every job.

**Query Parameters:**

- `status`: Filter by status (pending, proposal, running, completed, failed, cancelled)
- `limit`: Max results (default: 20)
- `offset`: Skip first N results (default: 0)

**Example:** `GET /v1/jobs?status=completed&limit=10`

**Response:**

```json
{
  "jobs": [
    {
      "id": "501f9478",
      "status": "completed",
      "prompt": "calculate ndvi",
      "plan": {...},
      "progress": 1.0,
      "error": null,
      "created_at": "2026-10-23T05:58:27.874641",
      "started_at": "2026-10-23T05:58:27.920396",
      "completed_at": "2026-10-23T05:58:28.574690",
      "owner": "8b5280fd5cdf0e24c6c6075dc45aff25"
    }
  ],
  "total": 50,
  "limit": 10,
  "offset": 0
}
```

### Band math

Two registry components compute an index from a formula: `algorithms/band-math` (input `RasterPath`, output `BandMathMap`) and `algorithms/band-math-change` (inputs `RasterPathBefore` and `RasterPathAfter`, output `BandMathChangeMap`). Planning composes them for the named indices (NDMI, NBR, SAVI, NDBI), the LLM planner may emit them with a formula, and an edited plan sent to [confirm](#post-v1jobsjob_idconfirm) may use them. The step's `config`:

| Key | Meaning |
|-----|---------|
| `formula` | The expression, at most 256 characters, e.g. `"(B5 - B6) / (B5 + B6)"` |
| `name` | Display name, e.g. `"NDMI"`; default: the formula |
| `threshold` | Change mode only: a difference beyond ±`threshold` counts as an increase or decrease; positive, default 0.2 |

**Formula grammar.** The formula is parsed into a syntax tree and evaluated node by node; it is never passed to `eval`. It may contain:

- numbers (`0.5`, `2`);
- band codes, any case: `B5`, `B8A`, `SR_B5`, resolved through the band descriptions (only when no band has a description, `Bn` means band `n`);
- band roles: `red`, `green`, `blue`, `nir`, `swir` (SWIR1; there is no `swir2` role, so NBR needs the code `B7`);
- binary `+ - * /`, unary `-` and parentheses.

Anything else (a function call, `**`, a comparison, an unknown name) is rejected, as is a formula without a band. Bands are scaled to 0–1 reflectance first. A pixel that is nodata in a band the formula uses, or where the result is not finite (a zero denominator), is NaN. A normalized difference `(a - b) / (a + b)` is clipped to [-1, 1] and displayed on that range; other formulas are displayed on the 2nd–98th percentile of the data.

**Results.** `BandMathMap`: `index_type: "band-math"`, `formula`, `name`, `metadata`, `statistics` (`min`, `max`, `mean`, `std`, `median`, `valid_pixels`, `total_pixels`, `display_range`, `display_range_source`), `rasters.index` and `visualizations.index_map`. `BandMathChangeMap` (the difference is after minus before): `formula`, `name`, `statistics` (`threshold`, `min_change`, `max_change`, `mean_change`, `std_change`, `median_change`, `increase_pixels`, `decrease_pixels`, `stable_pixels`, `increase_area_km2`, `decrease_area_km2`, `areas_km2`, `display_range`, `display_range_source`), `index_statistics.before` and `.after`, `rasters.change`, `.index_before` and `.index_after`, and `visualizations.change_map`, `.index_before` and `.index_after`. NDMI change on the bundled sample at 0.2 gives 37.9 km² wetter and 42.0 km² drier.

**Errors** (in a plan, prefixed with "Step algorithms/band-math…: "):

- ``Formula not allowed: `abs(B5)`. Band math accepts only numbers, band names (B5, SR_B5, nir, swir, red, green, blue), + - * / and parentheses.``
- `Band B7 is not in this file (B2–B6). Available: B2, B3, B4, B5, B6, red, nir, green, blue, swir.`

### File Upload

#### POST /v1/upload

Upload a GeoTIFF file for analysis. The upload belongs to the caller's session.

**Request:** `multipart/form-data` with `file` field

```bash
curl -c cookies.txt -b cookies.txt -X POST http://localhost:8000/v1/upload \
  -F "file=@/path/to/image.tif"
```

**Response (the bundled sample; paths as under Docker Compose):**

```json
{
  "id": "35968520",
  "filename": "hlsl_20230601.tif",
  "path": "/app/resources/.cache/uploads/35968520_hlsl_20230601.tif",
  "size": [768, 768],
  "crs": "EPSG:32636",
  "bounds": [453240.0, 5144820.0, 476280.0, 5167860.0],
  "preview_png_base64": "iVBORw0KGgoA...",
  "preview_url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Fuploads%2F35968520_hlsl_20230601.png",
  "metadata": {
    "sensor": "HLS L30",
    "bands": [
      {"index": 1, "code": "B2", "description": "B2"},
      {"index": 2, "code": "B3", "description": "B3"},
      {"index": 3, "code": "B4", "description": "B4"},
      {"index": 4, "code": "B5", "description": "B5"},
      {"index": 5, "code": "B6", "description": "B6"}
    ],
    "dtype": "int16",
    "resolution_m": 30.0,
    "date": "2023-06-01",
    "size_bytes": 4180159,
    "size_label": "4.0 MB",
    "crs": "EPSG:32636",
    "bounds_wgs84": [32.388779, 46.455331, 32.691118, 46.663879],
    "line": "HLS L30 · B2–B6 · 30 m · 2023-06-01"
  },
  "owner": "8b5280fd5cdf0e24c6c6075dc45aff25"
}
```

The grayscale preview is written next to the upload as `<id>_<file name stem>.png`; `preview_url` is its download URL. `preview_png_base64` holds the same PNG and is kept for compatibility: do not persist it or send it back in job submissions.

`metadata` is a best guess, and any field may be missing: the sensor (`HLS L30`, `HLS S30`, `Sentinel-2`, `Landsat 8/9`), band codes from the band descriptions, data type, pixel size in metres, acquisition date (from the GeoTIFF tags or the file name), size, and the bounds in degrees. `line` is the chip text without the size; the chip appends ` · ` and `size_label`. A partial line such as `"3 bands · 11.1 m"` is possible, and `line` is `""` when nothing could be read. `GET /v1/files` returns the same `metadata` for each file.

**Supported Formats:**

- GeoTIFF (.tif, .tiff)

#### POST /v1/sample

Attach the bundled Kakhovka sample (`resources/kahovka_data/`, HLS L30, 2023-06-01 and 2023-06-09) as the caller's uploads. No body. Each file becomes an upload of the session, with `metadata` and a true-colour `preview_url`, and expires with a guest session like any upload. Calling it again in the same session returns the same files.

**Response (trimmed):**

```json
{
  "name": "kakhovka_hls_l30",
  "line": "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09",
  "attachments": [
    {"id": "6774daf7", "filename": "hlsl_20230601.tif", "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif", "preview_png_base64": "", "metadata": {"date": "2023-06-01", ...}, ...},
    {"id": "6774daf7", "filename": "hlsl_20230609.tif", "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230609.tif", "preview_png_base64": "", "metadata": {"date": "2023-06-09", ...}, ...}
  ]
}
```

`attachments` are `POST /v1/upload` responses, oldest first, with an empty `preview_png_base64` (use `preview_url`). Show the pair as one chip with `name` and `line`, and send both attachments, with their `metadata` and `"mime_type": "image/tiff"`, on jobs and chat messages. A server without the sample answers 404 with `error.code` `SAMPLE_NOT_BUNDLED`.

#### GET /v1/download

Download a job artifact, upload preview or other generated file by its server path. The `url` of an artifact reference and the `preview_url` of an upload already have this form: `/v1/download?path=` + the URL-encoded absolute path.

```bash
curl -b cookies.txt -o ndvi.tif "http://localhost:8000/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi.tif"
```

Only files under these directories are served: job outputs (`resources/.cache/outputs/`, or `DT4LC_OUTPUTS_PATH`), uploads (`resources/.cache/uploads/`, or `DT4LC_UPLOADS_PATH`), temporary and model outputs (`resources/.cache/tmp/`, or `DT4LC_TEMP_PATH`), Earth Engine exports (`resources/.cache/gee_exports/`) and `<system temp>/dt4lc_delineate`. The path is resolved first, so `..` cannot leave them. Any other path returns 403 (`error.code` `unauthorized`); a missing file returns 404. A job's outputs and an upload are served to their owner only (and to operators); for another session they are 404. The content type follows the extension: `.tif`/`.tiff` `image/tiff`, `.png` `image/png`, `.gpkg` `application/geopackage+sqlite3`, `.json` `application/json`, anything else `application/octet-stream`.

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
| 404 | `not_found`; `SAMPLE_NOT_BUNDLED` from `POST /v1/sample` |
| 413 | `file_too_large` |
| 422 | `validation_error` (raised by a route, e.g. an invalid edited plan on confirm) |
| 429 | `GUEST_QUOTA` for the guest limit, with `details: {limit, used, resets_at}` and `Retry-After`; `bad_request` when the job queue is full |
| 405, 409 and other 4xx | `bad_request` |
| 5xx | `internal_error` |

When planning fails on `POST /v1/plan` or `POST /v1/execute`, the response is 400 with code `planning_failed`; the rejected plan, if any, is in `error.details.candidate`.

Request validation errors (422), such as a missing field or an invalid `region`, keep FastAPI's format, a `detail` array:

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

Fixed rules run first, without an LLM call. Capability questions, then greetings, thanks, a bare "here" and questions about methods without an analysis verb, file or region, are conversation. A file (or region) with an action keyword or a named index such as NDMI, or a clear action phrase, is a pipeline request; an imperative request without a file is conversation that asks for the file. Only the remaining prompts go to the LLM classifier, and if that call fails or returns no valid JSON the request is treated as a pipeline request. `POST /v1/chat` turns a conversation verdict into a reply and submits a pipeline request as a job.

**Pipeline examples:**

- "calculate ndvi"
- "detect field boundaries"
- "run change detection"

**Conversation examples:**

- "what can we do next?"
- "explain what NDVI means"
- "help me understand this result"

## Real-time Progress

Job progress streams as Server-Sent Events from [GET /v1/jobs/{job_id}/events](#get-v1jobsjob_idevents); `POST /v1/chat` can stream its reply the same way. There is no WebSocket endpoint.

## Rate Limits

Guests may start `GUEST_ANALYSES_PER_HOUR` (default 5) analyses per rolling hour; the next one gets 429 `GUEST_QUOTA` (see [POST /v1/jobs](#post-v1jobs)). Conversation replies and proposals do not count. Members and operators are not limited. There are no other rate limits.

## CORS

CORS allows the origins in `CORS_ORIGINS` (default `*`) without credentials (`allow_credentials=False`). A cross-origin caller therefore sends no session cookie, and each of its requests is a new guest. Serve the UI and the API on one origin, as nginx does under Docker Compose and the Vite dev server's `/v1` proxy does locally.
