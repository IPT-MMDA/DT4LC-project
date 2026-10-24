# DT4LC API Reference

Complete REST API documentation for the DT4LC backend.

**Base URL:** `http://localhost:8000`

This page describes the main routes. The generated reference covers every route, including those not described here (`/v1/gee/*` and most of `/v1/ml-models/*`): Swagger UI at `/docs`, ReDoc at `/redoc` and the OpenAPI schema at `/openapi.json` on the backend (<http://localhost:8000/docs>), or through nginx under Docker Compose (<http://localhost/docs>).

## Sessions

The first request without a valid `dt4lc_session` cookie gets one: signed with `SESSION_SECRET` (a random secret per server start if it is unset), `HttpOnly`, `SameSite=Lax`, and `Secure` with `SESSION_COOKIE_SECURE=true`. It lasts `GUEST_SESSION_HOURS` (24). A session is a guest until it signs in with GitHub (see [Sign-in](#sign-in)); then the cookie also carries the user, and the role comes from the allowlists on every request. `DEV_ROLE=guest|member|operator` overrides the role of every request, signed in or not, for development.

Jobs and uploads record an `owner`: the session id for a guest, and the stable user id `github:<numeric id>` once signed in, so a member sees the same items from any browser. A caller sees only its own: another owner's job, its events, cancel, confirm, delete and statistics, and another owner's downloads, tiles and files answer 404, as for an unknown id. `GET /v1/jobs` lists only the caller's jobs, and `GET /v1/files` only the caller's uploads (Earth Engine exports are listed to everyone). Operators see everything, including jobs and uploads from before v0.6.0, which have no owner. When a guest signs in, the guest session's jobs and uploads move to the user id (they are *adopted*). Guest jobs, their outputs and guest uploads are removed by a periodic cleanup (every `GUEST_CLEANUP_MINUTES`, default 60) once they are older than `GUEST_SESSION_HOURS`; members' and operators' items are kept.

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
  "version": "0.6.0",
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
  "version": "0.6.0",
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

The caller's role, signed-in user, analyses in the last hour, retention and model-download permission, for the UI badge and the account menu. Issues the cookie if the request has none, like any route.

**Response (guest, after one analysis):**

```json
{
  "role": "guest",
  "guest": true,
  "user": null,
  "sign_in": {"configured": false},
  "analyses": {"limit": 5, "used": 1, "remaining": 4, "resets_at": "2026-10-23T06:58:27+00:00"},
  "kept_hours": 24,
  "expires_at": "2026-10-24T05:58:02+00:00",
  "can_download_models": false
}
```

**Response (signed in, on `ALLOWED_EMAILS`):**

```json
{
  "role": "member",
  "guest": false,
  "user": {
    "id": "github:42",
    "login": "user42",
    "name": "User 42",
    "email": "Ana@Example.org",
    "avatar_url": "https://avatars.test/42"
  },
  "sign_in": {"configured": true},
  "analyses": {"limit": null, "used": 0, "remaining": null, "resets_at": null},
  "kept_hours": null,
  "expires_at": "2026-10-24T06:00:18+00:00",
  "can_download_models": true
}
```

With `DEV_ROLE=member` the response is the same with `"user": null`.

| Field | Meaning |
|-------|---------|
| `role` | `guest`, `member` or `operator` |
| `user` | The signed-in GitHub user `{id, login, name, email, avatar_url}`, or `null`. `id` (`github:<numeric id>`) is the `owner` of the user's jobs and uploads; `email` is the primary, verified GitHub address the role was decided from |
| `sign_in.configured` | Whether GitHub sign-in is configured on this server (`GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` and `OAUTH_REDIRECT_BASE` all set) |
| `analyses.limit` | `GUEST_ANALYSES_PER_HOUR` for guests; `null` (no limit) for members and operators |
| `analyses.used` | Analyses started in the rolling last hour. For a guest, the larger of the session's count and the client IP's |
| `analyses.resets_at` | When `used` next drops by one (UTC); `null` if nothing was used or there is no limit |
| `kept_hours` | Hours a guest's jobs and uploads are kept (`GUEST_SESSION_HOURS`); `null` for members and operators |
| `expires_at` | When the session cookie expires (UTC). A signed-in session lasts `GUEST_SESSION_HOURS` too; after that the browser is a guest until it signs in again |
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
        {"id": "sample", "title": "Bundled sample", "detail": "kakhovka_hls_l30 · HLS L30 · 2023-06-01 → 06-09", "status": "always", "status_label": "Always"},
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

### Sign-in

Members and operators sign in with GitHub; there is no password and no login wall. Sign-in is configured when `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` and `OAUTH_REDIRECT_BASE` (the app's public origin, no trailing slash) are all set; how to create the OAuth app is in [DEPLOY_DEMO.md](DEPLOY_DEMO.md#5-github-sign-in-the-oauth-app). `GITHUB_OAUTH_BASE` (default `https://github.com`) and `GITHUB_API_BASE` (default `https://api.github.com`) point the flow at a fake GitHub in tests and E2E.

The role comes from the primary GitHub email, if GitHub has verified it, compared case-insensitively with comma-separated allowlists: `OPERATOR_EMAILS` makes an operator; `ALLOWED_EMAILS`, or a domain in `ALLOWED_DOMAINS`, a member. Anyone else stays a guest. The lists are checked again on every request, so a changed role applies at once, and a user removed from every list gets a new guest session.

#### GET /v1/auth/status

```json
{"configured": false, "user": null, "role": "guest"}
```

`configured` is `sign_in.configured` of `GET /v1/session`, `user` its `user` and `role` its `role`.

#### GET /v1/auth/github/login

The browser opens this URL (the account menu's "Sign in with GitHub" is a link to it). It answers 302 to GitHub's authorize page with `scope=read:user user:email`, `redirect_uri=<OAUTH_REDIRECT_BASE>/v1/auth/github/callback` and a fresh `state`. The `state` is also in a signed `dt4lc_oauth_state` cookie, sent only to `/v1/auth/github` and valid for 10 minutes.

Without the three variables it answers 404:

```json
{"ok": false, "error": {"code": "SIGN_IN_NOT_CONFIGURED", "message": "Sign-in is not configured on this server", "details": {}}}
```

#### GET /v1/auth/github/callback

GitHub redirects here with `code` and `state` (or `error`). The server checks the state cookie, exchanges the code for a token, reads `GET /user` and `GET /user/emails`, and decides the role. It always answers 302 to the app, and deletes the state cookie:

| Outcome | Redirect |
|---------|----------|
| Allowed | `<OAUTH_REDIRECT_BASE>/app`, with a new session cookie (a new session id) that carries the user. If the caller was a guest, its jobs and uploads are adopted: their `owner` becomes `github:<id>` |
| Not on an allowlist | `/app?signin=denied&reason=not_allowlisted&email=<primary email>` |
| Primary email not verified | `/app?signin=denied&reason=email_unverified` |
| A missing or wrong `state`, an `error` from GitHub (e.g. the user cancelled), or a failed exchange | `/app?signin=error` |

A denied or failed sign-in leaves the caller's session as it was. Without the three variables the callback is 404 `SIGN_IN_NOT_CONFIGURED`, like the login.

#### POST /v1/auth/logout

Replaces the caller's session with a fresh guest session (a new cookie) and returns it as [GET /v1/session](#get-v1session) does. The signed-in user's jobs and uploads stay with the user id.

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
| 404 | Unknown job, or another owner's |
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

A finished job replays its whole log (or the part after `after`) and closes. An unknown job, or another owner's, is 404.

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
  "card": {"title": "NDVI · 2023-06-01", "analysis": {"id": "algorithms/ndvi", "label": "NDVI"}, "kind": "index", "thumbnail_url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi_map_bare.png&name=ndvi_2023-06-01_map.png", "duration_s": 0.7, "error": null},
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
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi_map.png&name=ndvi_2023-06-01.png",
              "stats": {},
              "size_bytes": 1495834,
              "download_name": "ndvi_2023-06-01.png"
            },
            "ndvi_map_bare": {
              "kind": "png",
              "path": "/app/resources/.cache/outputs/501f9478/1_NDVIMap_ndvi_map_bare.png",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi_map_bare.png&name=ndvi_2023-06-01_map.png",
              "stats": {"width": 768, "height": 768, "style": {"kind": "index", "range": [-0.2, 0.8], "ramp": ["#994c1a", "#cc9933", "#ffff66", "#99cc33", "#339933", "#006600"]}},
              "size_bytes": 1072560,
              "download_name": "ndvi_2023-06-01_map.png"
            }
          },
          "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230601.tif",
          "index_type": "ndvi",
          "rasters": {
            "ndvi": {
              "kind": "geotiff",
              "path": "/app/resources/.cache/outputs/501f9478/1_NDVIMap_ndvi.tif",
              "url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi.tif&name=ndvi_2023-06-01.tif",
              "stats": {"min": -1.0, "max": 1.0, "mean": 0.5451, "valid_pixels": 589824},
              "size_bytes": 2100449,
              "download_name": "ndvi_2023-06-01.tif"
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

The result holds no pixel data. Each raster is a `geotiff` reference in `rasters` and each image a `png` reference in `visualizations`, both `{kind, path, url, stats, size_bytes, download_name}`; fetch the file with a `GET` on its `url` (see [GET /v1/download](#get-v1download)). `size_bytes` is the file's size. `download_name` is a readable file name built from the analysis and the input dates (`ndvi_2023-06-01.tif`, `water-change_ndwi_2023-06-01_2023-06-09.tif`), unique within the job, and `url` ends in `&name=<download_name>`; the stored file keeps its name. Jobs finished before v0.6.0 have neither field, so fall back to the base name of `path`. A `*_bare` image is the map alone, without axes or legend, for cards and thumbnails; its `stats` has `width`, `height` and the `style` that the legend and the [map tiles](#get-v1tileszxy) use: `{kind: "change" | "index", range, ramp}` or `{kind: "class", classes}`. The reference fields, file names and the keys each algorithm returns are in [Job artifacts](ARCHITECTURE.md#job-artifacts); band math is described [below](#band-math). Jobs stored before v0.5.0 are returned as they were stored, with arrays (as nested lists) and base64 images in `result`.

`result.followups` repeats the `done` event's follow-up prompts. `proposal` is `{text, plan}` once planning has proposed a composed plan, and `region` is the region echo of a region job.

**Change results.** Change detection clips each date's index to its range ([-1, 1] for NDVI, NDWI, NDSI and EVI) before it takes the difference (after minus before), so the change lies in [-2, 2]. Areas are in km² when the raster's CRS is projected; otherwise `areas_km2`, `water`, `vegetation` and each class's `area_km2` are `null`. `ChangeMap.statistics` (and `BandMathChangeMap.statistics`) has `min_change`, `max_change`, `mean_change`, `std_change`, `median_change` and `areas_km2: {increase, decrease, stable}`. For NDWI, `ChangeMap.statistics.water` is `{before_km2, after_km2, gained_km2, lost_km2, threshold: 0.2}`: open water is NDWI > 0.2 on each date. For NDVI, `vegetation` has the same fields with `threshold: 0.3`.

`class_statistics` is the class table that the Result detail and the CSV show, keyed by class name, each `{id, name, colour, pixels, percent, area_km2, mean_change}`, where `percent` is the share of valid pixels and `mean_change` the mean difference in the class. For NDWI the classes are Water gained (open water on the second date only), Water lost and No change; for NDVI Vegetation gained, Vegetation lost and No change; for band-math NDMI Wetter, Drier and No change; for other indices Increase, Decrease and Stable. NDWI change on the bundled crop:

```json
{
  "Water gained": {"id": 1, "name": "Water gained", "colour": "#2d5f9e", "pixels": 174179, "percent": 29.53, "area_km2": 156.761, "mean_change": 0.806},
  "Water lost": {"id": 2, "name": "Water lost", "colour": "#8c4a2f", "pixels": 670, "percent": 0.11, "area_km2": 0.603, "mean_change": -0.233},
  "No change": {"id": 0, "name": "No change", "colour": "#c4cad6", "pixels": 414973, "percent": 70.36, "area_km2": 373.476, "mean_change": 0.1642}
}
```

with `water` `{"before_km2": 13.881, "after_km2": 170.039, "gained_km2": 156.761, "lost_km2": 0.603, "threshold": 0.2}`: open water grows from 13.9 km² to 170.0 km². These are the crop's numbers, the ones the app computes on the sample. The full 2906 × 1479 px scene that the landing image shows differs (open water 32 → 256 km², 428 km² gained and 43 km² lost, measured with ΔNDWI > 0.3), and is not in the repository. Land-cover and snow results have a `class_statistics` of the same shape without `mean_change`.

**Raster statistics.** `Statistics.bands` is keyed by the band code of each band's description (`B2` … `B6` on the sample, as upload metadata names them), else `band_<n>`. Each band has `min`, `max`, `mean`, `std`, `median`, `percentiles {p5, p25, p75, p95}`, `valid_pixels`, `total_pixels`, `nodata_pixels` and a `histogram`.

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

List the caller's job summaries, newest first, with optional filtering and pagination. A summary is the `GET /v1/jobs/{job_id}` record without `result`, `events`, `proposal` and `region`, but with the job's [`card`](#job-cards), so a list such as the Library needs no result; fetch a job by id for the rest. Operators see every job.

**Query Parameters:**

- `status`: Filter by status (pending, proposal, running, completed, failed, cancelled)
- `limit`: Max results (default: 20, at most 100)
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
      "owner": "8b5280fd5cdf0e24c6c6075dc45aff25",
      "card": {
        "title": "NDVI · 2023-06-01",
        "analysis": {"id": "algorithms/ndvi", "label": "NDVI"},
        "kind": "index",
        "thumbnail_url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi_map_bare.png&name=ndvi_2023-06-01_map.png",
        "duration_s": 0.7,
        "error": null
      }
    }
  ],
  "total": 50,
  "limit": 10,
  "offset": 0
}
```

`total` counts every job the query matches, not just this page: the caller's jobs (an operator's: all of them), and only those with `status` when it is given.

#### Job cards

Every job record and summary has `card`, the Library row and Result-detail header of the job. It is computed once, when the job completes, fails or is cancelled, and stored with the job, so `GET /v1/jobs` shows it without loading the result. It is `null` until then (pending, proposal, running), and on jobs from before v0.6.0: show the prompt instead.

| Field | Meaning |
|-------|---------|
| `title` | From the plan's main analysis step (its last algorithm or model step): "Water change · NDWI difference", "Moisture change · NDMI difference", "NDVI · 2023-06-01", "Raster statistics · 2023-06-09". A job that never got a plan (it failed at planning) is titled with a short form of its prompt |
| `analysis` | `{id, label}` of that step, e.g. `{"id": "algorithms/change-detection", "label": "Change detection"}`; for a job that failed at planning, the analysis its `intent` sentence announced; `null` if none |
| `kind` | `change`, `index`, `classification`, `fields`, `statistics`, `image`, or `composed` for a composed plan; `null` without an analysis |
| `thumbnail_url` | The main step's first `*_bare` map image, for completed jobs; else `null` |
| `duration_s` | `completed_at − started_at` in seconds; `null` if the job never started |
| `error` | `{code, message}` of the `failed` event, else `null` |

A region job that failed without Earth Engine:

```json
{
  "title": "Calculate NDVI",
  "analysis": {"id": "algorithms/ndvi", "label": "NDVI"},
  "kind": "index",
  "thumbnail_url": null,
  "duration_s": 0.0,
  "error": {"code": "EE_NOT_CONFIGURED", "message": "Earth Engine isn't configured on this server, so I can't fetch imagery for this region."}
}
```

#### DELETE /v1/jobs/{job_id}

Delete a finished job (completed, failed or cancelled) or a proposal, with its output files.

**Response:** `{"ok": true, "id": "cc76d13d"}`

A deleted proposal's open event stream ends with `cancelled {reason: "deleted"}`. A pending or running job is 409 (cancel it first); an unknown job or another owner's is 404.

#### GET /v1/jobs/{job_id}/statistics.csv

The main analysis's statistics as a CSV attachment named after the card title, `<slug of the title>-statistics.csv` (`job-<id>-statistics.csv` without a title), with `Content-Type: text/csv; charset=utf-8`:

- the class table, `class,area_km2,percent,pixels[,mean_change]`, for change detection, band-math change, land cover and snow;
- one row per band, `band,min,max,mean,std,median,p5,p25,p75,p95,valid_pixels,nodata_pixels`, for raster statistics;
- one row of index statistics, `index,min,max,mean,…`, for a single index.

NDWI change on the sample (`water-change-ndwi-difference-statistics.csv`):

```text
class,area_km2,percent,pixels,mean_change
Water gained,156.761,29.53,174179,0.806
Water lost,0.603,0.11,670,-0.233
No change,373.476,70.36,414973,0.1642
```

404 for an unknown job, another owner's, or a job without statistics (not completed, or an analysis that reports none).

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
    "footprint_wgs84": [[32.388779, 46.662667], [32.689939, 46.663879], [32.691118, 46.456535], [32.391103, 46.455331]],
    "line": "HLS L30 · B2–B6 · 30 m · 2023-06-01"
  },
  "owner": "8b5280fd5cdf0e24c6c6075dc45aff25"
}
```

The grayscale preview is written next to the upload as `<id>_<file name stem>.png`; `preview_url` is its download URL. `preview_png_base64` holds the same PNG and is kept for compatibility: do not persist it or send it back in job submissions.

`metadata` is a best guess, and any field may be missing: the sensor (`HLS L30`, `HLS S30`, `Sentinel-2`, `Landsat 8/9`), band codes from the band descriptions, data type, pixel size in metres, acquisition date (from the GeoTIFF tags or the file name), size, the bounds in degrees, and `footprint_wgs84`: the raster's four corners in degrees, `[[lon, lat] × 4]` from the north-west corner clockwise (NW, NE, SE, SW). A raster in a projected CRS such as UTM is rotated against the degree grid, so draw the footprint polygon, not the `bounds_wgs84` box. `line` is the chip text without the size; the chip appends ` · ` and `size_label`. A partial line such as `"3 bands · 11.1 m"` is possible, and `line` is `""` when nothing could be read. [GET /v1/files](#get-v1files) returns the same `metadata` for each file.

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

#### GET /v1/files

List the caller's uploads (operators: every upload, including those from before v0.6.0 without an owner) and the Earth Engine exports on the server, newest first.

**Response (after `POST /v1/sample`, trimmed to one of the two files; paths as under Docker Compose):**

```json
{
  "ok": true,
  "files": [
    {
      "id": "6774daf7_hlsl_20230609",
      "upload_id": "6774daf7",
      "filename": "6774daf7_hlsl_20230609.tif",
      "path": "/app/resources/.cache/uploads/6774daf7_hlsl_20230609.tif",
      "size": [768, 768],
      "crs": "EPSG:32636",
      "bounds": [453240.0, 5144820.0, 476280.0, 5167860.0],
      "size_bytes": 3839983,
      "source": "upload",
      "modified": 1792633679.79,
      "metadata": {"sensor": "HLS L30", "date": "2023-06-09", "footprint_wgs84": [...], "line": "HLS L30 · B2–B6 · 30 m · 2023-06-09", ...},
      "owner": "8b5280fd5cdf0e24c6c6075dc45aff25",
      "kind": "sample",
      "group": {"name": "kakhovka_hls_l30.tif", "line": "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636"},
      "preview_url": "/v1/download?path=%2Fapp%2Fresources%2F.cache%2Fuploads%2F6774daf7_hlsl_20230609.png",
      "created_at": "2026-10-24T00:29:32.973964"
    }
  ],
  "count": 2
}
```

| Field | Meaning |
|-------|---------|
| `id` | The stored file name without its extension |
| `upload_id` | The `id` that `POST /v1/upload` or `POST /v1/sample` returned; both sample dates share one. `null` for exports |
| `source` | `upload` or `export` |
| `kind` | `upload`, `sample` (the caller's copy of the bundled sample) or `export` (an Earth Engine export) |
| `group` | Set on both sample files, with the same value: show the pair as one row named `group.name`, with `group.line` |
| `metadata` | As in the [upload response](#post-v1upload), including `footprint_wgs84` |
| `preview_url` | The upload's PNG preview, or `null` if there is none (exports) |
| `created_at` | ISO time of the upload, else of the file's last modification |

#### DELETE /v1/files/{file_id}

Delete an upload with its preview and owner record. `file_id` is an `upload_id` or a file's `id` from `GET /v1/files`. The sample's two dates share one `upload_id`, so deleting by it removes the pair; `POST /v1/sample` attaches a fresh copy. Only copies in the uploads directory are deleted, never the bundled sample.

**Response:**

```json
{"ok": true, "deleted": ["6774daf7_hlsl_20230601.tif", "6774daf7_hlsl_20230609.tif"]}
```

404 for an unknown id, or another owner's upload (operators may delete any).

#### GET /v1/download

Download a job artifact, upload preview or other generated file by its server path. The `url` of an artifact reference and the `preview_url` of an upload already have this form: `/v1/download?path=` + the URL-encoded absolute path, and for an artifact of a finished job `&name=` + its `download_name`.

```bash
curl -b cookies.txt -OJ "http://localhost:8000/v1/download?path=%2Fapp%2Fresources%2F.cache%2Foutputs%2F501f9478%2F1_NDVIMap_ndvi.tif&name=ndvi_2023-06-01.tif"
# saved as ndvi_2023-06-01.tif
```

| Parameter | Description |
|-----------|-------------|
| `path` | The file on the server. It alone decides what is served |
| `name` | Optional: the file name of the download. The response is always an attachment, `Content-Disposition: attachment; filename="…"; filename*=UTF-8''…`, with `name` sanitised: `/`, `\`, double quotes, `<`, `>`, `:`, `\|`, `?`, `*` and control characters become `-`, leading and trailing dots, dashes and spaces are dropped, it is cut to 200 characters, and it ends in the served file's extension (appended if it ends in another). Without `name`, or when nothing is left of it, the stored file name is used |

The NDWI change GeoTIFF of the sample, for example, is stored as `2_ChangeMap_change.tif` and downloads as `water-change_ndwi_2023-06-01_2023-06-09.tif`.

Only files under these directories are served: job outputs (`resources/.cache/outputs/`, or `DT4LC_OUTPUTS_PATH`), uploads (`resources/.cache/uploads/`, or `DT4LC_UPLOADS_PATH`), temporary and model outputs (`resources/.cache/tmp/`, or `DT4LC_TEMP_PATH`), Earth Engine exports (`resources/.cache/gee_exports/`) and `<system temp>/dt4lc_delineate`. The path is resolved first, so `..` cannot leave them. Any other path returns 403 (`error.code` `unauthorized`); a missing file returns 404. A job's outputs and an upload are served to their owner only (and to operators); for anyone else they are 404. The content type follows the extension: `.tif`/`.tiff` `image/tiff`, `.png` `image/png`, `.gpkg` `application/geopackage+sqlite3`, `.json` `application/json`, anything else `application/octet-stream`.

### Map tiles

#### GET /v1/tiles/{z}/{x}/{y}

A 256 × 256 RGBA PNG of a GeoTIFF on the XYZ (Web Mercator) grid, for the map. `path` is the raster's server path, as for [GET /v1/download](#get-v1download), with the same directory and owner checks (403 outside the allowed directories, 404 for a missing file or another owner's).

```text
/v1/tiles/12/2416/1446?path=%2Fapp%2Fresources%2F.cache%2Fuploads%2F6774daf7_hlsl_20230601.tif
```

| Parameter | Description |
|-----------|-------------|
| `path` | Server path of the GeoTIFF |
| `style` | Optional, only `truecolor`: the red, green and blue bands, each stretched to its 2nd–98th percentile over the raster. A raster without resolvable red, green and blue bands is 422 |

The raster is reprojected onto the tile's grid (EPSG:3857) on the fly, so a UTM raster such as the sample has straight edges at every zoom. Pixels off the raster and nodata pixels are transparent, and a tile that does not meet the raster at all is a fully transparent 200. Without `style`:

- a job artifact carries its style in GeoTIFF tags (see [Job artifacts](ARCHITECTURE.md#job-artifacts)) and is drawn in the same colours as its `*_bare` map image: change and index rasters on their ramp and range (bilinear), class rasters in their class colours (nearest neighbour, so classes never blend);
- an untagged raster with three or more bands whose red, green and blue bands resolve (band descriptions, then band codes, then the band count; see "Expected Band Order" in the [README](../README.md#expected-band-order)) is drawn in true colour, as with `style=truecolor`. The stretch is computed once per file and cached. So the sample layer is just `/v1/tiles/{z}/{x}/{y}?path=<attachment path>`;
- any other untagged raster keeps the earlier rendering: the first three bands, stretched to their 2nd–98th percentile unless they are 8-bit, when it has three or more bands; else its one band, coloured by its file name (the NDVI, NDWI or NDSI colour map) or stretched in viridis.

`style=truecolor` on a single-band change raster:

```json
{"ok": false, "error": {"code": "validation_error", "message": "The raster has no red, green and blue bands", "details": {}}}
```

A served file that isn't a raster, such as a job's JSON output, is 415 `NOT_A_RASTER`. The body never holds the path:

```json
{"ok": false, "error": {"code": "NOT_A_RASTER", "message": "This file isn't a raster the map can display.", "details": {}}}
```

### Model downloads

`/v1/ml-models` manages the model weights (the generated reference has every route). Listing and status are open to everyone. `POST /v1/ml-models/{model_id}/download` and `POST /v1/ml-models/{model_id}/cancel` need a role that `MODEL_DOWNLOAD_ON_DEMAND` lets download (members and operators by default), and `DELETE /v1/ml-models/{model_id}` needs an operator. Anyone else gets 403 `MODEL_GATED`, with the same sentence a job's refusal has:

```json
{"ok": false, "error": {"code": "MODEL_GATED", "message": "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator.", "details": {"model_id": "prithvi-eo-v1-100m"}}}
```

With `MODEL_DOWNLOAD_ON_DEMAND=none` the message says that the model isn't installed and asks for an operator, for every role; a delete by a non-operator says "Only an operator can delete … from this server."

### Settings

Operator-only server settings, which the Settings page (`/app/settings`) reads and writes. For guests and members the three routes do not exist: any method answers the plain 404 of an unknown path, `{"ok": false, "error": {"code": "not_found", "message": "Not Found", "details": {}}}`. An operator is a user signed in with an `OPERATOR_EMAILS` address, or any caller under `DEV_ROLE=operator`.

What `PUT` changes is written to an overrides file, never to `.env`: `DT4LC_SETTINGS_PATH`, by default `<DT4LC_CACHE_PATH>/settings.json`. It is a flat JSON object from environment variable name to string. An override wins over the environment, which wins over the default, and it applies at once, without a restart. Only these variables can be overridden: `LLM_PROVIDER_ORDER`, `LLM_ENABLE_<PROVIDER>`, `GEMINI_MODELS`, `GROQ_MODEL`, `OLLAMA_MODEL`, `APERTUS_MODEL`, `ANTHROPIC_MODELS`, `MISTRAL_MODELS`, `OPENAI_MODELS`, `MODEL_DOWNLOAD_ON_DEMAND`, `ALLOWED_EMAILS`, `ALLOWED_DOMAINS`, `OPERATOR_EMAILS`, `GUEST_ANALYSES_PER_HOUR` and `GUEST_SESSION_HOURS`. None is a secret; API keys appear only as `key_present`. The file is read again whenever it changes, so a hand edit applies too; unknown names in it are ignored with a warning.

#### GET /v1/settings

Five sections. Every section object, and every provider in `assistant.providers`, has `editable: {field: bool}` (what `PUT` accepts) and, for its editable fields, `source: {field: "override" | "env" | "default"}` (where the value comes from).

**Response (trimmed; assistant offline, no keys):**

```json
{
  "assistant": {
    "provider_order": ["ollama"],
    "providers": {
      "ollama": {
        "enabled": false, "models": ["llama3.2"], "multiple_models": false,
        "key_required": false, "key_present": false, "available": false,
        "editable": {"enabled": true, "models": true, "multiple_models": false, "key_required": false, "key_present": false, "available": false},
        "source": {"enabled": "env", "models": "env"}
      },
      "gemini": {"enabled": true, "models": ["gemini-3.5-flash-lite"], "multiple_models": true, "key_required": true, "key_present": false, "available": false, ...},
      ...
    },
    "connected": false, "provider": null, "model": null,
    "test_path": "/v1/settings/assistant/test",
    "editable": {"provider_order": true, "providers": true, "connected": false, "provider": false, "model": false, "test_path": false},
    "source": {"provider_order": "env"}
  },
  "models": {
    "items": [{"id": "prithvi-eo-v1-100m", "display_name": "Prithvi EO v1 (100M)", "installed": false, "size_bytes": 471859200, "est_download_s": 75, "licence": "Apache-2.0", "licence_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M", "steps": ["models/prithvi-reconstruction"]}, ...],
    "download_policy": "members",
    "download_policy_options": ["all", "members", "operators", "none"],
    "pre_download": {"method": "POST", "path": "/v1/ml-models/{id}/download"},
    "editable": {"items": false, "download_policy": true, "download_policy_options": false, "pre_download": false},
    "source": {"download_policy": "env"}
  },
  "access": {
    "allowed_emails": [], "allowed_domains": [], "operator_emails": [],
    "guest_analyses_per_hour": 5, "guest_session_hours": 24,
    "bounds": {"guest_analyses_per_hour": {"min": 0, "max": 1000}, "guest_session_hours": {"min": 1, "max": 720}},
    "sign_in": {"configured": false},
    "editable": {"allowed_emails": true, "allowed_domains": true, "operator_emails": true, "guest_analyses_per_hour": true, "guest_session_hours": true, "bounds": false, "sign_in": false},
    "source": {"allowed_emails": "env", "allowed_domains": "env", "operator_emails": "env", "guest_analyses_per_hour": "default", "guest_session_hours": "default"}
  },
  "data_sources": {
    "earth_engine": {"configured": false, "project_id_present": false, "key_file_present": false},
    "uploads": {"path": "/app/resources/.cache/uploads", "files": 0, "bytes": 0},
    "sample": {"name": "kakhovka_hls_l30", "bundled": true, "path": "/app/resources/kahovka_data"},
    "editable": {"earth_engine": false, "uploads": false, "sample": false}
  },
  "server": {
    "version": "<dta.__version__>",
    "deployment": "local",
    "paths": {"cache": "...", "outputs": "...", "temp": "...", "uploads": "...", "models": "..."},
    "storage": {"outputs": {"bytes": 12645517, "files": 9, "truncated": false}, ...},
    "jobs": {"total": 2, "by_status": {"pending": 0, "proposal": 0, "running": 0, "completed": 1, "failed": 1, "cancelled": 0}, "queue_size": 0, "workers": 3},
    "editable": {"version": false, "deployment": false, "paths": false, "storage": false, "jobs": false}
  }
}
```

- `assistant.providers` lists every known provider, `provider_order` first. `available` means the provider is in the assistant's router (listed in the order, enabled, with its key if it needs one); it makes no network call. `connected`, `provider` and `model` are the `assistant` of `GET /v1/health`.
- `models.items` are the installable models; `pre_download` is the route that installs one (replace `{id}`; see [Model downloads](#model-downloads)).
- `server.storage` has `{bytes, files, truncated}` per path, reused for 10 s; `truncated` means the walk stopped at 100,000 entries and the numbers are lower bounds. With the default paths, `cache` contains the others.

#### PUT /v1/settings

A partial document with only the fields to change; it returns the new `GET` document. Writable fields:

```json
{
  "assistant": {"provider_order": ["groq", "ollama"], "providers": {"mistral": {"enabled": true, "models": ["mistral-medium-latest"]}}},
  "models": {"download_policy": "operators"},
  "access": {"allowed_emails": ["ana@example.org"], "allowed_domains": ["unige.ch"], "operator_emails": ["ops@example.org"], "guest_analyses_per_hour": 3, "guest_session_hours": 24}
}
```

- A field set to `null` drops its override, so the environment (or the default) applies again: after `{"access": {"guest_analyses_per_hour": 3}}` the field's `source` is `override`; after `{"access": {"guest_analyses_per_hour": null}}` it is `env` or `default` again.
- `provider_order` lists known providers without duplicates; `groq`, `ollama` and `apertus` take exactly one model, the others up to ten. Emails are lower-cased, and domains may start with `@`.
- Changing the assistant rebuilds the LLM router and drops the cached assistant status, so `GET /v1/health` and `GET /v1/capabilities` show the change at once.
- A signed-in operator cannot drop their own address from `operator_emails`: the request is refused, as are an unknown or read-only key, a wrong type and a value out of `bounds`. Each is a 422 in FastAPI's `detail` format, e.g.

```json
{"detail": [{"type": "less_than_equal", "loc": ["body", "access", "guest_analyses_per_hour"], "msg": "Input should be less than or equal to 1000", "input": 5000, "ctx": {"le": 1000}}]}
```

and for `{"server": {"version": "x"}}`, `{"type": "extra_forbidden", "loc": ["body", "server"], "msg": "Extra inputs are not permitted", ...}`.

#### POST /v1/settings/assistant/test

Sends a one-word prompt to every provider of the assistant's router, concurrently, and reports each in router order. A provider that does not answer within 20 s fails. Errors have API keys redacted. From `tests/test_settings.py` (latencies vary):

```json
{
  "connected": true,
  "providers": [
    {"name": "gemini", "model": "gemini-x", "ok": true, "latency_ms": 12, "error": null},
    {"name": "groq", "model": "llama-y", "ok": false, "latency_ms": 3, "error": "RuntimeError: 401 for key *** (also ***)"},
    {"name": "ollama", "model": "llama3.2", "ok": false, "latency_ms": 0, "error": "not available"}
  ],
  "error": null
}
```

`connected` is true when at least one probe answered. With no provider in the router, `providers` is empty and `connected` false; `error` is set only when the router itself could not be built.

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

Neither endpoint takes a file, so it suits requests that name their inputs in the prompt or need none. A prompt that the intent classifier reads as conversation is answered with 422 and no plan, and never counts against the guest quota:

| `error.code` | When | `error.message` |
|--------------|------|-----------------|
| `NOT_AN_ANALYSIS` | A greeting, a question about methods, a capability question ("hello") | "This reads as a question, not an analysis — send it to POST /v1/chat." |
| `missing_input` | An analysis that needs a raster ("Calculate NDVI"). `POST /v1/execute` also returns it, after planning, for a plan that stops on a missing input file ([below](#post-v1execute)) | "This analysis needs a raster, and this endpoint takes no file: send it to POST /v1/chat or POST /v1/jobs with the file attached." |

Both carry `details: {"intent": "conversation"}`:

```json
{"ok": false, "error": {"code": "NOT_AN_ANALYSIS", "message": "This reads as a question, not an analysis — send it to POST /v1/chat.", "details": {"intent": "conversation"}}}
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
  "result": {
    "ok": true,
    "artifacts": {...}
  },
  "progress": [...]
}
```

**Errors**, besides the 422s of [POST /v1/plan](#post-v1plan), which come before the quota counts:

| Status | `error.code` | When |
|--------|--------------|------|
| 429 | `GUEST_QUOTA` | A guest beyond `GUEST_ANALYSES_PER_HOUR`; counted as for [POST /v1/jobs](#post-v1jobs) |
| 409 | `CONFIRMATION_REQUIRED` | Planning composed a plan (a proposal), which runs only after confirmation: "This request composes a new analysis, which runs only after you confirm it: submit it with POST /v1/jobs, then confirm with POST /v1/jobs/{job_id}/confirm." Not counted |
| 422 | `missing_input` | Planning succeeded, but the plan stops on a missing input file, e.g. "Map moisture change with NDMI between the two dates", which needs a before and an after file. The body is the classifier's `missing_input` one above. Not counted: the quota is released |
| 400 | `planning_failed` | See [Error Responses](#error-responses). Counted |

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
| 403 | `unauthorized`; `MODEL_GATED` from the [model routes](#model-downloads) |
| 404 | `not_found`; `SAMPLE_NOT_BUNDLED` from `POST /v1/sample`; `SIGN_IN_NOT_CONFIGURED` from the [GitHub sign-in routes](#sign-in) |
| 409 | `CONFIRMATION_REQUIRED` from `POST /v1/execute`; otherwise `bad_request` |
| 413 | `file_too_large` |
| 415 | `NOT_A_RASTER` from [`GET /v1/tiles`](#get-v1tileszxy) |
| 422 | `validation_error` (raised by a route, e.g. an invalid edited plan on confirm, or `style=truecolor` on a raster without RGB); `NOT_AN_ANALYSIS` and `missing_input` from [`POST /v1/plan` and `POST /v1/execute`](#post-v1plan) |
| 429 | `GUEST_QUOTA` for the guest limit, with `details: {limit, used, resets_at}` and `Retry-After`; `bad_request` when the job queue is full; `RATE_LIMITED` from the demo's nginx (see [Rate Limits](#rate-limits)) |
| 405 and other 4xx | `bad_request` |
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

Guests may start `GUEST_ANALYSES_PER_HOUR` (default 5) analyses per rolling hour, counted per session and per client IP; the next one gets 429 `GUEST_QUOTA` (see [POST /v1/jobs](#post-v1jobs)). Conversation replies and proposals do not count. Members and operators are not limited. Behind a reverse proxy, the client IP is the one in `X-Forwarded-For` when the proxy's address is in `FORWARDED_ALLOW_IPS` (uvicorn runs with `--proxy-headers` in the Docker image); otherwise it is the proxy's own address, and every guest behind it shares one count.

The backend has no other limit. The public demo's nginx (`deploy/nginx-demo.conf.template`, see [DEPLOY_DEMO.md](DEPLOY_DEMO.md#7-rate-limits-and-quotas)) adds per-IP limits for every role: `POST` to `/v1/jobs`, `/v1/chat`, `/v1/upload`, `/v1/sample`, `/v1/plan` or `/v1/execute` 10 a minute with a burst of 5, and any `/v1/` request 20 a second with a burst of 200. A request over a limit gets 429 with `Retry-After: 10`:

```json
{"ok": false, "error": {"code": "RATE_LIMITED", "message": "Too many requests from your address. Wait a few seconds and try again.", "details": {}}}
```

## CORS

CORS allows the origins in `CORS_ORIGINS` (default `*`) without credentials (`allow_credentials=False`). A cross-origin caller therefore sends no session cookie, and each of its requests is a new guest. Serve the UI and the API on one origin, as nginx does under Docker Compose and the Vite dev server's `/v1` proxy does locally.
