# Changelog

All notable changes to DT4LC are listed here, newest first. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release, with its full notes and container images, is on the [GitHub releases page](https://github.com/IPT-MMDA/DT4LC-project/releases).

## [0.6.0] - Unreleased

A conversation-first workspace replaces the job-centred interface: you describe an analysis in plain language, and the assistant says what it will do, runs it and shows the result in the same message. `v0.6.0-alpha.1` was a preview with the backend contracts and the Workspace.

### Added

- **Workspace.** One message per analysis: the assistant's intent, the plan steps with their times and **Cancel**, then a result card, an interpretation and follow-up suggestions. A failed analysis keeps its steps open, with the reason in plain words and fixes as buttons.
- **Proposals.** When no built-in analysis fits, the assistant proposes one (**Run · Change · Explain**), and the analysis that runs is marked **Composed by the assistant**. NDMI, NBR, SAVI and NDBI are composed even without an assistant. The new `algorithms/band-math` and `algorithms/band-math-change` accept only numbers, band names, arithmetic operators and parentheses.
- **Result cards** for index, change, classification, field-boundary and statistics results, with a legend, the source line, km² statistics, and **On map · GeoTIFF · PNG · GeoPackage · Details**.
- Conversation replies stream as markdown and never start a job. Dates in the prompt pick the file ("NDVI for 1 June").
- **Composer:** **+**, **From map**, **From library** and drag-and-drop, with attachment chips that show the sensor, bands, resolution, size and dates.
- **The bundled Kakhovka sample** attaches with one click (`POST /v1/sample`).
- A **Preparing model** step with the download size and time left when a model's weights are missing.
- **New screens:**
  - the landing page at `/`;
  - **Capabilities**, six tiers computed from the live server state;
  - **Map mode**, with the imagery panel, region presets, drawing and **Send to chat**, result layers with legends, and a **Compare** swipe (building on pull requests #51 by nik4nd and #35 by NataliaGon);
  - the **Library**, analyses and files in one table;
  - **Result detail** (`/app/results/:id`), with readable download names such as `water-change_ndwi_2023-06-01_2023-06-09.tif`;
  - **Settings** for operators (`/app/settings`: Assistant, Models, Access, Data sources, Server), saved to an overrides file, never to `.env`.
- **Sessions and sign-in.** Every visitor starts as a guest: 5 analyses an hour, with work kept for 24 hours in that browser. GitHub sign-in makes an allowlisted address a member (no limit, work kept on any browser) or an operator (also Settings and every owner's jobs). A guest's work moves to the account on sign-in.
- **Public demo deployment:** `docker compose --profile demo` with `docker-compose.demo.yml`. It pre-installs the model weights, rate-limits each client IP in nginx, publishes no backend port and removes guest data on a schedule. The runbook is `docs/DEPLOY_DEMO.md`.
- **Look and accessibility:** a new visual language with light and dark themes, self-hosted Inter and IBM Plex Mono fonts, phone (390 px) and tablet (834 px) layouts, and keyboard and screen-reader support.
- **New endpoints:** `GET /v1/jobs/{id}/events`, `POST /v1/jobs/{id}/confirm`, `DELETE /v1/jobs/{id}`, `GET /v1/jobs/{id}/statistics.csv`, `POST /v1/sample`, `DELETE /v1/files/{id}`, `GET /v1/session`, `GET /v1/auth/status`, `GET /v1/auth/github/login`, `GET /v1/auth/github/callback`, `POST /v1/auth/logout`, `GET` and `PUT /v1/settings`, `POST /v1/settings/assistant/test` and `GET /v1/tiles/{z}/{x}/{y}`.
- The University of Maryland seal in the landing page's footer, next to the MMDA and University of Geneva logos.

### Changed

- **Breaking:** every response sets a signed `dt4lc_session` cookie, and a client that does not send it back is a new guest on every request. API scripts must keep a cookie jar (`curl -c jar -b jar`).
- **Breaking:** `POST /v1/chat` is the conversation endpoint. It streams markdown replies and never plans or runs a job, so the `planning`, `plan_ready`, `executing` and `complete` frames are gone. Analyses go through `POST /v1/jobs` and its event stream.
- **Breaking:** `GET /v1/capabilities` returns `{summary, assistant, offline, banner, tiers}` for the caller instead of `{instances: [...]}`. `GET /v1/models` still lists the models.
- **Breaking:** a composed plan stops at `proposal` until `POST /v1/jobs/{id}/confirm`, and `POST /v1/execute` refuses an unconfirmed one with 409 `CONFIRMATION_REQUIRED`.
- **Breaking:** `/v1/plan` and `/v1/execute` answer conversation and file-less prompts with 422 (`NOT_AN_ANALYSIS`, or `missing_input`) instead of 500.
- **Breaking:** result statistics are keyed by band code (`B2`, …) instead of `band_<n>`, each date's change index is clipped to [−1, 1], artifact references gain `size_bytes` and `download_name` (their `url` ends in `&name=`), and every finished job has a `card`. Older jobs are returned as stored.
- Map tiles are warped to Web Mercator, and multi-band rasters render in true colour by default.
- The v0.5 routes (`/chat`, `/dashboard`, `/map`, `/jobs`, `/jobs/:jobId`, `/data`, `/models`, `/settings`) redirect to the new screens under `/app`.
- `CITATION.cff` and the README give the licence as `AGPL-3.0-or-later` (version 3 or any later version), as `pyproject.toml` does.
- From this release on, `release/vX.Y.Z` is merged into `dev`, and `dev` into `master`, with merge commits, never squashed, so `master` keeps every commit with its author. The detailed v0.5.x history stays on the `release/v0.5.0` and `release/v0.5.1` branches.

### Removed

- The v0.5 pages: Chat, the Dashboard page, the Jobs page with its job details, the Data page and the Models page. The Workspace takes over chat and the dashboard, the Library jobs and data, and Result detail the job details. Capabilities shows each model's state, and operators pre-download models in Settings → Models.

### Fixed

- Tiles of areas outside a raster are transparent.
- The map panels no longer overlap (#67, bug 4).
- The map credits the oblast boundaries in its attribution control ("Oblast boundaries: OCHA / HDX, CC BY-IGO 3.0"), as their licence requires, and `THIRD_PARTY_LICENSES.md` gives their source and licence.

### Security

- Jobs, events, statistics, deletes, downloads, tiles and files of another owner answer 404. Jobs and uploads from before v0.6.0 have no owner and are visible to operators only.
- Attachments on `POST /v1/jobs` and `POST /v1/chat` must be the caller's own upload, sample copy or job output, or a file that belongs to no one, such as an Earth Engine export. A path outside the server's roots answers 403, and a missing file or another owner's 404, before the guest quota counts the request.
- A guest's sixth analysis in an hour (`POST /v1/jobs`, counted at confirm for proposals, and `POST /v1/execute`) answers 429 `GUEST_QUOTA`.
- Model download, cancel and delete need a member or an operator (403 `MODEL_GATED`). On the demo, only operators download.

### Upgrading from 0.5.1

- Local installs now start everyone as a guest. On a single-user machine, set `DEV_ROLE=operator` (or `member`), configure GitHub sign-in with yourself in `OPERATOR_EMAILS`, or raise `GUEST_ANALYSES_PER_HOUR` and `GUEST_SESSION_HOURS`.
- Set `SESSION_SECRET`, or every session resets when the server restarts. Behind HTTPS, set `SESSION_COOKIE_SECURE=true`.
- Existing jobs and uploads are kept, without an owner: operators see them, and the guest cleanup never removes them. The job database migrates itself on start.
- The new variables are all in `.env.example`, with comments.
- Settings changes go to `resources/.cache/settings.json` (`DT4LC_SETTINGS_PATH`) and take precedence over `.env`, so back the file up with the job database.
- Docker: rebuild both images (`docker compose up -d --build`), and run `docker compose down` before switching between the default and the demo stack.

## [0.5.1] - 2026-10-23

A correctness and safety hotfix for 0.5.0, with no API changes.

### Fixed

- Reflectance scaling follows the raster's data type, so dark integer scenes no longer give a wrong EVI or false snow, in spectral indices, change detection and snow classification (#64).
- Landsat 8/9 `B1..B11` stacks and `SR_B*`/`ST_B*` band names map to the right red, NIR and SWIR bands (#64).
- HLS preprocessing for Prithvi writes NaN and nodata pixels as −9999 with a nodata tag, and rounds values (#60).
- Job cleanup deletes the job's output files (#65).
- `GET /v1/jobs` no longer loads job results: 35 ms instead of 2.4 s for 200 large jobs (#67, bug 2).
- Algorithm images render without pyplot's global state, so concurrent jobs no longer corrupt each other's images.
- `/v1/download` serves Earth Engine exports again, and a directory answers 404 instead of 500.

### Security

- `/v1/tiles` reads only files under the server's allowed directories, the same check as `/v1/download`; any other path answers 403.

## [0.5.0] - 2026-10-22

The revision for the JOSS review (openjournals/joss-reviews#11334). It also merges community pull requests and fixes the stability problems reported from the CREODIAS deployment.

### Added

- A 768×768 HLS L30 crop of the 2023 Kakhovka dam breach (lower Dnipro, 2023-06-01 and 2023-06-09), bundled as the test and sample dataset (#59).
- A deterministic fallback: with no LLM reachable, context extraction and planning fall back to deterministic logic, so analysis jobs still complete.
- Structured error responses, `{ok, error: {code, message, details}}` (#10; pull request #49 by nik4nd).
- Swagger UI and ReDoc through nginx, and response models on all routes (#7; pull request #56 by NataliaGon).
- A developer guide (#9; pull request #54 by NataliaGon).
- An OpenAI provider (`OPENAI_API_KEY` or `GPT_SECRET`, default `gpt-4o-mini`).
- vitest store and logic tests, and a `frontend-test` CI job.

### Changed

- Job results no longer contain pixel arrays or base64 images. Rasters are written as GeoTIFF and visualizations as PNG, and results hold `{kind, path, url, stats}` references; `GET /v1/jobs` returns summaries, and uploads return a `preview_url`. A 1000×1000 NDVI job response drops from 47 MB to under 1 MB (#65).
- Band roles are read from GeoTIFF band descriptions (Sentinel-2 and Landsat/HLS codes, or names), with a warned fallback to the band count (#64).
- Gemini uses `gemini-3.5-flash-lite` by default and falls back across `GEMINI_MODELS` on 404, 429 and 503. The LLM router logs each provider's real error, with API keys redacted.
- The version is served from `dta.__version__`.
- The paper and the documentation were revised for the review (#61, #62).

### Fixed

- Spectral indices are correct on integer rasters (×10000 reflectance), mask nodata and NaN, clip to the formula's range and compute statistics over valid pixels only (#64).
- Delineate-Anything runs from a pipeline, with a `delineate` extra for the GDAL bindings; the Prithvi `scale-to-hls` preprocessor is applied, and a wrong band count fails with a clear error (#60).
- The documented test command passes without API keys: LLM, network and GPU tests skip with a reason (#63).
- The Docker quickstart works without keys with `docker compose --profile ollama up -d`.
- `pip install .` works without extras.
- Cancelled jobs stay cancelled, job images and previews render after a reload, the map delete tool works, and the selected region survives a reload (#67, bugs 1, 2, 3, 5 and 6).
- CREODIAS deployment fixes: API URL fallbacks, the Claude 5 temperature rule, the Earth Engine RGB band order and stretch, and oblast boundaries (pull request #66 by annayailymova).
- The `task run:api:dev` and `task download:prithvi` targets.

### Security

- `/v1/download` serves only files under the application's own directories.
- Plan resource limits are enforced in the orchestrator.

## [0.4.0] - 2026-05-06

The version submitted to JOSS.

[0.6.0]: https://github.com/IPT-MMDA/DT4LC-project/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/IPT-MMDA/DT4LC-project/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/IPT-MMDA/DT4LC-project/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/IPT-MMDA/DT4LC-project/tree/v0.4.0
