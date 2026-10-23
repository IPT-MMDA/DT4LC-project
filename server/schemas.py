"""Pydantic schemas for the FastAPI server.

Defines request/response models for the REST API endpoints including
job submission, chat messages, and file attachments.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dta import __version__

# Re-export Attachment from domain schemas to avoid duplication
from dta.dti.schemas import Attachment, RasterMetadata

__all__ = [
    "Attachment",
    "BulkFetchRequest",
    "BulkFetchResponse",
    "CancelJobResponse",
    "CapabilitiesResponse",
    "ChatMessage",
    "ChatRequest",
    "ChatTurn",
    "ConversationAnalysis",
    "ConversationReply",
    "ConversationRequest",
    "CreateJobRequest",
    "DatasetListResponse",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "ExecuteResponse",
    "FileInfo",
    "FileListResponse",
    "GEEDatesResponse",
    "GEEExportResponse",
    "GEEFetchResponse",
    "JobListResponse",
    "JobRecord",
    "JobRegion",
    "JobStatus",
    "JobSubmitRequest",
    "MetricsResponse",
    "MLModelActionResponse",
    "MLModelDetailResponse",
    "MLModelDownloadResponse",
    "MLModelsListResponse",
    "PendingIntent",
    "PersistLayerRequest",
    "PersistLayerResponse",
    "Plan",
    "PlanResponse",
    "QueueStatsResponse",
    "RegistryModelsResponse",
    "SessionAnalyses",
    "SessionResponse",
    "UploadResponse",
]


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    INTERNAL_ERROR = "internal_error"
    MISSING_INPUT = "missing_input"
    MODEL_NOT_INSTALLED = "model_not_installed"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    BAD_REQUEST = "bad_request"
    FILE_TOO_LARGE = "file_too_large"
    UNAUTHORIZED = "unauthorized"


class ErrorDetail(BaseModel):  # type: ignore[misc]
    """Detailed error information."""

    code: ErrorCode | str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):  # type: ignore[misc]
    """Standardized error response format."""

    ok: Literal[False] = False
    error: ErrorDetail


Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):  # type: ignore[misc]
    """Single message in a chat conversation."""

    role: Role = Field(description="Message author role")
    content: str = Field(description="Message text")


class ChatRequest(BaseModel):  # type: ignore[misc]
    """Request containing chat message history."""

    messages: list[ChatMessage] = Field(description="Conversation messages; last user message is used as the prompt")


class JobRegion(BaseModel):  # type: ignore[misc]
    """A region sent from the map: what to fetch from Earth Engine for the analysis.

    Exactly one of ``geometry`` and ``bbox``; ``start_date`` ≤ ``end_date``; at most
    ``MAX_REGION_KM2`` (2500 km²). Anything else is a 422.
    """

    model_config = {"extra": "forbid"}

    geometry: dict[str, Any] | None = Field(None, description="GeoJSON Polygon or MultiPolygon in WGS84")
    bbox: list[float] | None = Field(None, description="[west, south, east, north] in degrees")
    name: str | None = Field(None, max_length=80, description="Region name, e.g. 'Dnipro floodplain'")
    place: str | None = Field(None, max_length=80, description="Place, e.g. 'Kherson'")
    dataset: Literal["HLS L30", "HLS S30", "Sentinel-2", "Landsat 8/9", "MODIS"]
    start_date: date
    end_date: date
    bands: list[str] | None = Field(
        None, description="Band codes, e.g. ['B2', 'B5']; default: the dataset's optical bands"
    )
    cloud_max: int = Field(20, ge=0, le=100, description="Maximum cloud cover, percent")
    _area_km2: float = PrivateAttr(0.0)
    _bounds: list[float] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")  # type: ignore[misc]
    def _check(self) -> "JobRegion":
        from dta.dti.data_sources.gee_region import DATASETS, band_code

        from .regions import MAX_REGION_KM2, area_km2, format_area, region_shape

        if self.start_date > self.end_date:
            raise ValueError("region start_date is after end_date")
        geom = region_shape(self.geometry, self.bbox)
        area = area_km2(geom)
        if area > MAX_REGION_KM2:
            raise ValueError(
                f"region is {format_area(area)}; the limit is {format_area(MAX_REGION_KM2)}. Draw a smaller region."
            )
        if self.bands is not None:
            if not self.bands:
                raise ValueError("region bands is empty; leave it out for the dataset's default bands")
            codes = [band_code(self.dataset, b) for b in self.bands]
            unknown = [b for b, c in zip(self.bands, codes, strict=True) if c is None]
            if unknown:
                bands = ", ".join(DATASETS[self.dataset].bands)
                raise ValueError(f"{self.dataset} has no band {', '.join(unknown)}; its bands are {bands}")
            self.bands = list(dict.fromkeys(c for c in codes if c))
        self._area_km2 = area
        self._bounds = [round(v, 6) for v in geom.bounds]
        return self

    def echo(self, note: str | None = None) -> dict[str, Any]:
        """The job's ``region``: the request (bands resolved), ``bounds``, ``area_km2``, ``line`` and ``note``."""
        from dta.dti.data_sources.gee_region import DATASETS

        from .regions import region_line

        echo: dict[str, Any] = self.model_dump(mode="json")
        echo["bands"] = self.bands or list(DATASETS[self.dataset].default_bands)
        echo["bounds"] = self._bounds
        echo["area_km2"] = round(self._area_km2, 1)
        echo["line"] = region_line(echo)
        if note:
            echo["note"] = note
        return echo


class JobSubmitRequest(BaseModel):  # type: ignore[misc]
    """Request for submitting a new job."""

    prompt: str = Field(description="Natural-language analysis request")
    mode: str = Field(default="hybrid", description="Planning mode: hybrid, llm, or template")
    attachments: list[Attachment] = Field(default_factory=list, description="Uploaded GeoTIFF attachments")
    context: dict[str, Any] | None = Field(default=None, description="Optional session context for follow-up requests")
    region: JobRegion | None = Field(
        default=None,
        description="Region from the map, fetched from Earth Engine when no file is attached (attachments win)",
    )


class Plan(BaseModel):  # type: ignore[misc]
    """Execution plan for a pipeline of analysis steps."""

    tags: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    pipeline: list[str] = Field(default_factory=list, description="Ordered tool or algorithm IDs")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input bindings, e.g. file paths")
    meta: dict[str, Any] = Field(default_factory=dict)


class CreateJobRequest(BaseModel):  # type: ignore[misc]
    """Request to create a job from a pre-defined plan."""

    plan: Plan


class JobStatus(BaseModel):  # type: ignore[misc]
    """Legacy job status schema (prefer JobRecord for API responses)."""

    id: str
    state: Literal["queued", "running", "succeeded", "failed"] = "queued"
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# --- Jobs ---


class JobCardAnalysis(BaseModel):  # type: ignore[misc]
    id: str = Field(description="Registry id of the main analysis step, e.g. 'algorithms/change-detection'")
    label: str = Field(description="e.g. 'Change detection', 'Band math', 'Statistics', 'Delineate-Anything'")


class JobCardError(BaseModel):  # type: ignore[misc]
    code: str = Field(description="The failed event's code, e.g. 'EE_NOT_CONFIGURED'")
    message: str


class JobCard(BaseModel):  # type: ignore[misc]
    """A Library row without the result (server/library.py), stored when the job reaches a terminal state."""

    title: str = Field(description="e.g. 'Water change · NDWI difference', 'NDVI · 2023-06-01'")
    analysis: JobCardAnalysis | None = None
    kind: Literal["change", "index", "classification", "fields", "statistics", "composed", "image"] | None = None
    thumbnail_url: str | None = Field(None, description="Download URL of the main map-only PNG (*_bare)")
    duration_s: float | None = Field(None, description="completed_at - started_at, seconds")
    error: JobCardError | None = Field(None, description="Failed jobs: the failed event's code and message")


class JobSummary(BaseModel):  # type: ignore[misc]
    """Async job status without results (``GET /v1/jobs`` list item)."""

    id: str
    status: str = Field(description="pending, proposal, running, completed, failed, or cancelled")
    prompt: str
    plan: dict[str, Any] | None = None
    progress: float = Field(ge=0.0, le=1.0)
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    owner: str | None = Field(default=None, description="Submitter's session id; null for jobs from before v0.6.0")
    card: JobCard | None = Field(
        default=None, description="Library row summary once the job is finished; null before, and for older jobs"
    )


class JobRecord(JobSummary):
    """Async job status and results (``GET /v1/jobs/{id}``)."""

    result: dict[str, Any] | None = Field(
        default=None, description="Pipeline result; files are referenced as {kind, path, url, stats, size_bytes}"
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Event log as replayed by GET /v1/jobs/{id}/events: [{seq, type, data}] (server/job_events.py)",
    )
    proposal: dict[str, Any] | None = Field(
        default=None,
        description="{text, plan} when planning proposed a composed plan (status proposal until confirmed)",
    )
    region: dict[str, Any] | None = Field(
        default=None,
        description="The submitted region with bounds, area_km2 and the chip line; note if attachments won",
    )


class JobConfirmRequest(BaseModel):  # type: ignore[misc]
    """``POST /v1/jobs/{id}/confirm``: run the proposal, or ``plan`` if the user changed it."""

    plan: dict[str, Any] | None = Field(
        default=None, description="Edited execution plan; its file inputs are the job's own attachments"
    )


class JobListResponse(BaseModel):  # type: ignore[misc]
    """Paginated job list (summaries; fetch a job by id for its result)."""

    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int


class QueueStatsResponse(BaseModel):  # type: ignore[misc]
    """Job queue statistics."""

    model_config = {"extra": "allow"}


class CancelJobResponse(BaseModel):  # type: ignore[misc]
    """Job cancellation result."""

    model_config = {"extra": "allow"}


# --- Chat / COE ---


class PlanResponse(BaseModel):  # type: ignore[misc]
    """Successful plan generation response."""

    ok: bool = True
    plan: dict[str, Any] = Field(description="Validated execution plan")


class ExecuteResponse(BaseModel):  # type: ignore[misc]
    """Plan execution response."""

    ok: bool = True
    plan: dict[str, Any]
    result: dict[str, Any] = Field(description="Pipeline execution artifacts and outputs")
    progress: list[dict[str, Any]] = Field(default_factory=list, description="Step progress events")


class ChatTurn(BaseModel):  # type: ignore[misc]
    """One earlier turn of the conversation, as the client shows it."""

    role: Role
    text: str = Field(description="Turn text; for an analysis, its summary or interpretation")


class PendingIntent(BaseModel):  # type: ignore[misc]
    """An analysis request that waits for its input; the client stores it and sends it back."""

    prompt: str = Field(description="The request to run once the input arrives")
    item_id: str | None = Field(None, description="Registry id of the analysis it names, if any")
    needs: list[Literal["RasterPath", "RasterPathBefore", "RasterPathAfter"]] = Field(
        min_length=1, description='["RasterPath"], or ["RasterPathBefore", "RasterPathAfter"] for two dates'
    )
    asked_at: datetime = Field(description="When the assistant asked (UTC); a pending older than 30 min expires")


class ConversationRequest(BaseModel):  # type: ignore[misc]
    """``POST /v1/chat``: one Workspace message."""

    message: str = Field("", description="The user's message; may be empty when only files are attached")
    history: list[ChatTurn] = Field(
        default_factory=list, description="Earlier turns, oldest first; the server keeps the last 10 (8000 chars)"
    )
    attachments: list[Attachment] = Field(default_factory=list, description="Uploads, with their metadata")
    region: JobRegion | None = Field(None, description="As in POST /v1/jobs")
    pending: PendingIntent | None = Field(None, description="The pending of the previous reply, if any")


class ConversationReply(BaseModel):  # type: ignore[misc]
    """A conversation reply: markdown, no job."""

    kind: Literal["reply"] = "reply"
    text: str = Field(description="Markdown")
    pending: PendingIntent | None = Field(None, description="Set when the reply asks for the input of an analysis")


class ConversationAnalysis(BaseModel):  # type: ignore[misc]
    """An analysis request: the job submitted as by POST /v1/jobs; follow GET /v1/jobs/{job_id}/events."""

    kind: Literal["analysis"] = "analysis"
    job_id: str
    job: JobRecord


# --- Files ---


class UploadResponse(BaseModel):  # type: ignore[misc]
    """GeoTIFF upload response."""

    id: str = Field(description="Short file identifier")
    filename: str
    path: str = Field(description="Server path for pipeline attachments")
    size: list[int] = Field(description="Raster width and height in pixels")
    crs: str | None = None
    bounds: list[float] = Field(description="Bounding box [left, bottom, right, top]")
    preview_png_base64: str = Field(description="Grayscale PNG preview as base64 (do not persist)")
    preview_url: str = Field(description="Download URL of the PNG preview written next to the upload")
    metadata: RasterMetadata = Field(description="Sensor guess, bands, resolution, date, size and display line")
    owner: str | None = Field(default=None, description="Uploader's session id")


class SampleResponse(BaseModel):  # type: ignore[misc]
    """The bundled Kakhovka sample attached as the caller's uploads (``POST /v1/sample``)."""

    name: str = Field(description="Chip name, 'kakhovka_hls_l30'")
    line: str = Field(description="Pair chip line, e.g. 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09'")
    attachments: list[UploadResponse] = Field(
        description="Both dates, oldest first, as uploads (preview_png_base64 is empty: use preview_url)"
    )


class FileGroup(BaseModel):  # type: ignore[misc]
    """Files the Library shows as one row: the two dates of the bundled sample."""

    name: str = Field(description="Row name, 'kakhovka_hls_l30.tif'")
    line: str = Field(description="e.g. 'HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636'")


class FileInfo(BaseModel):  # type: ignore[misc]
    """Metadata for an on-disk GeoTIFF."""

    id: str = Field(description="The file name without its extension")
    upload_id: str | None = Field(
        None,
        description="The id POST /v1/upload or POST /v1/sample returned (the sample pair shares one); "
        "DELETE /v1/files/{upload_id}; null for exports",
    )
    filename: str
    path: str
    size: list[int]
    crs: str | None = None
    bounds: list[float]
    size_bytes: int
    source: Literal["upload", "export"] = Field(description="upload or gee export")
    modified: float = Field(description="Unix modification time")
    metadata: RasterMetadata | None = Field(None, description="As in UploadResponse.metadata")
    owner: str | None = Field(None, description="Uploader's session id; null for exports and older uploads")
    kind: Literal["upload", "sample", "export"] = Field(
        description="upload, sample (a copy of the bundled sample from POST /v1/sample) or export (Earth Engine)"
    )
    group: FileGroup | None = Field(None, description="Set on both sample files: show them as one row")
    preview_url: str | None = Field(None, description="Download URL of the PNG preview; null if there is none")
    created_at: str = Field(description="ISO time of the upload (the owner record), else of the last modification")


class FileDeleteResponse(BaseModel):  # type: ignore[misc]
    """``DELETE /v1/files/{id}``: the files removed with their previews and owner records."""

    ok: bool = True
    deleted: list[str] = Field(description="File names removed, e.g. both sample dates")


class FileListResponse(BaseModel):  # type: ignore[misc]
    """List of available GeoTIFF files."""

    ok: bool = True
    files: list[FileInfo]
    count: int


# --- Session ---


class SessionAnalyses(BaseModel):  # type: ignore[misc]
    """Analyses in the last hour; ``limit``, ``remaining`` and ``resets_at`` are null for unlimited roles."""

    limit: int | None
    used: int
    remaining: int | None
    resets_at: str | None = Field(description="ISO time (UTC) when ``used`` next drops by one; null if nothing used")


class SessionResponse(BaseModel):  # type: ignore[misc]
    """The caller's session (``GET /v1/session``): role, guest quota and retention for the UI badge."""

    role: Literal["guest", "member", "operator"]
    guest: bool
    analyses: SessionAnalyses
    kept_hours: int | None = Field(description="Hours a guest's jobs and uploads are kept; null for other roles")
    expires_at: str = Field(description="ISO time (UTC) when the session cookie expires")
    can_download_models: bool = Field(description="Whether this role may download model weights on demand")


# --- Health / registry ---


class LLMProviderStatus(BaseModel):  # type: ignore[misc]
    """Status of one LLM provider."""

    name: str | None = None
    model: str | None = None
    available: bool | None = None
    error: str | None = None


class GEEStatus(BaseModel):  # type: ignore[misc]
    """Google Earth Engine initialization status."""

    initialized: bool
    service_account_configured: bool
    error: str | None = None


class ModelEntry(BaseModel):  # type: ignore[misc]
    """Summary of one registered model."""

    id: str
    available: bool


class ModelsInfo(BaseModel):  # type: ignore[misc]
    """Installed / available models summary for health diagnostics."""

    total: int = 0
    available: int = 0
    models: list[ModelEntry] = Field(default_factory=list)
    error: str | None = None


class DiskEntry(BaseModel):  # type: ignore[misc]
    """Disk usage for one cache directory."""

    bytes: int
    human: str


class DiskUsage(BaseModel):  # type: ignore[misc]
    """Disk usage breakdown."""

    uploads: DiskEntry | None = None
    cache: DiskEntry | None = None
    models: DiskEntry | None = None
    exports: DiskEntry | None = None
    error: str | None = None


class AssistantStatus(BaseModel):  # type: ignore[misc]
    """Whether a conversational assistant (LLM provider) is reachable."""

    connected: bool = Field(description="At least one configured LLM provider is available")
    offline: bool = Field(description="No provider is available: analyses still run, answers come from the catalog")
    provider: str | None = Field(None, description="First available provider in LLM_PROVIDER_ORDER, e.g. 'gemini'")
    model: str | None = Field(None, description="That provider's model, e.g. 'gemini-2.5-flash'")


class HealthResponse(BaseModel):  # type: ignore[misc]
    """Health check response."""

    ok: bool = True
    service: str = "DT4LC"
    version: str = __version__
    assistant: AssistantStatus | None = None
    llm_providers: list[LLMProviderStatus] | None = None
    gee: GEEStatus | None = None
    models: ModelsInfo | None = None
    disk: DiskUsage | None = None


class CapabilitiesSummary(BaseModel):  # type: ignore[misc]
    """The server line: ``parts`` and their ``line`` (the deployment only when it is the public demo)."""

    deployment: Literal["demo", "local"] = Field(description="DT4LC_DEPLOYMENT (default local)")
    parts: dict[str, str] = Field(description="deployment, role, assistant, models, earth_engine labels")
    line: str = Field(description="e.g. 'public demo · guest access · Gemini 3.5 Flash connected · …'")


class CapabilitiesAssistant(BaseModel):  # type: ignore[misc]
    """The assistant from ``/v1/health`` plus its display name."""

    connected: bool
    provider: str | None = None
    model: str | None = None
    display: str | None = Field(None, description="e.g. 'Gemini 3.5 Flash'; null when offline")


class CapabilityTier(BaseModel):  # type: ignore[misc]
    """One tier of the Capabilities page; ``items`` differ per tier (see server/capabilities.py)."""

    id: Literal["ask", "composed", "builtin", "models", "data", "coming"]
    number: int
    title: str
    subtitle: str | None = None
    status: str = Field(description="ask: connected|offline; composed: available|offline; others available|coming")
    status_label: str | None = None
    needs_assistant: bool
    badge: str | None = None
    collapsed: bool = False
    groups: list[dict[str, str]] | None = Field(None, description="builtin: [{id, title}] in display order")
    note: str | None = None
    items: list[dict[str, Any]]


class GuestLimits(BaseModel):  # type: ignore[misc]
    analyses_per_hour: int
    kept_hours: int


class ExamplePrompt(BaseModel):  # type: ignore[misc]
    tier: str
    id: str
    prompt: str


class CapabilitiesResponse(BaseModel):  # type: ignore[misc]
    """What this server can do for the caller (``GET /v1/capabilities``), tier by tier."""

    summary: CapabilitiesSummary
    assistant: CapabilitiesAssistant
    offline: bool = Field(description="No assistant: show the banner and grey tiers 1-2")
    banner: str | None = None
    tiers: list[CapabilityTier]
    guest_limits: GuestLimits
    session: SessionResponse
    example_prompts: list[ExamplePrompt]


class RegistryModelsResponse(BaseModel):  # type: ignore[misc]
    """ML and hosted models from registries."""

    models: list[dict[str, Any]]
    count: int


class MetricsResponse(BaseModel):  # type: ignore[misc]
    """Execution and LLM usage metrics."""

    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_seconds: float
    total_llm_calls: int
    total_llm_tokens: int
    total_llm_cost: float
    llm_by_provider: dict[str, Any] = Field(default_factory=dict)


# --- ML model management ---


class MLModelsListResponse(BaseModel):  # type: ignore[misc]
    """List of downloadable ML models."""

    models: list[dict[str, Any]]
    cache_dir: str
    total_installed_mb: float


class MLModelDetailResponse(BaseModel):  # type: ignore[misc]
    """Detailed model metadata."""

    model_config = {"extra": "allow"}


class MLModelDownloadResponse(BaseModel):  # type: ignore[misc]
    """Background download started."""

    model_id: str
    status: str
    message: str
    size_mb: float | int


class MLModelActionResponse(BaseModel):  # type: ignore[misc]
    """Model cancel or delete action result."""

    model_config = {"extra": "allow"}


# --- Google Earth Engine ---


class GEEFetchResponse(BaseModel):  # type: ignore[misc]
    """GEE tile fetch response."""

    model_config = {"extra": "allow"}


class BulkFetchRequest(BaseModel):  # type: ignore[misc]
    """Bulk GEE fetch for pre/post periods."""

    bbox: list[float] = Field(description="Bounding box [minX, minY, maxX, maxY] in WGS84")
    dataset_id: str = Field(description="sentinel-2, modis, or landsat-8")
    bands: list[str] = Field(default_factory=list, description="Band IDs to fetch")
    indices: list[str] = Field(default_factory=list, description="Spectral indices: ndvi, ndwi, ndsi")
    pre_start: str = Field(description="Pre-period start YYYY-MM-DD")
    pre_end: str = Field(description="Pre-period end YYYY-MM-DD")
    post_start: str | None = Field(default=None, description="Post-period start YYYY-MM-DD")
    post_end: str | None = Field(default=None, description="Post-period end YYYY-MM-DD")
    cloud_cover_max: float = Field(default=20.0, ge=0, le=100)
    use_now: bool = Field(default=False, description="If true, post period is last 7 days")


class BulkFetchResponse(BaseModel):  # type: ignore[misc]
    """Bulk fetch result with layer metadata."""

    model_config = {"extra": "allow"}


class DatasetListResponse(BaseModel):  # type: ignore[misc]
    """Available GEE datasets."""

    ok: bool = True
    datasets: dict[str, Any]


class PersistLayerRequest(BaseModel):  # type: ignore[misc]
    """Persist GEE layer metadata for export."""

    layer_id: str = Field(min_length=1, description="Unique layer identifier")
    layer_name: str = ""
    dataset_id: str = ""
    bands: list[str] = Field(default_factory=list)
    indices: list[str] = Field(default_factory=list)
    period: str = Field(default="", description="pre or post")
    bbox: list[float] = Field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    tile_url: str = ""
    cloud_cover_max: float = 20.0


class PersistLayerResponse(BaseModel):  # type: ignore[misc]
    """Layer metadata saved."""

    ok: bool = True
    layer_id: str


class GEEExportResponse(BaseModel):  # type: ignore[misc]
    """GeoTIFF export for chat attachment."""

    model_config = {"extra": "allow"}


class GEEDatesResponse(BaseModel):  # type: ignore[misc]
    """Available Sentinel-2 acquisition dates."""

    model_config = {"extra": "allow"}


class GEELayersListResponse(BaseModel):  # type: ignore[misc]
    """Persisted GEE layers."""

    ok: bool = True
    layers: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class GEELayerIdResponse(BaseModel):  # type: ignore[misc]
    """Single-layer operation result."""

    ok: bool = True
    layer_id: str
