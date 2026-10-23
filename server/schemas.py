"""Pydantic schemas for the FastAPI server.

Defines request/response models for the REST API endpoints including
job submission, chat messages, and file attachments.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

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
    "JobStatus",
    "JobSubmitRequest",
    "MetricsResponse",
    "MLModelActionResponse",
    "MLModelDetailResponse",
    "MLModelDownloadResponse",
    "MLModelsListResponse",
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


class JobSubmitRequest(BaseModel):  # type: ignore[misc]
    """Request for submitting a new job."""

    prompt: str = Field(description="Natural-language analysis request")
    mode: str = Field(default="hybrid", description="Planning mode: hybrid, llm, or template")
    attachments: list[Attachment] = Field(default_factory=list, description="Uploaded GeoTIFF attachments")
    context: dict[str, Any] | None = Field(default=None, description="Optional session context for follow-up requests")


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


class JobSummary(BaseModel):  # type: ignore[misc]
    """Async job status without results (``GET /v1/jobs`` list item)."""

    id: str
    status: str = Field(description="pending, running, completed, failed, or cancelled")
    prompt: str
    plan: dict[str, Any] | None = None
    progress: float = Field(ge=0.0, le=1.0)
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    owner: str | None = Field(default=None, description="Submitter's session id; null for jobs from before v0.6.0")


class JobRecord(JobSummary):
    """Async job status and results (``GET /v1/jobs/{id}``)."""

    result: dict[str, Any] | None = Field(
        default=None, description="Pipeline result; files are referenced as {kind, path, url, stats}"
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Event log as replayed by GET /v1/jobs/{id}/events: [{seq, type, data}] (server/job_events.py)",
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


class FileInfo(BaseModel):  # type: ignore[misc]
    """Metadata for an on-disk GeoTIFF."""

    id: str
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


class CapabilitiesResponse(BaseModel):  # type: ignore[misc]
    """Component registry snapshot."""

    version: str
    types: list[str]
    instances: list[dict[str, Any]]
    count: int


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
