"""Pydantic schemas for the FastAPI server.

Defines request/response models for the REST API endpoints including
job submission, chat messages, and file attachments.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

# Re-export Attachment from domain schemas to avoid duplication
from dta.dti.schemas import Attachment

__all__ = [
    "Attachment",
    "ChatMessage",
    "ChatRequest",
    "CreateJobRequest",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "JobStatus",
    "JobSubmitRequest",
    "Plan",
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

    role: Role
    content: str


class ChatRequest(BaseModel):  # type: ignore[misc]
    """Request containing chat message history."""

    messages: list[ChatMessage]


class JobSubmitRequest(BaseModel):  # type: ignore[misc]
    """Request for submitting a new job."""

    prompt: str
    mode: str = "hybrid"  # hybrid/llm/template
    attachments: list[Attachment] = []
    context: dict[str, Any] | None = None


class Plan(BaseModel):  # type: ignore[misc]
    """Execution plan for a pipeline of analysis steps."""

    tags: list[str] = []
    goals: list[str] = []
    pipeline: list[str] = []  # tool ids
    inputs: dict[str, Any] = {}  # e.g., {"file_path": "..."}
    meta: dict[str, Any] = {}


class CreateJobRequest(BaseModel):  # type: ignore[misc]
    """Request to create a job from a pre-defined plan."""

    plan: Plan


class JobStatus(BaseModel):  # type: ignore[misc]
    """Current status and results of a job."""

    id: str
    state: Literal["queued", "running", "succeeded", "failed"] = "queued"
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


# Health endpoint response schemas


class LLMProviderStatus(BaseModel):  # type: ignore[misc]
    name: str | None = None
    model: str | None = None
    available: bool | None = None
    error: str | None = None


class GEEStatus(BaseModel):  # type: ignore[misc]
    initialized: bool
    service_account_configured: bool
    error: str | None = None


class ModelEntry(BaseModel):  # type: ignore[misc]
    id: str
    available: bool


class ModelsInfo(BaseModel):  # type: ignore[misc]
    total: int = 0
    available: int = 0
    models: list[ModelEntry] = []
    error: str | None = None


class DiskEntry(BaseModel):  # type: ignore[misc]
    bytes: int
    human: str


class DiskUsage(BaseModel):  # type: ignore[misc]
    uploads: DiskEntry | None = None
    cache: DiskEntry | None = None
    models: DiskEntry | None = None
    exports: DiskEntry | None = None
    error: str | None = None


class HealthResponse(BaseModel):  # type: ignore[misc]
    ok: bool = True
    service: str = "DT4LC"
    version: str = "1.0.0"
    llm_providers: list[LLMProviderStatus] | None = None
    gee: GEEStatus | None = None
    models: ModelsInfo | None = None
    disk: DiskUsage | None = None
