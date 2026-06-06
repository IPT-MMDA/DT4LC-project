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

Role = Literal["user", "assistant"]


class ErrorCode(str, Enum):
    """Standardised error codes for API responses."""

    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):  # type: ignore[misc]
    """Structured error information."""

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):  # type: ignore[misc]
    """Standard error response envelope."""

    ok: Literal[False] = False
    error: ErrorDetail


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
