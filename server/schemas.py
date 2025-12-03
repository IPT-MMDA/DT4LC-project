"""Pydantic schemas for the FastAPI server.

Defines request/response models for the REST API endpoints including
job submission, chat messages, and file attachments.
"""

from typing import Any, Literal

from pydantic import BaseModel

# Re-export Attachment from domain schemas to avoid duplication
from dta.dti.schemas import Attachment

__all__ = [
    "Attachment",
    "ChatMessage",
    "ChatRequest",
    "CreateJobRequest",
    "JobStatus",
    "JobSubmitRequest",
    "Plan",
]

Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """Request containing chat message history."""

    messages: list[ChatMessage]


class JobSubmitRequest(BaseModel):
    """Request for submitting a new job."""

    prompt: str
    mode: str = "hybrid"  # hybrid/llm/template
    attachments: list[Attachment] = []
    context: dict[str, Any] | None = None


class Plan(BaseModel):
    """Execution plan for a pipeline of analysis steps."""

    tags: list[str] = []
    goals: list[str] = []
    pipeline: list[str] = []  # tool ids
    inputs: dict[str, Any] = {}  # e.g., {"file_path": "..."}
    meta: dict[str, Any] = {}


class CreateJobRequest(BaseModel):
    """Request to create a job from a pre-defined plan."""

    plan: Plan


class JobStatus(BaseModel):
    """Current status and results of a job."""

    id: str
    state: Literal["queued", "running", "succeeded", "failed"] = "queued"
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
