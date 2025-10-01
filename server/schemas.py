from typing import Any, Literal

from pydantic import BaseModel

Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class JobSubmitRequest(BaseModel):
    """Request for submitting a new job."""

    prompt: str
    mode: str = "hybrid"  # hybrid/llm/template
    context: dict[str, Any] | None = None


class Plan(BaseModel):
    tags: list[str] = []
    goals: list[str] = []
    pipeline: list[str] = []  # tool ids
    inputs: dict[str, Any] = {}  # e.g., {"file_path": "..."}
    meta: dict[str, Any] = {}


class CreateJobRequest(BaseModel):
    plan: Plan


class JobStatus(BaseModel):
    id: str
    state: Literal["queued", "running", "succeeded", "failed"] = "queued"
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
