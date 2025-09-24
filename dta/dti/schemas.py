from typing import Any, Literal

from pydantic import BaseModel, Field


# --- HTTP payloads ---
class Attachment(BaseModel):  # type: ignore[misc]
    id: str
    filename: str
    mime_type: str = Field(..., description="e.g., image/jpeg, image/tiff")
    path: str | None = None  # local temp path
    url: str | None = None  # remote URL (optional)
    size_bytes: int | None = None


class ChatRequest(BaseModel):  # type: ignore[misc]
    prompt: str
    attachments: list[Attachment] = []
    # room to grow:
    metadata: dict[str, Any] = {}


# --- Registry in-memory models ---
class Runner(BaseModel):  # type: ignore[misc]
    type: Literal["python", "agent", "passthrough"]
    entrypoint: str | None = None
    env: dict[str, str] = {}


class RegistryItem(BaseModel):  # type: ignore[misc]
    id: str
    kind: Literal["input", "algorithm", "model", "postprocess"]
    keywords: list[str] = []
    inputs: list[str] = []
    outputs: list[str] = []
    runner: Runner


class Registry(BaseModel):  # type: ignore[misc]
    version: str
    types: list[str]
    instances: list[RegistryItem]


# --- Planning / execution models ---
class PlanStep(BaseModel):  # type: ignore[misc]
    uses: str  # registry id (e.g., "algorithms/ndvi")
    binds: dict[str, str] = {}  # type -> source alias


class ExecutionPlan(BaseModel):  # type: ignore[misc]
    flow: str = "auto"
    steps: list[PlanStep]
    outputs: list[str] = []  # friendly names, topics, channels, etc.


class ContextUnderstanding(BaseModel):  # type: ignore[misc]
    goal: str  # compact intent
    desired_outputs: list[str]  # registry type names we want (e.g., ["NDVIMap"])
    required_inputs: list[str]  # e.g., ["RasterPath"]
    hints: dict[str, Any] = {}  # keywords, tags, georegion, horizon, etc.
