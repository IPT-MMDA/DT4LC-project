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
    function: str | None = None  # Function name to call (default: run or main)
    args_map: dict[str, Any] | None = None  # Map function args to values/variables
    env: dict[str, str] = {}


class Integration(BaseModel):  # type: ignore[misc]
    """External model integration configuration."""

    type: str  # e.g., "huggingface-spaces", "google-earth-engine"
    url: str
    status: Literal["planned", "active", "deprecated"] = "planned"
    requires: list[str] = []  # required packages


class PreprocessorRef(BaseModel):  # type: ignore[misc]
    """Reference to a preprocessor to apply before execution."""

    id: str  # Registry ID of the preprocessor (e.g., "preprocessors/scale-to-hls")
    apply_to: str  # Input type to transform (e.g., "RasterPath")


# --- Registry-driven extension models ---------------------------------
# Per-item natural-language hooks, response templates, parameterized config,
# and (reserved) plugin-source pointers — letting users extend the system by
# editing registry.yaml rather than touching the orchestration code.


class Triggers(BaseModel):  # type: ignore[misc]
    """Natural-language hooks driving intent classification and planning.

    Loaded into the intent classifier's keyword/action-phrase index at
    startup. Adding a new algorithm = filling these on its registry item;
    no edit to intent_classifier.py needed.
    """

    keywords: list[str] = []  # nouns/topics: "ndvi", "vegetation", "glacier change"
    action_phrases: list[str] = []  # imperatives: "calculate ndvi", "detect boundaries"


class UserGuide(BaseModel):  # type: ignore[misc]
    """Human-readable strings the system surfaces for this item.

    Used by the intent classifier to render per-algorithm responses
    (capability_response when the user asks "can you...?", missing_file_response
    when an action is requested without an upload).
    """

    capability_response: str | None = None
    missing_file_response: str | None = None
    summary_template: str | None = None  # one-line template used post-execution


class Source(BaseModel):  # type: ignore[misc]
    """Third-party plugin source (reserved for a future plugin loader).

    Indicates the algorithm/model isn't bundled in this repo and must be
    fetched/installed before its runner.entrypoint resolves. Currently
    declarative-only; no consumer wired in yet.
    """

    type: Literal["github", "huggingface", "local-path"] = "local-path"
    repo: str | None = None
    ref: str | None = None
    install: str | None = None


class RegistryItem(BaseModel):  # type: ignore[misc]
    id: str
    kind: Literal["input", "algorithm", "model", "postprocess", "preprocessor"]
    keywords: list[str] = []
    inputs: list[str] = []
    outputs: list[str] = []
    runner: Runner | None = None  # Optional for hosted models
    description: str | None = None  # Optional description
    interpretation: str | None = None  # Domain-specific interpretation guide for LLM analysis
    preprocessors: list[PreprocessorRef] = []  # Preprocessors to apply before execution
    integration: Integration | None = None  # For hosted models (HuggingFace, GEE, etc.)
    metadata: dict[str, Any] = {}  # Additional metadata (team, author, hosting, etc.)

    # Registry-driven extension fields. See Triggers / UserGuide / Source above.
    display_name: str | None = None  # human-readable label for UI / responses
    triggers: Triggers | None = None  # consumed by the intent classifier
    user_guide: UserGuide | None = None  # capability / missing-file responses
    config: dict[str, Any] = {}  # runner-consumed config block (formula, bands, colormap, ...)
    model_id: str | None = None  # model-manager id for locally-managed models
    source: Source | None = None  # third-party plugin source (reserved)


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
