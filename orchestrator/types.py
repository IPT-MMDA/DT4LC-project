from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestrationContext:
    """Minimal context extracted from a user request.

    This mirrors the left side of the diagrams where the Cognitive UI/Server
    passes a prompt, selected area, and any attachments.
    """

    prompt: str
    selected_area: dict[str, Any] | None = None
    attached_files: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    """A single step within a pipeline plan."""

    id: str
    uses: str
    reads: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    with_params: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)


@dataclass
class PipelinePlan:
    """High-level pipeline specification created by the Planner Agent."""

    flow: str
    steps: list[PipelineStep]
    notes: str | None = None
