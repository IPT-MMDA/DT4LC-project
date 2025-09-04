"""Context Orchestration Engine (COE).

This package provides lightweight agent primitives to parse a user request,
draft a pipeline plan, and validate it before delegating execution to the
Digital Twin Aggregator (DTA).

The design mirrors the attached architecture diagrams and intentionally keeps
dependencies minimal. Agents communicate via simple, typed Python objects so
the system can run without an external LLM while remaining LLM-ready.
"""

from .types import (
    OrchestrationContext,
    PipelinePlan,
    PipelineStep,
)
from .agents import (
    ContextUnderstandingAgent,
    DecisionMakingAgent,
    PlannerAgent,
    orchestrate_request,
)

__all__ = [
    "OrchestrationContext",
    "PipelinePlan",
    "PipelineStep",
    "ContextUnderstandingAgent",
    "DecisionMakingAgent",
    "PlannerAgent",
    "orchestrate_request",
]
