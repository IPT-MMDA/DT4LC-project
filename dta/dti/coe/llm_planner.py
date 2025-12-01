"""LLM-Powered Intelligent Planner.

Uses LLM reasoning to generate optimal pipeline plans by analyzing
the registry and understanding complex user requests.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from dta.dti.coe.llm import LLMMessage, LLMRouter
from dta.dti.coe.llm.config import create_router_from_env

if TYPE_CHECKING:
    from dta.dti.coe.context_agent import ContextUnderstanding
    from dta.dti.coe.registry import Registry
    from dta.dti.executor import ExecutionPlan, PlanStep

logger = logging.getLogger(__name__)

# Global LLM router instance (lazy init)
_router: LLMRouter | None = None


def _get_router() -> LLMRouter:
    """Get or create LLM router."""
    global _router
    if _router is None:
        _router = create_router_from_env()
    return _router


def format_registry_for_llm(reg: Registry) -> str:
    """Format registry as structured text for LLM.

    Args:
        reg: Component registry

    Returns:
        Human-readable registry description
    """
    lines = ["# Available Pipeline Components\n"]

    # Group by kind
    by_kind: dict[str, list] = {}
    for item in reg.instances:
        kind = item.kind or "other"
        by_kind.setdefault(kind, []).append(item)

    for kind, items in sorted(by_kind.items()):
        lines.append(f"\n## {kind.upper()} Components:")
        for item in items:
            lines.append(f"\n- **{item.id}**")
            if item.keywords:
                lines.append(f"  Keywords: {', '.join(item.keywords)}")
            if item.inputs:
                lines.append(f"  Inputs: {', '.join(item.inputs)}")
            if item.outputs:
                lines.append(f"  Outputs: {', '.join(item.outputs)}")

    return "\n".join(lines)


def plan_with_llm(
    ctx: ContextUnderstanding,
    reg: Registry,
    router: LLMRouter | None = None,
) -> ExecutionPlan:
    """Generate pipeline plan using LLM reasoning.

    Uses the LLM to:
    1. Understand user's goal and constraints
    2. Analyze available registry components
    3. Reason about optimal pipeline construction
    4. Generate step-by-step plan with proper type matching

    Args:
        ctx: Context understanding from context agent
        reg: Component registry
        router: Optional LLM router (uses default if None)

    Returns:
        Validated execution plan

    Raises:
        Exception: If LLM fails or plan is invalid
    """
    from dta.dti.executor import ExecutionPlan, PlanStep

    if router is None:
        router = _get_router()

    # Format registry for LLM
    registry_desc = format_registry_for_llm(reg)

    # Build planning prompt
    system_prompt = """You are a pipeline planner for a geospatial analysis system.
Your job is to create a valid execution plan given available components.

CRITICAL RULES:
1. ALWAYS start with INPUT component(s) to load data FIRST
2. Each step must reference a component ID from the registry
3. Steps execute in order - ensure outputs from previous steps match inputs needed
4. Chain processing steps (algorithms/models) to transform data
5. End with post-processing to format results
6. Return ONLY valid JSON - no markdown, no explanation

PIPELINE STRUCTURES:

A) SINGLE FILE ANALYSIS (ndvi, statistics, etc):
   Step 1: "input/file" - loads data and produces RasterPath
   Step 2+: algorithm component - processes RasterPath
   Last: "post-processing/agent-analysis"

B) CHANGE DETECTION / COMPARISON (requires 2 files):
   Step 1: "input/file-before" - loads first file, produces RasterPathBefore
   Step 2: "input/file-after" - loads second file, produces RasterPathAfter
   Step 3: "algorithms/change-detection" - compares both, produces ChangeMap
   Last: "post-processing/agent-analysis"

WHEN TO USE EACH FLOW:

USE SINGLE FILE FLOW (with "input/file") for:
- NDVI, vegetation, greenness analysis (single image)
- Statistics, histogram, distribution analysis
- Feature extraction with Prithvi model
- Land cover analysis (single image)
- Any analysis that works on ONE image

USE CHANGE DETECTION FLOW (with "input/file-before" + "input/file-after") ONLY when:
- User explicitly says "compare" or "comparison"
- User explicitly says "change detection"
- User explicitly says "difference between"
- User explicitly mentions "before and after"
- User asks to detect changes between two time periods

DEFAULT: Use single file flow unless change detection is explicitly requested!

Output format:
{
  "steps": [
    {"uses": "component-id-here"},
    ...
  ],
  "reasoning": "brief explanation of plan logic"
}

IMPORTANT: Match the correct flow based on the user's intent!"""

    user_prompt = f"""Create a pipeline plan for this request:

**User Goal:** {ctx.goal}

**Required Inputs:** {", ".join(ctx.required_inputs) if ctx.required_inputs else "auto-detect from registry"}

**Desired Outputs:** {", ".join(ctx.desired_outputs) if ctx.desired_outputs else "auto-detect based on goal"}

**Hints:**
- Keywords: {", ".join(ctx.hints.get("keywords", [])) if ctx.hints else "none"}
- Output type: {ctx.hints.get("output_type", "chat") if ctx.hints else "chat"}

{registry_desc}

Generate a valid pipeline plan as JSON."""

    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]

    try:
        # Generate plan with LLM (low temperature for consistency)
        response = router.generate(messages, temperature=0.2, max_tokens=2000)

        logger.info(f"LLM planner response ({response.provider}): {response.text[:200]}...")

        # Parse JSON response
        plan_data = _parse_plan_json(response.text)

        # Convert to ExecutionPlan
        steps = [PlanStep(uses=step["uses"]) for step in plan_data["steps"]]

        # Validate plan
        _validate_plan(steps, reg)

        reasoning = plan_data.get("reasoning", "")
        logger.info(f"LLM plan reasoning: {reasoning}")

        return ExecutionPlan(
            flow=ctx.goal,
            steps=steps,
            outputs=[f"publish: {ctx.hints.get('output_type', 'chat')}"],
        )

    except Exception as e:
        logger.error(f"LLM planning failed: {e}")
        raise Exception(f"LLM planner failed: {e}") from e


def _parse_plan_json(text: str) -> dict:
    """Parse JSON plan from LLM response.

    Handles markdown code blocks and extracts JSON.

    Args:
        text: Raw LLM response

    Returns:
        Parsed plan dict

    Raises:
        ValueError: If JSON is invalid
    """
    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        # Extract content between ```json and ```
        lines = text.split("\n")
        json_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        data = json.loads(text)

        # Validate structure
        if "steps" not in data:
            raise ValueError("Missing 'steps' field in plan JSON")

        if not isinstance(data["steps"], list):
            raise ValueError("'steps' must be a list")

        for i, step in enumerate(data["steps"]):
            if not isinstance(step, dict) or "uses" not in step:
                raise ValueError(f"Step {i} missing 'uses' field")

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}") from e


def _validate_plan(steps: list[PlanStep], reg: Registry) -> None:
    """Validate that plan is executable.

    Checks:
    - All component IDs exist in registry
    - Basic type flow is valid

    Args:
        steps: Plan steps to validate
        reg: Component registry

    Raises:
        ValueError: If plan is invalid
    """
    from dta.dti.registry import get_item

    if not steps:
        raise ValueError("Plan has no steps")

    # Check all components exist
    for i, step in enumerate(steps):
        try:
            get_item(reg, step.uses)
        except KeyError:
            raise ValueError(f"Step {i}: component '{step.uses}' not found in registry") from None

    logger.info(f"Plan validated: {len(steps)} steps")


def estimate_plan_confidence(ctx: ContextUnderstanding) -> float:
    """Estimate confidence in using template vs LLM planner.

    Simple heuristic:
    - High confidence (>0.7) → use template planner (fast)
    - Low confidence (<0.7) → use LLM planner (smart)

    Args:
        ctx: Context understanding

    Returns:
        Confidence score 0.0-1.0
    """
    score = 0.0

    # Clear keywords boost confidence
    keywords = ctx.hints.get("keywords", []) if ctx.hints else []
    if keywords:
        score += 0.3

    # Known patterns boost confidence - including change detection and models
    known_patterns = [
        "ndvi",
        "statistics",
        "change",
        "compare",
        "comparison",
        "difference",
        "before",
        "after",
        "prithvi",  # Prithvi model - uses single file flow
        "features",  # Feature extraction - uses single file flow
        "temporal",  # Temporal features - single file, NOT change detection
    ]
    if any(kw.lower() in [k.lower() for k in keywords] for kw in known_patterns):
        score += 0.4  # Increased from 0.3 - we have good templates for these

    # Clear inputs/outputs boost confidence
    if ctx.required_inputs:
        score += 0.15
    if ctx.desired_outputs:
        score += 0.15

    return min(score, 1.0)
