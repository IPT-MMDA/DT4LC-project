"""Hybrid Planner Agent - Template + LLM Planning.

Intelligently chooses between:
- Template-based planning (fast, for simple/common requests)
- LLM-powered planning (smart, for complex/ambiguous requests)
"""

import logging

from dta.dti.registry import find_items_by_keywords, find_items_producing
from dta.dti.schemas import ContextUnderstanding, ExecutionPlan, PlanStep, Registry

logger = logging.getLogger(__name__)

# Confidence threshold for using template planner
TEMPLATE_CONFIDENCE_THRESHOLD = 0.7


def plan(ctx: ContextUnderstanding, reg: Registry, use_llm: bool = True) -> ExecutionPlan:
    """Hybrid planner - chooses between template and LLM.

    Args:
        ctx: Context understanding from context agent
        reg: Component registry
        use_llm: Enable LLM planning (default True, set False to force template)

    Returns:
        Execution plan
    """
    if use_llm:
        # Try LLM planner first if confidence is low
        try:
            from dta.dti.coe.llm_planner import estimate_plan_confidence, plan_with_llm

            confidence = estimate_plan_confidence(ctx)
            logger.info(f"Planning confidence score: {confidence:.2f}")

            if confidence < TEMPLATE_CONFIDENCE_THRESHOLD:
                logger.info("Using LLM planner (low confidence in template)")
                return plan_with_llm(ctx, reg)
            else:
                logger.info("Using template planner (high confidence)")

        except Exception as e:
            logger.warning(f"LLM planner unavailable, falling back to template: {e}")

    # Fall back to template planner
    return plan_template(ctx, reg)


def plan_template(ctx: ContextUnderstanding, reg: Registry) -> ExecutionPlan:
    """Template-based planning using keyword matching.

    Fast but limited - works for common patterns like:
    - "ndvi on kahovka data"
    - "statistics on uploaded raster"

    Args:
        ctx: Context understanding
        reg: Component registry

    Returns:
        Execution plan
    """
    steps: list[PlanStep] = []

    # 1) ensure inputs exist; if user attached raster, use "input/file"
    for need in ctx.required_inputs or []:
        # naive: prefer passthrough inputs that produce need
        for it in reg.instances:
            if it.kind == "input" and need in it.outputs:
                steps.append(PlanStep(uses=it.id))
                break

    # 2) choose a chain to reach desired outputs
    kw_ranked = find_items_by_keywords(reg, ctx.hints.get("keywords", []))
    for want in ctx.desired_outputs or []:
        candidates = [i for i in kw_ranked if want in i.outputs] or find_items_producing(reg, want)
        if not candidates:
            continue
        chosen = candidates[0]
        steps.append(PlanStep(uses=chosen.id))

    # 3) optional postprocess if LLM summary is desired
    for it in kw_ranked:
        if it.kind == "postprocess":
            steps.append(PlanStep(uses=it.id))
            break

    plan_obj = ExecutionPlan(
        flow=ctx.goal,
        steps=steps,
        outputs=["publish: chat"],
    )

    logger.info(f"Template plan: {len(steps)} steps")
    return plan_obj
