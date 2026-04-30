"""Hybrid Planner Agent - Template + LLM Planning.

Intelligently chooses between:
- Template-based planning (fast, for simple/common requests)
- LLM-powered planning (smart, for complex/ambiguous requests)
"""

import logging

from dta.dti.registry import find_items_by_keywords, find_items_producing
from dta.dti.schemas import ContextUnderstanding, ExecutionPlan, PlanStep, Registry, RegistryItem

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


def _get_change_detection_item(reg: Registry) -> RegistryItem | None:
    """Return the change-detection registry item, or None if missing.

    Test registries (e.g. test_planner.py) may omit it; callers must
    tolerate None and fall back to "no change-detection signal".
    """
    for it in reg.instances:
        if it.id == "algorithms/change-detection":
            return it
    return None


def _is_change_detection_request(ctx: ContextUnderstanding, reg: Registry) -> bool:
    """Check if request is for change detection (comparing two images).

    Trigger keywords come from ``algorithms/change-detection.triggers.keywords``
    in the registry. Adding a new change-detection trigger phrase = a YAML edit;
    no code change. Two structural signals stay in code (they're not item-
    specific knowledge): "before AND after" co-occurrence, and the user
    explicitly requesting ChangeMap as a desired output.
    """
    item = _get_change_detection_item(reg)
    triggers_kw: list[str] = []
    if item is not None and item.triggers is not None:
        triggers_kw = [k.lower() for k in item.triggers.keywords]

    keywords = ctx.hints.get("keywords", []) if ctx.hints else []
    goal_lower = (ctx.goal or "").lower()
    keywords_lower = " ".join(kw.lower() for kw in keywords)

    has_strong_signal = any(kw in goal_lower or kw in keywords_lower for kw in triggers_kw)

    # Structural signals — generic, not algorithm-specific:
    # "before" + "after" co-occurrence, or explicit ChangeMap desired output.
    has_before_after = ("before" in goal_lower and "after" in goal_lower) or (
        "before" in keywords_lower and "after" in keywords_lower
    )
    wants_changemap = "ChangeMap" in (ctx.desired_outputs or [])

    return has_strong_signal or has_before_after or wants_changemap


def _detect_index_type(ctx: ContextUnderstanding, reg: Registry) -> str:
    """Detect which index type to use for change detection.

    Keyword sets per index come from
    ``algorithms/change-detection.config.index_keyword_map`` in the registry.
    Adding a new index = a YAML edit to that map; no code change.

    Resolution order:
        1. Explicit hint ``ctx.hints["index_type"]`` if it's a known map key.
        2. First map key whose keyword set hits the goal/keywords blob.
        3. ``config["default_index_type"]`` (or "ndvi" if unset).
    """
    item = _get_change_detection_item(reg)
    config = item.config if item is not None else {}
    index_map: dict[str, list[str]] = config.get("index_keyword_map", {})
    default_index = str(config.get("default_index_type", "ndvi"))

    # Explicit hint wins, but only if it names a known index in the map.
    if ctx.hints and ctx.hints.get("index_type"):
        index_type = str(ctx.hints["index_type"]).lower()
        if index_type in index_map:
            return index_type

    keywords = ctx.hints.get("keywords", []) if ctx.hints else []
    goal_lower = (ctx.goal or "").lower()
    keywords_lower = " ".join(kw.lower() for kw in keywords)
    combined = goal_lower + " " + keywords_lower

    for index_name, kws in index_map.items():
        if any(kw.lower() in combined for kw in kws):
            logger.info(f"Detected {index_name.upper()} change detection from registry triggers")
            return index_name

    logger.info(f"Defaulting to {default_index.upper()} change detection (registry default)")
    return default_index


def plan_template(ctx: ContextUnderstanding, reg: Registry) -> ExecutionPlan:
    """Template-based planning using keyword matching.

    Fast but limited - works for common patterns like:
    - "ndvi on kahovka data"
    - "statistics on uploaded raster"
    - "compare before and after images"

    Args:
        ctx: Context understanding
        reg: Component registry

    Returns:
        Execution plan
    """
    steps: list[PlanStep] = []

    # Check if this is a change detection request (needs two files)
    is_change_detection = _is_change_detection_request(ctx, reg)

    if is_change_detection:
        # Change detection flow: two input files + change detection algorithm
        logger.info("Detected change detection request - using dual-file input")

        # Detect which index type to use
        index_type = _detect_index_type(ctx, reg)
        logger.info(f"Using index type: {index_type}")

        # Add before/after input steps
        steps.append(PlanStep(uses="input/file-before", binds={}))
        steps.append(PlanStep(uses="input/file-after", binds={}))

        # Add change detection algorithm with index type
        steps.append(PlanStep(uses="algorithms/change-detection", binds={"IndexType": index_type}))

        # Add post-processing
        for it in reg.instances:
            if it.kind == "postprocess":
                steps.append(PlanStep(uses=it.id))
                break

        plan_obj = ExecutionPlan(
            flow=ctx.goal,
            steps=steps,
            outputs=["publish: chat"],
        )
        logger.info(f"Change detection plan: {len(steps)} steps, index_type={index_type}")
        return plan_obj

    # Standard single-file flow
    # 1) First, determine the best matching algorithm/model via keywords
    # We need this BEFORE deciding on data loader because the matched item
    # determines whether we need inputs
    kw_ranked = find_items_by_keywords(reg, ctx.hints.get("keywords", []))
    keywords = ctx.hints.get("keywords", []) if ctx.hints else []
    keywords_lower = [kw.lower() for kw in keywords]
    goal_lower = (ctx.goal or "").lower()

    # Find best matching algorithm/model
    best_item = None
    best_score = 0
    for item in kw_ranked:
        if item.kind in ("algorithm", "model"):
            item_keywords = [k.lower() for k in (item.keywords or [])]
            match_count = sum(1 for kw in item_keywords if kw in goal_lower or any(kw in uk for uk in keywords_lower))
            id_name = item.id.split("/")[-1].lower()
            if id_name in goal_lower:
                match_count += 5
            if match_count > best_score:
                best_score = match_count
                best_item = item

    # 2) Check if we need a data loader
    needs_data_loader = False
    required_inputs = ctx.required_inputs or []

    # Check if any desired algorithms/models need inputs
    for want in ctx.desired_outputs or []:
        for item in reg.instances:
            if want in item.outputs and item.inputs:
                needs_data_loader = True
                break

    # Also check if the best matched item needs inputs
    if best_item and best_item.inputs:
        needs_data_loader = True
        logger.debug(f"Best matched item {best_item.id} requires inputs: {best_item.inputs}")

    # If we need data or have required inputs, add input/file loader
    if (
        needs_data_loader
        or required_inputs
        or any(kw in ctx.hints.get("keywords", []) for kw in ["kahovka", "data", "raster", "load"])
    ):
        for it in reg.instances:
            if it.kind == "input" and it.id == "input/file":
                # Add input/file step without binds - the file path will be provided
                # by the frontend/API when the user uploads a file
                steps.append(PlanStep(uses=it.id, binds={}))
                logger.info("Added data loader step: input/file (waiting for user upload)")
                break

    # 3) Add the matched algorithm/model step
    added_step = False

    if best_item and best_score > 0:
        steps.append(PlanStep(uses=best_item.id))
        added_step = True
        logger.info(f"Added step from keyword match (score={best_score}): {best_item.id}")

    # If no keyword match, try desired outputs
    if not added_step:
        for want in ctx.desired_outputs or []:
            candidates = [i for i in kw_ranked if want in i.outputs] or find_items_producing(reg, want)
            if not candidates:
                continue
            chosen = candidates[0]
            steps.append(PlanStep(uses=chosen.id))
            added_step = True

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
