import logging
from pathlib import Path
import re
from typing import Any

from dta.dti.data_sources.gee_region import inject_region_fetch
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Attachment, ChatRequest, ExecutionPlan, Registry

from .context_agent import analyze
from .intent_classifier import IntentType, classify_intent
from .named_indices import named_index_request, proposal_text
from .plan_validator import PlanError, validate
from .planner import plan
from .prompt_dates import attachments_named_by_date

logger = logging.getLogger(__name__)


def orchestrate(req: ChatRequest) -> dict[str, Any]:
    """Orchestrate the planning and validation of a pipeline request.

    First classifies user intent:
    - CONVERSATION: Returns helpful response without pipeline execution
    - PIPELINE: Plans and validates the data processing pipeline

    Handles both single-file and multi-file (change detection) requests. Without attachments,
    ``req.metadata["region"]`` (a job's region echo) replaces the file inputs with Earth Engine
    fetches of that region (:func:`~dta.dti.data_sources.gee_region.inject_region_fetch`).

    Args:
        req: Chat request with prompt and optional attachments

    Returns:
        Dictionary with:
            - ok: True if successful
            - intent: "pipeline" or "conversation"
            - plan: Execution plan (if pipeline); ``plan["composed"]`` when it is not a single template
            - proposal: Sentence offering a composed plan for confirmation (if composed)
            - response: Helpful response (if conversation)
            - rule: The conversational rule that matched, e.g. ``"missing_input"`` (if conversation)
            - error: Error message if failed (also when the plan has no analysis step, or
              names a band the input lacks: no template retry then)
    """
    # First, classify the user's intent
    intent_result = classify_intent(req)
    intent = intent_result.get("intent", IntentType.PIPELINE)

    logger.info(f"Intent classification: {intent} - {intent_result.get('reason', '')}")

    # If conversational, return helpful response without running pipeline
    if intent == IntentType.CONVERSATION:
        return {
            "ok": True,
            "intent": "conversation",
            "response": intent_result.get("response", "How can I help you with your geospatial analysis?"),
            "reason": intent_result.get("reason", ""),
            "rule": intent_result.get("rule"),
        }

    # Continue with pipeline planning
    reg = load_registry()

    # A named index no built-in analysis computes (NDMI, NBR, ...): composed from the table, or
    # refused when a file lacks its bands, before any LLM or template planning.
    named = named_index_request(req.prompt, req.attachments, reg)
    if named is not None:
        if named.missing:
            logger.info(f"Named index {named.index.name} refused: {named.missing}")
            return {"ok": False, "intent": "pipeline", "error": named.missing, "candidate": named.plan.model_dump()}
        try:
            final_plan = validate(named.plan, reg)
        except PlanError as e:
            return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": named.plan.model_dump()}
        return {"ok": True, "intent": "pipeline", "plan": final_plan.model_dump(), "proposal": named.text}

    ctx = analyze(req, registry_types=reg.types)
    candidate = plan(ctx, reg)
    # A region sent from the map (the job queue passes it only when no file is attached): fetch it.
    region = req.metadata.get("region") if not req.attachments else None

    # Inject file paths from attachments into input steps
    if req.attachments:
        _inject_file_bindings(candidate, req.attachments, req.prompt)
    elif region:
        inject_region_fetch(candidate, region)
    else:
        logger.warning("No attachments provided - input steps will have no file bindings")

    try:
        final_plan = require_analysis_step(validate(candidate, reg), reg)
    except PlanError as e:
        if BAND_NOT_IN_FILE.search(str(e)):
            # The plan names a band the file lacks: say so; an unrelated template plan would not help.
            logger.info(f"Plan needs a band the input lacks ({e}); not retrying with the template planner")
            return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": candidate.model_dump()}
        # Retry once with the template planner; keep the original error if it doesn't help.
        fallback = plan(ctx, reg, use_llm=False)
        if req.attachments:
            _inject_file_bindings(fallback, req.attachments, req.prompt)
        elif region:
            inject_region_fetch(fallback, region)
        if fallback.model_dump() == candidate.model_dump():
            logger.debug("Template plan equals the rejected LLM plan; returning original error")
            return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": candidate.model_dump()}
        try:
            final_plan = validate(fallback, reg)
        except PlanError as e2:
            logger.warning(f"Template fallback also failed validation ({e2}); returning original error")
            return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": candidate.model_dump()}
        # A template plan with only input/post-processing steps would silently drop the user's goal.
        if not _analysis_steps(final_plan, reg):
            logger.warning(
                f"Template fallback has no algorithm or model step ({[s.uses for s in final_plan.steps]}); "
                "returning original error"
            )
            return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": candidate.model_dump()}
        logger.warning(
            f"LLM plan failed validation ({e}); using template plan instead: {[s.uses for s in final_plan.steps]}"
        )

    result: dict[str, Any] = {"ok": True, "intent": "pipeline"}
    if is_composed(final_plan, reg):
        final_plan.composed = True
        result["proposal"] = proposal_text(final_plan, reg)
    result["plan"] = final_plan.model_dump()
    return result


# The band-math validator's message for a band the input file lacks (dta.dti.algorithms.band_math).
BAND_NOT_IN_FILE = re.compile(r"Band \w+ is not in this file")
NO_ANALYSIS_STEP = (
    "I can't map this request to an analysis I can run here: no built-in analysis matches it, "
    "and I couldn't compose one."
)


def _analysis_steps(plan: ExecutionPlan, reg: Registry) -> list[str]:
    return [s.uses for s in plan.steps if get_item(reg, s.uses).kind in ("algorithm", "model")]


def require_analysis_step(plan: ExecutionPlan, reg: Registry) -> ExecutionPlan:
    """``plan``, unless it has no algorithm or model step (only reads and summarises): PlanError."""
    if not _analysis_steps(plan, reg):
        raise PlanError(NO_ANALYSIS_STEP)
    return plan


def is_composed(plan: ExecutionPlan, reg: Registry) -> bool:
    """True unless ``plan`` is a single registered template: exactly one analysis step, not band math."""
    analysis = _analysis_steps(plan, reg)
    return plan.composed or len(analysis) != 1 or analysis[0].startswith("algorithms/band-math")


def single_image_attachment(attachments: list[Attachment], prompt: str = "") -> Attachment:
    """The file a single-image analysis reads when several are attached: the only one whose
    ``metadata.date`` the prompt names ("NDVI for 1 June"), else the one with the latest
    ``metadata.date`` (ISO dates), else the last attachment. On the sample pair that is 9 June."""
    named = attachments_named_by_date(prompt, attachments)
    if len(named) == 1:
        return named[0]
    dated = [(a.metadata.date, i) for i, a in enumerate(attachments) if a.metadata and a.metadata.date]
    return attachments[max(dated)[1]] if dated else attachments[-1]


def _inject_file_bindings(candidate: Any, attachments: list[Any], prompt: str = "") -> None:
    """Inject file paths from attachments into input step bindings.

    Handles multiple attachment scenarios:
    - Single file: binds to input/file step
    - Two files: binds to input/file-before and input/file-after steps
    - Two or more files and a plan with one input/file step and no before/after steps (a
      single-image analysis): binds :func:`single_image_attachment`
    - A before/after plan whose prompt names exactly two attachments' dates ("between 1 and
      9 June"): binds those two, the earlier as before

    Args:
        candidate: Execution plan candidate
        attachments: List of file attachments
        prompt: The request, for the dates it names
    """
    uses = [step.uses for step in candidate.steps]
    two_dates = "input/file-before" in uses or "input/file-after" in uses
    if len(attachments) > 1 and uses.count("input/file") == 1 and not two_dates:
        attachments = [single_image_attachment(attachments, prompt)]
    elif two_dates and len(named := attachments_named_by_date(prompt, attachments)) == 2:
        attachments = named

    # Track which attachments have been used
    attachment_index = 0

    for step in candidate.steps:
        if attachment_index >= len(attachments):
            break

        # Single file input
        if step.uses == "input/file":
            if attachments[attachment_index].path:
                if not Path(attachments[attachment_index].path).exists():
                    logger.warning(f"Attachment file not found: {attachments[attachment_index].path}")
                step.binds["RasterPath"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPath: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"Attachment {attachment_index} has no path!")

        # Before file for change detection
        elif step.uses == "input/file-before":
            if attachments[attachment_index].path:
                if not Path(attachments[attachment_index].path).exists():
                    logger.warning(f"Attachment file not found: {attachments[attachment_index].path}")
                step.binds["RasterPathBefore"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPathBefore: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"Attachment {attachment_index} has no path!")

        # After file for change detection
        elif step.uses == "input/file-after":
            if attachment_index < len(attachments) and attachments[attachment_index].path:
                if not Path(attachments[attachment_index].path).exists():
                    logger.warning(f"Attachment file not found: {attachments[attachment_index].path}")
                step.binds["RasterPathAfter"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPathAfter: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"No attachment available for after file (index {attachment_index})")
