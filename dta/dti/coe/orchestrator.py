import logging
from typing import Any

from dta.dti.registry import load_registry
from dta.dti.schemas import ChatRequest

from .context_agent import analyze
from .intent_classifier import IntentType, classify_intent
from .plan_validator import PlanError, validate
from .planner import plan

logger = logging.getLogger(__name__)


def orchestrate(req: ChatRequest) -> dict[str, Any]:
    """Orchestrate the planning and validation of a pipeline request.

    First classifies user intent:
    - CONVERSATION: Returns helpful response without pipeline execution
    - PIPELINE: Plans and validates the data processing pipeline

    Handles both single-file and multi-file (change detection) requests.

    Args:
        req: Chat request with prompt and optional attachments

    Returns:
        Dictionary with:
            - ok: True if successful
            - intent: "pipeline" or "conversation"
            - plan: Execution plan (if pipeline)
            - response: Helpful response (if conversation)
            - error: Error message if failed
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
        }

    # Continue with pipeline planning
    reg = load_registry()
    ctx = analyze(req, registry_types=reg.types)
    candidate = plan(ctx, reg)

    # Inject file paths from attachments into input steps
    if req.attachments:
        _inject_file_bindings(candidate, req.attachments)
    else:
        logger.warning("No attachments provided - input steps will have no file bindings")

    try:
        final_plan = validate(candidate, reg)
        return {"ok": True, "intent": "pipeline", "plan": final_plan.model_dump()}
    except PlanError as e:
        return {"ok": False, "intent": "pipeline", "error": str(e), "candidate": candidate.model_dump()}


def _inject_file_bindings(candidate: Any, attachments: list) -> None:
    """Inject file paths from attachments into input step bindings.

    Handles multiple attachment scenarios:
    - Single file: binds to input/file step
    - Two files: binds to input/file-before and input/file-after steps

    Args:
        candidate: Execution plan candidate
        attachments: List of file attachments
    """
    # Track which attachments have been used
    attachment_index = 0

    for step in candidate.steps:
        if attachment_index >= len(attachments):
            break

        # Single file input
        if step.uses == "input/file":
            if attachments[attachment_index].path:
                step.binds["RasterPath"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPath: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"Attachment {attachment_index} has no path!")

        # Before file for change detection
        elif step.uses == "input/file-before":
            if attachments[attachment_index].path:
                step.binds["RasterPathBefore"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPathBefore: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"Attachment {attachment_index} has no path!")

        # After file for change detection
        elif step.uses == "input/file-after":
            if attachment_index < len(attachments) and attachments[attachment_index].path:
                step.binds["RasterPathAfter"] = attachments[attachment_index].path
                logger.info(f"Injected RasterPathAfter: {attachments[attachment_index].path}")
                attachment_index += 1
            else:
                logger.warning(f"No attachment available for after file (index {attachment_index})")
