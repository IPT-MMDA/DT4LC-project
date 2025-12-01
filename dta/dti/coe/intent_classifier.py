"""Intent Classifier - Determines if a request needs pipeline execution or conversation.

This module classifies user requests into:
- PIPELINE: Requests that need data processing (NDVI, change detection, etc.)
- CONVERSATION: Requests that need helpful responses (questions, guidance, explanations)
"""

from enum import Enum
import json
import logging
import re
from typing import Any

from dta.dti.coe.llm import LLMMessage, LLMRouter
from dta.dti.coe.llm.config import create_router_from_env
from dta.dti.schemas import ChatRequest

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Types of user intent."""

    PIPELINE = "pipeline"  # Needs data processing
    CONVERSATION = "conversation"  # Needs helpful response


# Lazy router initialization
_router: LLMRouter | None = None


def _get_router() -> LLMRouter:
    """Lazy initialization of LLM router."""
    global _router
    if _router is None:
        _router = create_router_from_env()
    return _router


SYSTEM_PROMPT = """You are an Intent Classifier for a geospatial Digital Twin system.

Your task is to classify user messages into one of two categories:

1. PIPELINE - The user wants to process/analyze geospatial data:
   - Calculate NDVI, vegetation index
   - Detect field boundaries, agricultural parcels
   - Run change detection, compare images
   - Extract features, run models
   - Generate statistics, analyze data
   - Any request that needs actual data processing

2. CONVERSATION - The user is asking questions or seeking guidance:
   - "What can we do next?"
   - "What analyses are available?"
   - "Help me understand this result"
   - "Explain what NDVI means"
   - "What does this data show?"
   - Questions about capabilities
   - Requests for clarification
   - General conversation

Return ONLY valid JSON with this structure:
{
  "intent": "pipeline" or "conversation",
  "reason": "brief explanation",
  "response": "If conversation, provide a helpful response here. If pipeline, leave empty."
}

For CONVERSATION responses, be helpful and suggest available analyses:
- Field boundary detection (Delineate-Anything model)
- NDVI calculation (vegetation health)
- Change detection (compare two images)
- Statistics extraction
- Prithvi feature extraction (foundation model embeddings)
"""


def classify_intent(req: ChatRequest) -> dict[str, Any]:
    """Classify the intent of a user request.

    Args:
        req: Chat request with prompt and optional attachments

    Returns:
        Dictionary with:
            - intent: IntentType (PIPELINE or CONVERSATION)
            - reason: Brief explanation of classification
            - response: Helpful response if conversation
    """
    router = _get_router()

    # Build context about what's available
    context = req.prompt

    # If the request has attachments and looks like an action request, likely pipeline
    if req.attachments and _looks_like_action(req.prompt):
        logger.info("Quick classification: PIPELINE (has attachments + action keywords)")
        return {"intent": IntentType.PIPELINE, "reason": "Has attachments and action keywords"}

    # Even without attachments, if it's a clear action request, classify as pipeline
    # This handles follow-up requests like "ndvi calculation" after previous data upload
    if _is_clear_action_request(req.prompt):
        logger.info("Quick classification: PIPELINE (clear action request)")
        return {"intent": IntentType.PIPELINE, "reason": "Clear action request (may use previous data)"}

    # Use LLM for nuanced classification
    messages = [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=context),
    ]

    try:
        response = router.generate(messages, temperature=0.3)

        # Parse JSON response
        m = re.search(r"\{.*\}", response.text, re.S)
        if m:
            data = json.loads(m.group(0))
            intent_str = data.get("intent", "pipeline").lower()
            intent = IntentType.CONVERSATION if intent_str == "conversation" else IntentType.PIPELINE

            return {
                "intent": intent,
                "reason": data.get("reason", ""),
                "response": data.get("response", ""),
            }
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to PIPELINE")

    # Default to pipeline if classification fails
    return {"intent": IntentType.PIPELINE, "reason": "Default fallback"}


def _looks_like_action(prompt: str) -> bool:
    """Quick check if prompt looks like an action request.

    Args:
        prompt: User prompt

    Returns:
        True if prompt contains action keywords
    """
    action_keywords = [
        "calculate",
        "compute",
        "detect",
        "extract",
        "analyze",
        "run",
        "process",
        "generate",
        "create",
        "find",
        "identify",
        "measure",
        "ndvi",
        "vegetation",
        "boundary",
        "boundaries",
        "change",
        "statistics",
        "features",
    ]

    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in action_keywords)


def _is_clear_action_request(prompt: str) -> bool:
    """Check if prompt is a clear, unambiguous action request.

    This catches short, direct requests like:
    - "ndvi calculation"
    - "calculate ndvi"
    - "detect boundaries"
    - "run change detection"

    But NOT questions like:
    - "what is ndvi?"
    - "explain ndvi calculation"
    - "how does change detection work?"

    Args:
        prompt: User prompt

    Returns:
        True if this is a clear action request
    """
    prompt_lower = prompt.lower().strip()

    # Question words indicate conversation, not action
    question_words = ["what", "how", "why", "when", "where", "which", "explain", "describe", "tell me about"]
    if any(prompt_lower.startswith(qw) for qw in question_words):
        return False

    # Check for specific action patterns
    action_patterns = [
        r"^(calculate|compute|run|do|perform)\s+(ndvi|statistics|change\s*detection|field\s*detection)",
        r"^ndvi(\s+calculation|\s+analysis)?$",
        r"^(detect|find|identify)\s+(field\s*)?(boundaries|parcels)",
        r"^change\s*detection$",
        r"^(extract|get)\s+(features|statistics)",
        r"^field\s*(boundary|boundaries)\s*(detection)?$",
    ]

    for pattern in action_patterns:
        if re.search(pattern, prompt_lower):
            return True

    return False
