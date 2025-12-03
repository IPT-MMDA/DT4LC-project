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

    # First, check if this is a capability question (even if it contains action keywords)
    # e.g., "can you calculate ndvi?", "could you detect boundaries?"
    capability_response = _check_capability_question(req.prompt)
    if capability_response:
        logger.info("Quick classification: CONVERSATION (capability question)")
        return {
            "intent": IntentType.CONVERSATION,
            "reason": "Capability question - user asking if we can do something",
            "response": capability_response,
        }

    # If the request has attachments and looks like an action request, likely pipeline
    if req.attachments and _looks_like_action(req.prompt):
        logger.info("Quick classification: PIPELINE (has attachments + action keywords)")
        return {"intent": IntentType.PIPELINE, "reason": "Has attachments and action keywords"}

    # Check if it's an action request but missing required files
    # Return a helpful prompt instead of failing later
    if _is_action_without_file(req):
        logger.info("Quick classification: CONVERSATION (action request but no file)")
        return {
            "intent": IntentType.CONVERSATION,
            "reason": "Action request without required file",
            "response": _get_missing_file_response(req.prompt),
        }

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


def _check_capability_question(prompt: str) -> str | None:
    """Check if the user is asking about capabilities rather than requesting action.

    Detects patterns like:
    - "can you calculate ndvi?"
    - "could you detect boundaries?"
    - "are you able to run change detection?"
    - "is it possible to extract features?"

    Args:
        prompt: User prompt

    Returns:
        Helpful response if this is a capability question, None otherwise
    """
    prompt_lower = prompt.lower().strip()

    # Remove common prefixes that don't change meaning
    # e.g., "so, you can..." -> "you can..."
    # e.g., "ok, can you..." -> "can you..."
    prefix_pattern = r"^(so,?\s*|ok,?\s*|okay,?\s*|well,?\s*|hey,?\s*|hi,?\s*|hello,?\s*)"
    prompt_cleaned = re.sub(prefix_pattern, "", prompt_lower).strip()

    # Capability question patterns - checking if the system CAN do something
    capability_patterns = [
        r"^(can|could|would|will)\s+(you|it|the system)\s+",
        r"^(are you|is it)\s+(able|possible)\s+to\s+",
        r"^(is it|is there)\s+(possible|a way)\s+to\s+",
        r"^(do you|does it)\s+(support|have|offer)\s+",
        r"^what\s+(can|could)\s+(you|it|the system)\s+do",
        # "what else can you do?" / "what more can you do?"
        r"^what\s+(else|more)\s+(can|could)\s+(you|it)\s+do",
        # Confirmatory questions like "so you can calculate...?"
        r"^you\s+(can|could)\s+",
        # Questions with "any" indicating capability inquiry
        r"(can|could)\s+(you|it)\s+.+\s+(any|all)\s+",
    ]

    is_capability_question = any(re.search(p, prompt_cleaned) for p in capability_patterns)

    # Also check for question marks with capability keywords in original prompt
    if not is_capability_question and "?" in prompt_lower:
        # Questions like "you can calculate ndvi for any image?"
        confirmatory_patterns = [
            r"you\s+(can|could)\s+.+\?",
            r"(can|could)\s+(it|this|the system)\s+.+\?",
        ]
        is_capability_question = any(re.search(p, prompt_lower) for p in confirmatory_patterns)

    if not is_capability_question:
        return None

    # Determine what capability they're asking about and provide helpful response
    if any(kw in prompt_lower for kw in ["ndvi", "vegetation", "vegetation index"]):
        return (
            "Yes, I can calculate NDVI (Normalized Difference Vegetation Index) to analyze "
            "vegetation health from satellite imagery. Please upload a GeoTIFF image with "
            "NIR and Red bands, and I'll process it for you."
        )

    if any(kw in prompt_lower for kw in ["boundary", "boundaries", "field", "parcel", "delineate"]):
        return (
            "Yes, I can detect field boundaries using the Delineate-Anything model. "
            "This will identify agricultural parcels in your satellite image. "
            "Please upload a GeoTIFF image and I'll detect the field boundaries for you."
        )

    if any(kw in prompt_lower for kw in ["change", "compare", "difference", "before", "after"]):
        return (
            "Yes, I can perform change detection to compare two images and identify "
            "vegetation changes over time. Please upload two GeoTIFF images (before and after) "
            "and I'll analyze the differences."
        )

    if any(kw in prompt_lower for kw in ["feature", "prithvi", "embedding"]):
        return (
            "Yes, I can extract features using the Prithvi foundation model from NASA/IBM. "
            "This generates rich embeddings from satellite imagery. "
            "Please upload a GeoTIFF image to get started."
        )

    if any(kw in prompt_lower for kw in ["statistic", "stats", "analyze"]):
        return (
            "Yes, I can calculate statistics from your raster data including min, max, mean, "
            "and standard deviation. Please upload a GeoTIFF image and I'll analyze it."
        )

    # Generic capability response
    return (
        "Yes! Our geospatial Digital Twin system can perform several analyses:\n\n"
        "• **NDVI Calculation** - Analyze vegetation health\n"
        "• **Field Boundary Detection** - Identify agricultural parcels\n"
        "• **Change Detection** - Compare before/after images\n"
        "• **Statistics Extraction** - Calculate raster statistics\n"
        "• **Prithvi Features** - Extract foundation model embeddings\n\n"
        "Upload a GeoTIFF image and tell me what analysis you'd like to perform!"
    )


def _is_action_without_file(req: ChatRequest) -> bool:
    """Check if this is an action request that needs a file but doesn't have one.

    Args:
        req: Chat request

    Returns:
        True if this looks like an action request without required files
    """
    # If there are attachments, we have files
    if req.attachments:
        return False

    # If there's metadata with previous files, we might be okay
    if req.metadata and req.metadata.get("previous_files"):
        return False

    prompt_lower = req.prompt.lower()

    # Check if this is asking to perform an action (imperative form)
    action_starters = [
        r"^(please\s+)?(calculate|compute|run|do|perform|execute)\s+",
        r"^(please\s+)?(detect|find|identify|extract|get)\s+",
        r"^(please\s+)?(analyze|process|generate)\s+",
    ]

    is_action = any(re.search(p, prompt_lower) for p in action_starters)

    # Also check for action keywords without capability question framing
    if not is_action:
        # Direct object requests like "ndvi for this image"
        action_keywords = ["calculate ndvi", "run ndvi", "detect boundaries", "change detection"]
        is_action = any(kw in prompt_lower for kw in action_keywords)

    return is_action


def _get_missing_file_response(prompt: str) -> str:
    """Generate a helpful response when action is requested but file is missing.

    Args:
        prompt: User prompt

    Returns:
        Helpful response asking for file upload
    """
    prompt_lower = prompt.lower()

    if any(kw in prompt_lower for kw in ["ndvi", "vegetation"]):
        return (
            "I'd be happy to calculate NDVI for you! Please upload a GeoTIFF image "
            "with NIR and Red bands, and I'll analyze the vegetation health."
        )

    if any(kw in prompt_lower for kw in ["boundary", "boundaries", "field", "parcel"]):
        return (
            "I can detect field boundaries for you! Please upload a GeoTIFF satellite image "
            "and I'll identify the agricultural parcels."
        )

    if any(kw in prompt_lower for kw in ["change", "compare"]):
        return (
            "I can perform change detection! Please upload two GeoTIFF images "
            "(before and after) and I'll analyze the vegetation changes."
        )

    if any(kw in prompt_lower for kw in ["feature", "prithvi"]):
        return (
            "I can extract Prithvi features for you! Please upload a GeoTIFF image "
            "and I'll generate the foundation model embeddings."
        )

    if any(kw in prompt_lower for kw in ["statistic", "stats"]):
        return (
            "I can calculate statistics for you! Please upload a GeoTIFF image and I'll compute the raster statistics."
        )

    # Generic response
    return "I'd be happy to help with that analysis! Please upload a GeoTIFF image and I'll process it for you."
