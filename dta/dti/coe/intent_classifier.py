"""Intent Classifier — PIPELINE vs CONVERSATION routing.

The keyword lists, capability responses, and missing-file responses live on
each registry item's ``triggers`` and ``user_guide`` fields, loaded through
``coe/triggers.py``. Adding a new algorithm = a single registry.yaml edit;
no change to this file.

Routing logic:
1. Capability question framing → CONVERSATION with per-item capability response.
2. Attachments + action keyword → PIPELINE (fast path, no LLM call).
3. Imperative action without attachments → CONVERSATION asking for the file.
4. Clear action phrase (with or without attachments) → PIPELINE.
5. Otherwise fall through to the LLM (system prompt also rendered from the
   registry, not hardcoded).
"""

from __future__ import annotations

from enum import Enum
import json
import logging
import re
from typing import Any

from dta.dti.coe.llm import LLMMessage, get_llm_router
from dta.dti.coe.triggers import (
    GENERIC_ACTION_VERBS,
    get_trigger_index,
    is_capability_question,
)
from dta.dti.schemas import ChatRequest

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Types of user intent."""

    PIPELINE = "pipeline"  # Needs data processing
    CONVERSATION = "conversation"  # Needs helpful response


# Imperative-action prefixes recognised when the prompt has no attachments.
# Generic English; not algorithm-specific. Verb membership matches the original
# pre-Phase-2 list so test_intent_classifier.py keeps passing without changes.
_IMPERATIVE_PATTERNS = [
    re.compile(r"^(please\s+)?(calculate|compute|run|do|perform|execute)\s+", re.I),
    re.compile(r"^(please\s+)?(detect|find|identify|extract|get)\s+", re.I),
    re.compile(r"^(please\s+)?(analyze|process|generate)\s+", re.I),
]


def classify_intent(req: ChatRequest) -> dict[str, Any]:
    """Classify the intent of a user request.

    Returns a dict with at least:
        intent: IntentType (PIPELINE or CONVERSATION)
        reason: brief explanation
        response: helpful response (only for CONVERSATION intent)
    """
    idx = get_trigger_index()

    # 1. Capability question? Always CONVERSATION; respond from registry.
    if is_capability_question(req.prompt):
        logger.info("Quick classification: CONVERSATION (capability question)")
        return {
            "intent": IntentType.CONVERSATION,
            "reason": "Capability question - user asking if we can do something",
            "response": idx.render_capability_response(req.prompt),
        }

    # 2. Attachments + action keyword → PIPELINE (no LLM needed).
    if req.attachments and idx.has_keyword(req.prompt):
        logger.info("Quick classification: PIPELINE (has attachments + action keywords)")
        return {"intent": IntentType.PIPELINE, "reason": "Has attachments and action keywords"}

    # 3. Imperative action without a file → CONVERSATION asking for upload.
    if _is_action_without_file(req):
        logger.info("Quick classification: CONVERSATION (action request but no file)")
        return {
            "intent": IntentType.CONVERSATION,
            "reason": "Action request without required file",
            "response": idx.render_missing_file_response(req.prompt),
        }

    # 4. Clear action request (with prior context, perhaps) → PIPELINE.
    if idx.is_clear_action(req.prompt):
        logger.info("Quick classification: PIPELINE (clear action request)")
        return {"intent": IntentType.PIPELINE, "reason": "Clear action request (may use previous data)"}

    # 5. Fall through to LLM with a registry-rendered system prompt.
    router = get_llm_router()
    messages = [
        LLMMessage(role="system", content=idx.render_system_prompt()),
        LLMMessage(role="user", content=req.prompt),
    ]
    try:
        response = router.generate(messages, temperature=0.3)
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

    return {"intent": IntentType.PIPELINE, "reason": "Default fallback"}


# --- Internal helpers ---
# tests/test_intent_classifier.py imports these names directly, so they
# preserve their signatures and observable behavior. The keyword/regex lists
# they used to carry now live on coe/triggers.py + the registry.


def _looks_like_action(prompt: str) -> bool:
    """True if prompt contains a registered trigger keyword or generic action verb.

    Thin shim over the registry-driven TriggerIndex. The keyword set is the
    union of every user-runnable item's ``triggers.keywords`` plus the
    GENERIC_ACTION_VERBS constant.
    """
    return get_trigger_index().has_keyword(prompt)


def _is_clear_action_request(prompt: str) -> bool:
    """True if prompt is a clear, unambiguous action request.

    Thin shim over the registry-driven TriggerIndex. Matches against each
    item's ``triggers.action_phrases`` and ``triggers.keywords``, with the
    suffix patterns ("calculation", "analysis", "detection", "mapping",
    "classification") supported.
    """
    return get_trigger_index().is_clear_action(prompt)


def _is_action_without_file(req: ChatRequest) -> bool:
    """True if the prompt is an imperative action request and no file is attached."""
    if req.attachments:
        return False
    if req.metadata and req.metadata.get("previous_files"):
        return False
    prompt_lower = req.prompt.lower()
    if any(p.search(prompt_lower) for p in _IMPERATIVE_PATTERNS):
        return True
    # Fall back to the registry-driven action-phrase check (replaces the old
    # hardcoded ["calculate ndvi", "run ndvi", ...] list).
    return get_trigger_index().is_clear_action(prompt_lower)


# Re-export GENERIC_ACTION_VERBS so any external caller using the old import
# path doesn't break. (No internal users today; defensive.)
__all__ = [
    "GENERIC_ACTION_VERBS",
    "IntentType",
    "classify_intent",
]
