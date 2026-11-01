"""Conversation path: deterministic conversational rules and replies (v0.6.0).

Conversation never creates a job. The intent classifier runs :func:`conversational_rule` before
its keyword rules and the LLM, so greetings, thanks, a bare "here" and questions about methods
stay conversation even when no provider answers (the hybrid-deterministic fallback would
otherwise read them as pipeline requests). ``POST /v1/chat`` turns them into markdown replies:
the fixed texts below, or :func:`generate_reply` (the LLM, with the recent history), or
:func:`offline_reply` (the catalog) when no provider answers.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from dta.dti.coe.llm import LLMMessage, get_llm_router
from dta.dti.coe.named_indices import match_named_index
from dta.dti.coe.triggers import GENERIC_ACTION_VERBS, get_trigger_index, looks_like_explanation_question, phrase_in
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import ChatRequest

logger = logging.getLogger(__name__)

Rule = Literal["greeting", "thanks", "deictic", "question"]

_GREETING = re.compile(
    r"^(hi|hello|hey|hiya|howdy|greetings|good (morning|afternoon|evening|day))"
    r"( there| all| everyone| dt4lc| assistant)?"
    r"( how are you( doing)?( today)?)?$"
)
_THANKS = re.compile(
    r"^((ok|okay|great|perfect|nice|awesome|cool) )?(thanks|thank you|thx|ty|cheers)"
    r"( a lot| very much| so much)?( (that's|that is|that was) (great|helpful|perfect|useful))?$"
)
_DEICTIC = re.compile(
    r"^(here|there|here you go|here you are|here it is|here they are|there you go|there it is|"
    r"this|these|this one|these ones|this file|these files|this image|these images|attached|done|ok|okay)$"
)
# Verbs that make a question an analysis request ("how do I compute NDVI?" is not a method question).
ANALYSIS_VERBS: frozenset[str] = GENERIC_ACTION_VERBS | {"analyse", "delineate"}
# Imperative verbs that make an explanatory question an analysis request even with a file attached.
_EXPLANATORY_VETO: frozenset[str] = ANALYSIS_VERBS | {"map", "show", "compare"}
# An explanatory question asks about a method or a result, not about the data: conversation even with
# attachments or a region. "what is" / "what are" count only before a method or index name.
_EXPLANATORY = re.compile(
    r"^(why|explain|describe|tell me about|what would you do differently|how does|how do|is it\b.*\bbetter)\b"
    r"|^what does\b.*\bmean\b"
)
_WHAT_IS = re.compile(r"^what (is|are)\b")


def _normalize(text: str) -> str:
    """Lowercase, punctuation and repeated spaces dropped: ``"Hello, there!"`` → ``"hello there"``."""
    return " ".join(re.sub(r"[^\w\s'-]", " ", text.lower()).split())


def is_deictic(text: str) -> bool:
    """True for an empty message or one that only points at an attachment: "here", "this one", "here you go"."""
    return not text.strip() or _DEICTIC.match(_normalize(text)) is not None


def is_resume_message(text: str) -> bool:
    """True if ``text`` can only mean "use this for what you asked": deictic, or at most four words
    that are not a greeting, thanks or question and have no analysis verb, registry keyword or
    named index (so they are not a new request: "use this", "for that one")."""
    if is_deictic(text):
        return True
    if _GREETING.match(_normalize(text)) or _THANKS.match(_normalize(text)):
        return False
    words = _normalize(text).split()
    if len(words) > 4 or text.strip().endswith("?") or looks_like_explanation_question(text):
        return False
    keywords = get_trigger_index().all_keywords | ANALYSIS_VERBS
    return not any(phrase_in(k, text) for k in keywords) and match_named_index(text) is None


def _method_names() -> set[str]:
    """Names of the methods and indices a "what is ..." question can ask about: each analysis's id
    ("ndvi", "change-detection", "change detection"), a model's first word ("prithvi") and the
    named indices (NDMI, NBR, SAVI, NDBI)."""
    from dta.dti.coe.named_indices import NAMED_INDICES

    names = {i.name.lower() for i in NAMED_INDICES}
    for item in get_trigger_index().user_runnable_items:
        tail = item.id.rsplit("/", 1)[-1]
        names |= {tail, tail.replace("-", " ")}
        if item.kind == "model":
            names.add(tail.split("-")[0])
    return names


def is_explanatory_question(prompt: str) -> bool:
    """True for a question about a method or a result, which is conversation even with files attached:
    it starts with why, explain, describe, tell me about, what would you do differently, how does,
    how do, "is it ... better", "what does ... mean", or "what is/are" + a method or index name; and
    it has no imperative analysis verb (:data:`_EXPLANATORY_VETO`, whole words). Locating and
    quantifying questions ("where is", "how much", "how many", "what changed", "which areas") are not
    explanatory: with a file they run an analysis."""
    text = _normalize(prompt)
    if any(phrase_in(v, text) for v in _EXPLANATORY_VETO):
        return False
    if _EXPLANATORY.match(text):
        return True
    return _WHAT_IS.match(text) is not None and any(phrase_in(n, text) for n in _method_names())


def conversational_rule(prompt: str, has_input: bool) -> Rule | None:
    """The deterministic conversational rule ``prompt`` matches, or ``None``.

    - ``greeting``: the whole message is a greeting ("hello", "hi there", "good morning, how are you?");
    - ``thanks``: the whole message is thanks ("thanks", "ok thanks", "thank you very much");
    - ``deictic``: empty, or only points at a file ("here", "this one", "here you go");
    - ``question``: an explanatory question (:func:`is_explanatory_question`), with or without a file
      or region; or, with no file or region (``has_input``), a question form (starts with what, how,
      why, when, where, which, explain, describe or "tell me about", or ends with ``?``) and no
      analysis verb (:data:`ANALYSIS_VERBS`, whole words).

    Capability questions ("what can you do?") are matched before this, by
    :func:`~dta.dti.coe.triggers.is_capability_question`.
    """
    text = _normalize(prompt)
    if _GREETING.match(text):
        return "greeting"
    if _THANKS.match(text):
        return "thanks"
    if is_deictic(prompt):
        return "deictic"
    if is_explanatory_question(prompt):
        return "question"
    if has_input:
        return None
    question = looks_like_explanation_question(prompt) or prompt.strip().endswith("?")
    if question and not any(phrase_in(v, prompt) for v in ANALYSIS_VERBS):
        return "question"
    return None


GREETING_REPLY = (
    "Hello! I'm the DT4LC assistant. I run land-cover analyses on satellite imagery: spectral "
    "indices, change detection between two dates, classification and field boundaries.\n\n"
    "- Attach a GeoTIFF (or send a region from the map) and describe the analysis, "
    "for example *Calculate NDVI*.\n"
    "- Ask **What can you do?** for the full catalog."
)
THANKS_REPLY = "You're welcome! Ask a follow-up, or attach another image when you're ready."
DEICTIC_REPLY = (
    "What would you like me to do? Attach a GeoTIFF and describe the analysis, for example "
    "*Calculate NDVI* or *Map water change between the two dates*."
)
DEICTIC_WITH_FILES_REPLY = (
    "Got it. What should I compute from {files}? For example *Calculate NDVI*, or "
    "*Map water change between the two dates* for a before and an after image."
)
OFFLINE_PREFIX = (
    "The assistant is offline on this server, so I can only describe the catalog. "
    "Analyses still run: attach a GeoTIFF and name one.\n\n"
)
SYSTEM_PROMPT = (
    "You are the assistant of DT4LC, a digital twin for land-cover analysis of satellite imagery. "
    "Answer the user's last message directly, without a greeting, in concise markdown: short "
    "paragraphs, a list where it helps, at most about 200 words. Use the conversation so far, "
    "including any result summary in it. "
    "You cannot run an analysis from this reply: if the user wants one, tell them to ask for it "
    "with a GeoTIFF attached or a region sent from the map. The analyses this server runs:\n\n"
)


def rule_reply(rule: Rule, attachments: int = 0) -> str | None:
    """The fixed markdown reply for a greeting, thanks or a deictic message; ``None`` for a question."""
    if rule == "greeting":
        return GREETING_REPLY
    if rule == "thanks":
        return THANKS_REPLY
    if rule == "deictic":
        if attachments:
            return DEICTIC_WITH_FILES_REPLY.format(files="this file" if attachments == 1 else "these files")
        return DEICTIC_REPLY
    return None


def catalog_reply() -> str:
    """The capability reply ("what can you do?"): the registry catalog with whole descriptions."""
    return get_trigger_index().render_catalog_response()


def offline_reply() -> str:
    """The reply to any other message when no provider answers: limited to the catalog."""
    return OFFLINE_PREFIX + catalog_reply()


def generate_reply(message: str, history: list[tuple[str, str]]) -> str:
    """The LLM's markdown answer to ``message`` after ``history`` (``(role, text)``, oldest first).

    The history goes into the system message as a transcript, not as separate turns: not every
    provider sends assistant turns (Gemini drops them).
    Raises whatever the router raises when no provider answers; the caller falls back to
    :func:`offline_reply`.
    """
    system = SYSTEM_PROMPT + get_trigger_index().render_catalog_response()
    if history:
        transcript = "\n\n".join(f"{'Assistant' if r == 'assistant' else 'User'}: {t}" for r, t in history)
        system += f"\n\nThe conversation so far:\n\n{transcript}"
    messages = [LLMMessage(role="system", content=system), LLMMessage(role="user", content=message)]
    response = get_llm_router().generate(messages, temperature=0.3)
    text = response.text.strip()
    if not text:
        raise ValueError("empty reply")
    return text


# A named index (NDMI, NBR, ...) asked for with one of these words compares two dates.
_CHANGE_WORDS = ("change", "between", "compare", "difference", "before", "two dates")


def input_needs(prompt: str) -> tuple[str | None, list[str]]:
    """``(item_id, needs)`` for an analysis request that came without a file: the analysis it names
    (``None`` if none) and the inputs it waits for, ``["RasterPath"]`` or
    ``["RasterPathBefore", "RasterPathAfter"]``: a named index first, as the orchestrator plans it,
    else from the template plan of the registry triggers."""
    from dta.dti.coe.context_agent import _deterministic_context
    from dta.dti.coe.planner import plan_template

    if match_named_index(prompt) is not None:
        if any(phrase_in(w, prompt) for w in _CHANGE_WORDS):
            return "algorithms/band-math-change", ["RasterPathBefore", "RasterPathAfter"]
        return "algorithms/band-math", ["RasterPath"]
    reg = load_registry()
    template = plan_template(_deterministic_context(ChatRequest(prompt=prompt)), reg)
    uses = [s.uses for s in template.steps]
    analysis = [u for u in uses if get_item(reg, u).kind in ("algorithm", "model")]
    if "input/file-before" in uses:
        return (analysis[0] if analysis else None), ["RasterPathBefore", "RasterPathAfter"]
    if "input/file" in uses:
        return (analysis[0] if analysis else None), ["RasterPath"]
    item = get_trigger_index().find_matching_item(prompt)
    return (item.id if item else None), ["RasterPath"]
