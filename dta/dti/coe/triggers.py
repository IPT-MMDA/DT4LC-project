"""Registry-driven trigger index for intent classification.

Derives the per-algorithm keyword/action-phrase sets and capability response
strings from each registry item's ``triggers`` and ``user_guide`` fields, so
the intent classifier doesn't have to hardcode them. Adding a new algorithm
= a single edit to ``registry.yaml``; no code change here.

Generic English mechanics (action verbs, question framing) stay in code
because they're properties of imperative-prompt grammar, not knowledge about
which algorithms the system runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
import re

from dta.dti.registry import load_registry
from dta.dti.schemas import Registry, RegistryItem

# Generic English action verbs — not tied to any specific algorithm. Used to
# detect "the user is asking the system to DO something" without a specific
# algorithm name. Per-algorithm verb usage (e.g. "calculate ndvi") comes
# from each item's `triggers.action_phrases` in the registry.
# Mirrors the original `_looks_like_action` verb list. Any short word risks
# false-positive substring matches (e.g. "do" matches inside "random"), so
# we keep this list conservative and ≥4 characters.
GENERIC_ACTION_VERBS: frozenset[str] = frozenset(
    {
        "calculate",
        "compute",
        "detect",
        "extract",
        "analyze",
        "run",
        "process",
        "generate",
        "find",
        "identify",
        "measure",
        "perform",
        "execute",
        "classify",
    }
)

# Question-framing patterns that indicate the user is asking about capabilities,
# not requesting an action. Generic English; not algorithm-specific.
_QUESTION_PREFIXES = ("what", "how", "why", "when", "where", "which", "explain", "describe", "tell me about")
_CAPABILITY_QUESTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"^\s*(can|could|would|will)\s+(you|it|the system)\s+",
        r"^\s*(are you|is it)\s+(able|possible)\s+to\s+",
        r"^\s*(is it|is there)\s+(possible|a way)\s+to\s+",
        r"^\s*(do you|does it)\s+(support|have|offer)\s+",
        r"^\s*what\s+(can|could)\s+(you|it|the system)\s+do",
        r"^\s*what\s+(else|more)\s+(can|could)\s+(you|it)\s+do",
        r"^\s*you\s+(can|could)\s+",
    )
]
# A few common social-pleasantry prefixes to strip before question detection
# (e.g. "so, can you...?" → "can you...?"). Mirrors the original behavior.
_PLEASANTRY_PREFIX = re.compile(r"^(so|ok|okay|well|hey|hi|hello),?\s*", re.I)


def _strip_pleasantries(prompt: str) -> str:
    return _PLEASANTRY_PREFIX.sub("", prompt.strip())


def is_capability_question(prompt: str) -> bool:
    """True if the prompt is asking what the system can do (vs. asking it to act)."""
    cleaned = _strip_pleasantries(prompt)
    if any(pat.search(cleaned) for pat in _CAPABILITY_QUESTION_PATTERNS):
        return True
    # Confirmatory questions like "you can calculate ndvi for any image?"
    if "?" in prompt and re.search(r"\byou\s+(can|could)\s+", prompt, re.I):
        return True
    return False


def looks_like_explanation_question(prompt: str) -> bool:
    """True if the prompt starts with a question word (what/how/why/...)."""
    return any(prompt.lower().strip().startswith(qw) for qw in _QUESTION_PREFIXES)


@dataclass(frozen=True)
class TriggerIndex:
    """Compiled lookup index over registry triggers + user_guide fields.

    Build once at startup via `get_trigger_index()`. The lookups (`has_keyword`,
    `is_clear_action`, `find_matching_item`) are O(items × keywords) substring
    scans — fine for a registry of dozens of items, no need for a fancier
    data structure.
    """

    registry: Registry
    # Items eligible for trigger matching — algorithms + locally-managed
    # models. Excludes inputs, preprocessors, hosted-only models with no
    # triggers, and the agent-analysis postprocess.
    user_runnable_items: tuple[RegistryItem, ...]
    # Flat union of all keywords (lowercased) across user-runnable items,
    # plus the generic action verbs. Used by `has_keyword`.
    all_keywords: frozenset[str] = field(default_factory=frozenset)
    # All action phrases (lowercased) across user-runnable items.
    all_action_phrases: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_registry(cls, registry: Registry) -> TriggerIndex:
        runnable = tuple(_user_runnable_items(registry))
        all_kw: set[str] = set(GENERIC_ACTION_VERBS)
        all_ap: set[str] = set()
        for item in runnable:
            if item.triggers is None:
                continue
            all_kw.update(k.lower() for k in item.triggers.keywords)
            all_ap.update(p.lower() for p in item.triggers.action_phrases)
        return cls(
            registry=registry,
            user_runnable_items=runnable,
            all_keywords=frozenset(all_kw),
            all_action_phrases=frozenset(all_ap),
        )

    # --- Quick boolean checks driven from registry data --------------------

    def has_keyword(self, prompt: str) -> bool:
        """True if any registered trigger keyword (or generic verb) appears in prompt."""
        p = prompt.lower()
        return any(kw in p for kw in self.all_keywords)

    def is_clear_action(self, prompt: str) -> bool:
        """True if the prompt is a clear, unambiguous action request.

        Three ways a prompt qualifies:
        1. It starts with a generic verb followed by a registered keyword
           ("calculate ndvi", "run change detection").
        2. It IS a registered action phrase ("ndvi calculation", "delineate fields").
        3. It IS a registered keyword on its own ("ndvi", "field boundaries").

        Question-framed prompts ("what is ndvi?", "explain ndvi") are excluded.
        """
        p = prompt.lower().strip()
        if not p:
            return False
        if looks_like_explanation_question(p):
            return False

        # Direct match: prompt IS a registered action phrase.
        if p in self.all_action_phrases:
            return True

        # Direct match: prompt IS a registered keyword (e.g. "ndvi").
        if p in self.all_keywords - GENERIC_ACTION_VERBS:
            return True

        # Verb + keyword pattern: "<generic-verb> <registered-keyword/phrase>".
        first_word, _, rest = p.partition(" ")
        if first_word in GENERIC_ACTION_VERBS and rest:
            if rest in self.all_action_phrases or rest in self.all_keywords:
                return True
            # Allow looser match: any registered keyword appears in the rest.
            if any(kw in rest for kw in self.all_keywords - GENERIC_ACTION_VERBS):
                return True

        # "<registered-keyword> calculation/analysis/detection/mapping/classification"
        action_suffixes = (" calculation", " analysis", " detection", " mapping", " classification")
        for kw in self.all_keywords - GENERIC_ACTION_VERBS:
            if p == kw or p.startswith(f"{kw} "):
                if any(p.endswith(suffix) for suffix in action_suffixes):
                    return True
                if p == kw:
                    return True

        return False

    # --- Item lookup -------------------------------------------------------

    def find_matching_item(self, prompt: str) -> RegistryItem | None:
        """Return the user-runnable item whose triggers best match the prompt,
        or None if no item matches. Score = number of distinct trigger keywords
        appearing in the prompt; tie-broken by length of the longest match.
        """
        p = prompt.lower()
        best: RegistryItem | None = None
        best_score = 0
        best_max_len = 0
        for item in self.user_runnable_items:
            if item.triggers is None:
                continue
            keywords = [k.lower() for k in item.triggers.keywords]
            phrases = [k.lower() for k in item.triggers.action_phrases]
            matches = [kw for kw in keywords + phrases if kw in p]
            if not matches:
                continue
            score = len(matches)
            max_len = max((len(m) for m in matches), default=0)
            if score > best_score or (score == best_score and max_len > best_max_len):
                best = item
                best_score = score
                best_max_len = max_len
        return best

    # --- Response rendering -----------------------------------------------

    def render_capability_response(self, prompt: str) -> str:
        """Capability question → per-item response, or registry-wide catalog if no specific match."""
        item = self.find_matching_item(prompt)
        if item and item.user_guide and item.user_guide.capability_response:
            return item.user_guide.capability_response.strip()
        return self.render_catalog_response()

    def render_missing_file_response(self, prompt: str) -> str:
        """Action-without-attachment → per-item ask, or generic ask if no specific match."""
        item = self.find_matching_item(prompt)
        if item and item.user_guide and item.user_guide.missing_file_response:
            return item.user_guide.missing_file_response.strip()
        return "I'd be happy to help with that analysis! Please upload a GeoTIFF image and I'll process it for you."

    def render_catalog_response(self) -> str:
        """Bulleted list of every user-runnable item — used as the generic 'what can you do?' answer."""
        bullets: list[str] = []
        for item in self.user_runnable_items:
            label = item.display_name or item.id
            desc = (item.description or "").strip().split("\n", 1)[0]
            bullets.append(f"• **{label}** — {desc}" if desc else f"• **{label}**")
        return (
            "Yes! Our geospatial Digital Twin system can perform several analyses:\n\n"
            + "\n".join(bullets)
            + "\n\nUpload a GeoTIFF image and tell me what analysis you'd like to perform."
        )

    def render_system_prompt(self) -> str:
        """LLM system prompt enumerating supported analyses from the registry.

        Replaces the static SYSTEM_PROMPT in intent_classifier.py — adding a
        new algorithm to YAML now updates this text automatically.
        """
        bullets: list[str] = []
        for item in self.user_runnable_items:
            label = item.display_name or item.id
            desc = (item.description or "").strip().split("\n", 1)[0]
            bullets.append(f"   - {label}: {desc}" if desc else f"   - {label}")
        catalog = "\n".join(bullets) if bullets else "   - (no user-runnable items registered)"
        return (
            "You are an Intent Classifier for a geospatial Digital Twin system.\n\n"
            "Your task is to classify user messages into one of two categories:\n\n"
            "1. PIPELINE - The user wants to process or analyze geospatial data.\n"
            "   Examples: calculate an index, detect features, run a change "
            "analysis, generate statistics, run a model.\n\n"
            "2. CONVERSATION - The user is asking questions or seeking guidance.\n"
            "   Examples: 'what can we do next?', 'explain NDVI', 'help me "
            "understand this result', requests for clarification.\n\n"
            'Return ONLY valid JSON: {"intent": "pipeline" or "conversation", '
            '"reason": "...", "response": "if conversation, a helpful answer"}\n\n'
            "For CONVERSATION responses, you may suggest available analyses:\n"
            f"{catalog}\n"
        )


def _user_runnable_items(registry: Registry) -> Iterable[RegistryItem]:
    """Items the user can drive directly: algorithms + locally-managed models.

    Excludes inputs (passthrough), preprocessors (internal), the agent-analysis
    postprocess (always appended automatically), and hosted-only models that
    don't have local model_ids (huggingface/GEE integrations in :planned status).
    """
    for item in registry.instances:
        if item.kind in ("input", "preprocessor", "postprocess"):
            continue
        # Hosted-only models without a local model_id and no triggers are
        # `integration: planned` in the YAML — not user-runnable yet.
        if item.kind == "model" and item.model_id is None and item.triggers is None:
            continue
        yield item


@lru_cache(maxsize=1)
def get_trigger_index() -> TriggerIndex:
    """Cached singleton TriggerIndex over the default registry."""
    return TriggerIndex.from_registry(load_registry())


def reset_trigger_index() -> None:
    """Clear the cached index — used in tests after monkeypatching the registry."""
    get_trigger_index.cache_clear()
