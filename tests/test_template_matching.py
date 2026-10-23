"""Template matching by whole words and trigger phrases (v0.6.0 story "Plan proposal and confirmation").

Substring matching routed unrelated prompts to Delineate-Anything: "land" matched
"landscape", "detect" matched "detection". Delineate must run only for field
boundaries, parcels or delineation, and the template planner never picks band
math (it cannot write a formula).

No test here can reach a live provider: the router points at a closed Ollama port.
"""

from collections.abc import Iterator
from pathlib import Path
import re

import pytest

from dta.config import ROOT_DIR
from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.coe.planner import plan_template
from dta.dti.coe.triggers import get_trigger_index, phrase_in
from dta.dti.registry import load_registry
from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
DELINEATE = "models/delineate-anything"
NDMI_PROMPT = "Map moisture change with NDMI between the two dates"
UNRELATED = [
    "Detect water changes between these images",
    "Classify land cover",
    "Detect changes",
    "Map burn severity",
    NDMI_PROMPT,
    "Show me the landscape",
]


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    reset_llm_router()
    assert all(isinstance(p, OllamaProvider) for p in get_llm_router().providers)
    yield
    reset_llm_router()


def _attachments(*paths: str) -> list[Attachment]:
    return [Attachment(id=f"a{i}", filename=Path(p).name, mime_type="image/tiff", path=p) for i, p in enumerate(paths)]


def _llm_like_context(prompt: str) -> ContextUnderstanding:
    """What a context LLM typically returns: every word of the prompt as a keyword, no output type."""
    words = re.findall(r"[a-z]+", prompt.lower())
    return ContextUnderstanding(
        goal=prompt, desired_outputs=[], required_inputs=["RasterPath"], hints={"keywords": words}
    )


@pytest.mark.parametrize(
    ("phrase", "text", "expected"),
    [
        ("detect", "Detect changes", True),
        ("detect", "flood detection", False),
        ("land", "Show me the landscape", False),
        ("land", "a landsat scene", False),
        ("land cover", "Classify land cover", True),
        ("water change", "Detect water changes", True),
        ("ice", "notice the river", False),
        ("after", "the afternoon pass", False),
        ("field boundaries", "Detect field boundaries in this image", True),
        ("burn", "Map burn severity", True),
        # -ed and -ing endings (F-1), a final e dropped before them; nothing longer.
        ("flood", "Find the flooded areas", True),
        ("flood", "flooding after the breach", True),
        ("flood", "the floodplain", False),
        ("detect", "detected parcels", True),
        ("detect", "detecting parcels", True),
        ("detect", "a detector", False),
        ("change", "what changed?", True),
        ("change", "changing water", True),
        ("water change", "where has the water changed", True),
        ("compare", "compared with June", True),
        ("compare", "comparing the dates", True),
        ("compare", "a comparison", False),
        ("land", "landed", True),
        ("land", "the landsat archive", False),
        ("ice", "icing", True),
        ("ice", "a notice", False),
        ("snow", "a snowy ridge", False),
    ],
)
def test_phrase_in_matches_whole_words(phrase: str, text: str, expected: bool) -> None:
    assert phrase_in(phrase, text) is expected


@pytest.mark.parametrize(
    ("prompt", "item_id"),
    [
        # "instead of" turns the next term down; else the index named first wins.
        ("Use EVI instead of NDVI for the forest", "algorithms/evi"),
        ("Use NDVI rather than EVI", "algorithms/ndvi"),
        ("Calculate EVI or NDVI", "algorithms/evi"),
        ("Calculate NDVI or EVI", "algorithms/ndvi"),
        # A match inside another item's longer match does not count: "snow" in "classify snow".
        ("Classify snow and ice robustly", "algorithms/snow-classifier"),
        ("Map snow cover with NDSI", "algorithms/ndsi"),
        ("Compare with NDVI change between the same dates", "algorithms/change-detection"),
        # Prithvi only for its own phrases (D3, D4).
        ("Fill the cloud gaps in the 9 June image", "models/prithvi-reconstruction"),
        ("Reconstruct the masked pixels", "models/prithvi-reconstruction"),
        ("Extract the main features of this scene", None),
        ("Compute embeddings for this image", None),
        # Delineate only for boundaries, parcels or delineation, not for any "fields".
        ("Detect field boundaries east of the river", "models/delineate-anything"),
        ("Classify land cover on the flooded fields", "algorithms/lulc-classifier"),
        ("In these fields, map vegetation health", "algorithms/ndvi"),
    ],
)
def test_find_matching_item(prompt: str, item_id: str | None) -> None:
    item = get_trigger_index().find_matching_item(prompt)

    assert (item.id if item else None) == item_id


@pytest.mark.parametrize("prompt", ["Extract the main features of this scene", "Compute embeddings for this image"])
@pytest.mark.parametrize("files", [[SAMPLE_0601], [SAMPLE_0601, SAMPLE_0609]], ids=["one-file", "two-files"])
def test_generic_words_never_plan_prithvi(prompt: str, files: list[str]) -> None:
    """D3: "features" planned Prithvi, so a member's unrelated prompt would start a 450 MB download."""
    out = orchestrate(ChatRequest(prompt=prompt, attachments=_attachments(*files)))

    planned = out.get("plan") or out.get("candidate") or {"steps": []}
    assert "models/prithvi-reconstruction" not in [s["uses"] for s in planned["steps"]], out


@pytest.mark.parametrize("prompt", UNRELATED)
@pytest.mark.parametrize("files", [[SAMPLE_0601], [SAMPLE_0601, SAMPLE_0609]], ids=["one-file", "two-files"])
def test_unrelated_prompts_never_plan_delineate(prompt: str, files: list[str]) -> None:
    out = orchestrate(ChatRequest(prompt=prompt, attachments=_attachments(*files)))

    planned = out.get("plan") or out.get("candidate") or {"steps": []}
    assert DELINEATE not in [s["uses"] for s in planned["steps"]], out


@pytest.mark.parametrize("prompt", UNRELATED)
def test_template_planner_never_picks_delineate_from_llm_keywords(prompt: str) -> None:
    steps = [s.uses for s in plan_template(_llm_like_context(prompt), load_registry()).steps]

    assert DELINEATE not in steps


def test_field_boundaries_plan_delineate() -> None:
    """Positive control: the request names field boundaries."""
    out = orchestrate(
        ChatRequest(prompt="Detect field boundaries in this image", attachments=_attachments(SAMPLE_0601))
    )

    assert out["ok"], out.get("error")
    assert [s["uses"] for s in out["plan"]["steps"]] == ["input/file", DELINEATE, "post-processing/agent-analysis"]
    assert out["plan"]["composed"] is False
    assert "proposal" not in out
    template = plan_template(_llm_like_context("Detect field boundaries in this image"), load_registry())
    assert DELINEATE in [s.uses for s in template.steps]


@pytest.mark.parametrize("prompt", ["band math", "compute a custom index"])
def test_template_planner_never_picks_band_math(prompt: str) -> None:
    for ctx in (
        _llm_like_context(prompt),
        ContextUnderstanding(goal=prompt, desired_outputs=["BandMathMap"], required_inputs=["RasterPath"]),
    ):
        steps = [s.uses for s in plan_template(ctx, load_registry()).steps]
        assert not any(s.startswith("algorithms/band-math") for s in steps), steps


def test_planned_hosted_models_are_not_picked_by_keywords() -> None:
    """Without band math, "custom index" used to reach a planned Earth Engine model by its "index" keyword."""
    steps = [s.uses for s in plan_template(_llm_like_context("custom index"), load_registry()).steps]

    assert "models/vi-forecasting-dti" not in steps


@pytest.mark.parametrize(
    ("goal", "keywords"),
    [
        ("Detect land changes", ["detect", "land", "changes"]),
        ("Extract the land surface", ["extract", "land"]),
    ],
)
def test_generic_keywords_alone_never_select_delineate(goal: str, keywords: list[str]) -> None:
    """Delineate's broad registry keywords ("detect", "land", "extract") selected it on their own."""
    ctx = ContextUnderstanding(
        goal=goal, desired_outputs=[], required_inputs=["RasterPath"], hints={"keywords": keywords}
    )

    assert DELINEATE not in [s.uses for s in plan_template(ctx, load_registry()).steps]
