"""Every prompt the Workspace sends, routed with every LLM provider down (v0.6.0 routing fixes).

The table holds every registry ``ui.prompt`` (the Capabilities examples), every ``ui.followups``
entry (the chips under a result), every ``example_prompts`` entry of ``GET /v1/capabilities`` and
the Workspace's empty-state suggestions (``design/screens/workspace-empty-*.png``) and the
proposal's Explain questions. Each runs with
the attachments the UI sends:

- ``pair``: the bundled sample (1 and 9 June). The Workspace sends both files with every message
  of a sample conversation, so every prompt has a pair case.
- ``one``: one file (9 June), for prompts that run a single-image analysis.
- ``none``: no file: an empty-state suggestion without the sample, or a question.

A message goes through ``POST /v1/chat``'s decision (:func:`server.routes.chat.converse`); an
analysis is then planned as the job worker plans it (``orchestrate`` on the job's attachments).
A ``none`` case that asks for the file is resumed with "here" and the sample pair, as the
Workspace's pending flow does. Expectations:

- ``reply``: a conversation reply, no job;
- ``ask:<item>``: a reply asking for the file, its ``pending`` naming ``<item>``;
- ``<item>@<0601|0609>``: a single-image analysis reading that file;
- ``<item>:<variant>``: a two-date analysis, its ``IndexType`` (change detection) or index name
  (band math).

A prompt the server cannot route as intended without the assistant is a strict xfail with the
reason, so the Workspace's chips stay honest: a fix makes it pass and the mark must go.
"""

from collections.abc import Iterator
import functools
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import rasterio

from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.models import model_states
from dta.dti.raster import describe_raster
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import ChatRequest as COEChatRequest
from server.capabilities import build_capabilities
from server.jobs import Job, JobQueue, JobStatus
from server.routes import chat as chat_routes
from server.schemas import AssistantStatus, ConversationReply, ConversationRequest
from server.sessions import Session
from server.utils import UPLOAD_DIR

# The sample as attachable files (tests/conftest.py sample_uploads)
pytestmark = pytest.mark.usefixtures("sample_uploads")
FILES = {"0601": str(UPLOAD_DIR / "hlsl_20230601.tif"), "0609": str(UPLOAD_DIR / "hlsl_20230609.tif")}
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
_SESSION = Session(id="routing", role="operator", created_at=0.0, expires_at=0.0)

# The suggestions under "What do you want to know?" (workspace-empty-with-sample.png and
# workspace-empty-no-sample.png; askWater, askFields and askClassify in the prototype; with
# the sample also askBurn, which is NDMI, and askPrithvi). The Workspace's list is
# cognitive_ui/src/components/chat/suggestions.ts, which checks it against this tuple.
EMPTY_STATE = (
    "Compare water before and after the dam breach",
    "Detect field boundaries east of the river",
    "Classify land cover on the flooded fields",
    "Map moisture change with NDMI between the two dates",
    "Fill the cloud gaps in the 9 June image",
)
# The proposal card's Explain button (DESIGN §4.2) and a typed question about the proposal: the
# Workspace sends them with the conversation's files, and they must stay conversation.
PROPOSAL_QUESTIONS = ("Explain this proposal", "Explain why you chose NDMI")

CD = "algorithms/change-detection"
NDMI = "algorithms/band-math-change:NDMI"

# prompt -> {attachments: expected route}
ROUTES: dict[str, dict[str, str]] = {
    # Built-in spectral indices and their follow-ups: one file, the later one of the pair.
    "Map vegetation health across the region": {
        "pair": "algorithms/ndvi@0609",
        "one": "algorithms/ndvi@0609",
        "none": "ask:algorithms/ndvi",
    },
    "Map vegetation health with NDVI": {
        "pair": "algorithms/ndvi@0609",
        "one": "algorithms/ndvi@0609",
        "none": "ask:algorithms/ndvi",
    },
    "Use EVI instead of NDVI for the forest": {
        "pair": "algorithms/evi@0609",
        "one": "algorithms/evi@0609",
        "none": "ask:algorithms/evi",
    },
    "Use EVI for dense vegetation": {
        "pair": "algorithms/evi@0609",
        "one": "algorithms/evi@0609",
        "none": "ask:algorithms/evi",
    },
    "Where is open water on 9 June?": {
        "pair": "algorithms/ndwi@0609",
        "one": "algorithms/ndwi@0609",
        "none": "reply",  # a question without a file: answered, not asked for a file
    },
    "Map open water with NDWI": {
        "pair": "algorithms/ndwi@0609",
        "one": "algorithms/ndwi@0609",
        "none": "ask:algorithms/ndwi",
    },
    "Map snow cover with NDSI": {
        "pair": "algorithms/ndsi@0609",
        "one": "algorithms/ndsi@0609",
        "none": "ask:algorithms/ndsi",
    },
    "Classify snow and ice robustly": {
        "pair": "algorithms/snow-classifier@0609",
        "one": "algorithms/snow-classifier@0609",
        "none": "ask:algorithms/snow-classifier",
    },
    "Give me statistics for every band": {
        "pair": "algorithms/statistics@0609",
        "one": "algorithms/statistics@0609",
        "none": "ask:algorithms/statistics",
    },
    "Classify land cover in this image": {
        "pair": "algorithms/lulc-classifier@0609",
        "one": "algorithms/lulc-classifier@0609",
        "none": "ask:algorithms/lulc-classifier",
    },
    # Also the composed "masked classification" example: without the assistant, the classifier alone.
    "Classify land cover on the flooded fields": {
        "pair": "algorithms/lulc-classifier@0609",
        "one": "algorithms/lulc-classifier@0609",
        "none": "ask:algorithms/lulc-classifier",
    },
    # Models: only their own phrases (the sample then fails on bands or the guest gate, at run time).
    "Fill the cloud gaps in the 9 June image": {
        "pair": "models/prithvi-reconstruction@0609",
        "one": "models/prithvi-reconstruction@0609",
        "none": "ask:models/prithvi-reconstruction",
    },
    "Detect field boundaries east of the river": {
        "pair": "models/delineate-anything@0609",
        "one": "models/delineate-anything@0609",
        "none": "ask:models/delineate-anything",
    },
    # Two dates.
    "Compare water before and after the dam breach": {"pair": f"{CD}:ndwi", "none": f"ask:{CD}"},
    "Compare with NDVI change between the same dates": {"pair": f"{CD}:ndvi", "none": f"ask:{CD}"},
    "Compare with NDWI water change between the same dates": {"pair": f"{CD}:ndwi", "none": f"ask:{CD}"},
    "Map moisture change with NDMI between the two dates": {
        "pair": NDMI,
        "none": "ask:algorithms/band-math-change",
    },
    "Redo the moisture map with a 0.1 threshold": {"pair": NDMI, "none": "ask:algorithms/band-math"},
    "Redo the water map with a 0.1 threshold": {"pair": f"{CD}:ndwi", "none": "ask:algorithms/ndwi"},
    "Track water weekly through June": {"pair": f"{CD}:ndwi", "none": "ask:algorithms/ndwi"},
    # Questions. Explanatory ones are conversation even with the sample attached; "Where is open
    # water on 9 June?" (above) locates, so it runs NDWI.
    "What else can you do with this data?": {"pair": "reply", "one": "reply", "none": "reply"},
    "Why is NDWI better than NDVI for water?": {"pair": "reply", "none": "reply"},
    "Explain the last result in two sentences": {"pair": "reply", "none": "reply"},
    "What would you do differently with Landsat?": {"pair": "reply", "none": "reply"},
    "Explain this proposal": {"pair": "reply", "none": "reply"},
    "Explain why you chose NDMI": {"pair": "reply", "none": "reply"},
}

# Typed prompts that are not on a chip, held to the same routing: a date in the prompt picks that
# file of the pair (the run-1 rule read the later one whatever the prompt said).
TYPED: dict[str, dict[str, str]] = {
    "Calculate NDVI for 1 June": {
        "pair": "algorithms/ndvi@0601",
        "one": "algorithms/ndvi@0609",  # no file is dated 1 June: the existing rule
        "none": "ask:algorithms/ndvi",
    },
    "Where is open water on 1 June?": {
        "pair": "algorithms/ndwi@0601",
        "one": "algorithms/ndwi@0609",
        "none": "reply",
    },
}

XFAIL: dict[tuple[str, str], str] = {
    ("Redo the water map with a 0.1 threshold", "pair"): (
        "composed tier (needs the assistant): re-running the last water analysis with a new threshold; "
        "without it the prompt plans a single-date NDWI and the threshold is not applied"
    ),
    ("Track water weekly through June", "pair"): (
        "composed tier (needs the assistant): a multi-date comparison; without it the prompt plans a single-date NDWI"
    ),
}


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Real router with Ollama on a closed port as its only provider: every generate() raises."""
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


@functools.cache
def _attachment(tag: str) -> dict[str, Any]:
    """What the UI sends for a sample file: the ``POST /v1/sample`` attachment, metadata included."""
    path = FILES[tag]
    with rasterio.open(path) as src:
        meta = describe_raster(src, Path(path).name, os.path.getsize(path)).model_dump()
    return {"id": "kakhovka", "filename": Path(path).name, "mime_type": "image/tiff", "path": path, "metadata": meta}


ATTACHMENTS = {"pair": ("0601", "0609"), "one": ("0609",), "none": ()}


def ui_prompts() -> set[str]:
    """Every prompt the Workspace can send from a chip: registry, capabilities and empty states."""
    prompts: set[str] = set(EMPTY_STATE) | set(PROPOSAL_QUESTIONS)
    for item in load_registry().instances:
        if item.ui is None:
            continue
        if item.ui.prompt:
            prompts.add(item.ui.prompt)
        followups = item.ui.followups
        for variant in followups.values() if isinstance(followups, dict) else [followups or []]:
            prompts.update(variant)
    doc = build_capabilities(_SESSION, AssistantStatus(connected=False, offline=True), model_states(), False)
    prompts.update(e["prompt"] for e in doc["example_prompts"])
    return prompts


def _plan_route(prompt: str, attachments: list[dict[str, Any]]) -> str:
    """The job worker's plan for a submitted analysis, as a route string."""
    job = Job(id="routing", status=JobStatus.PENDING, prompt=prompt, attachments=attachments)
    out = orchestrate(COEChatRequest(prompt=prompt, attachments=JobQueue._coe_attachments(job)))
    if out.get("intent") == "conversation":
        return "reply"
    if not out.get("ok"):
        return f"failed: {out.get('error')}"
    reg = load_registry()
    steps = out["plan"]["steps"]
    analysis = next(s for s in steps if get_item(reg, s["uses"]).kind in ("algorithm", "model"))
    tags = {path: tag for tag, path in FILES.items()}
    if "RasterPath" in (binds := {k: v for s in steps for k, v in s["binds"].items()}):
        return f"{analysis['uses']}@{tags[binds['RasterPath']]}"
    assert (binds["RasterPathBefore"], binds["RasterPathAfter"]) == (FILES["0601"], FILES["0609"]), binds
    variant = analysis["binds"].get("IndexType") or analysis.get("config", {}).get("name")
    return f"{analysis['uses']}:{variant}"


async def _route(message: str, state: str, pending: Any = None) -> tuple[str, ConversationReply | None]:
    """``(route, reply)`` for one Workspace message: through ``converse``, then the job worker's planner."""
    attachments = [_attachment(tag) for tag in ATTACHMENTS[state]]
    submitted: list[tuple[str, list[dict[str, Any]]]] = []

    async def analysis(prompt: str, req: ConversationRequest, session: Session) -> str:
        submitted.append((prompt, [a.model_dump() for a in req.attachments]))
        return "analysis"

    req = ConversationRequest(message=message, attachments=attachments, pending=pending)
    with patch.object(chat_routes, "_analysis", analysis):
        outcome = await chat_routes.converse(req, _SESSION)
    if submitted:
        return _plan_route(*submitted[0]), None
    assert isinstance(outcome, ConversationReply)
    if outcome.pending is None:
        return "reply", outcome
    return f"ask:{outcome.pending.item_id}", outcome


def _cases() -> Iterator[Any]:
    for prompt, routes in (ROUTES | TYPED).items():
        for state, expected in routes.items():
            reason = XFAIL.get((prompt, state))
            marks = [pytest.mark.xfail(raises=AssertionError, strict=True, reason=reason)] if reason else []
            yield pytest.param(prompt, state, expected, id=f"{state}:{prompt}", marks=marks)


@pytest.mark.parametrize(("prompt", "state", "expected"), list(_cases()))
async def test_ui_prompt_routes_as_intended(prompt: str, state: str, expected: str) -> None:
    route, reply = await _route(prompt, state)

    assert route == expected
    if reply is not None and reply.pending is not None and (prompt, "pair") not in XFAIL:
        # The pending flow: "here" with the sample pair runs the request that waited.
        resumed, _ = await _route("here", "pair", reply.pending)
        assert resumed == (ROUTES | TYPED)[prompt]["pair"]


def test_the_table_covers_every_ui_prompt() -> None:
    """A new chip (registry ui.prompt or followup, capabilities example) needs its row here."""
    assert ui_prompts() == set(ROUTES)
    for prompt, routes in ROUTES.items():
        assert {"pair", "none"} <= set(routes), prompt
        if "@" in routes["pair"]:  # a single-image analysis also runs on one file
            assert "one" in routes, prompt
    assert set(XFAIL) <= {(p, s) for p, routes in ROUTES.items() for s in routes}
