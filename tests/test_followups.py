"""Follow-up prompts under a result (v0.6.0 Workspace, DESIGN §4.6): ``done.followups`` and ``result.followups``.

Each prompt in the registry's ``ui.followups`` must route where it says, with the files the
Workspace re-sends (the analysis's own: one file, or the sample pair for a change analysis).
They are checked the way ``POST /v1/chat`` routes a message: the intent classifier, then the
orchestrator. No test here reaches a live provider: the router points at a closed Ollama port.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from dta.config import ROOT_DIR
from dta.dti.coe.intent_classifier import IntentType, classify_intent
from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Attachment, ChatRequest, ExecutionPlan, PlanStep
from server.job_events import DEFAULT_FOLLOWUPS, followups
from server.jobs import JobQueue, MemoryJobStore

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
CONVERSATION = "conversation"

# Where each follow-up must go: a conversation reply, or exactly one analysis step with these
# binds / config values.
EXPECTED: dict[str, str | tuple[str, dict[str, Any]]] = {
    "What else can you do with this data?": CONVERSATION,
    "Compare with NDVI change between the same dates": ("algorithms/change-detection", {"IndexType": "ndvi"}),
    "Compare with NDWI water change between the same dates": ("algorithms/change-detection", {"IndexType": "ndwi"}),
    "Map moisture change with NDMI between the two dates": ("algorithms/band-math-change", {"name": "NDMI"}),
    "Redo the moisture map with a 0.1 threshold": ("algorithms/band-math-change", {"name": "NDMI", "threshold": 0.1}),
    "Map vegetation health with NDVI": ("algorithms/ndvi", {}),
    "Use EVI for dense vegetation": ("algorithms/evi", {}),
    "Map open water with NDWI": ("algorithms/ndwi", {}),
    "Map snow cover with NDSI": ("algorithms/ndsi", {}),
    "Classify land cover in this image": ("algorithms/lulc-classifier", {}),
    "Give me statistics for every band": ("algorithms/statistics", {}),
}


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


def _registry_followups() -> list[tuple[str, str, str]]:
    """``(item id, variant, prompt)`` for every registry follow-up (variant ``""`` for a plain list)."""
    out = []
    for item in load_registry().instances:
        configured = item.ui.followups if item.ui else []
        variants = configured if isinstance(configured, dict) else {"": configured}
        out += [(item.id, variant, prompt) for variant, prompts in variants.items() for prompt in prompts]
    return out


def _attachments(*paths: str) -> list[Attachment]:
    return [Attachment(id=f"a{i}", filename=Path(p).name, mime_type="image/tiff", path=p) for i, p in enumerate(paths)]


def test_every_followup_has_an_expected_route() -> None:
    prompts = {prompt for _, _, prompt in _registry_followups()}

    assert prompts == set(EXPECTED)
    analyses = [i for i in load_registry().instances if i.kind in ("algorithm", "model") and i.ui]
    for item in analyses:
        if item.ui and item.ui.tier != "coming":
            configured = item.ui.followups
            lists = configured.values() if isinstance(configured, dict) else [configured]
            assert all(2 <= len(prompts) <= 3 for prompts in lists), item.id
    assert {"algorithms/band-math", "algorithms/band-math-change"} <= {i.id for i in analyses if i.ui.followups}


@pytest.mark.parametrize(("item_id", "variant", "prompt"), _registry_followups())
def test_followup_routes_where_expected(item_id: str, variant: str, prompt: str) -> None:
    """A follow-up never routes to an unrelated analysis, with the files of the result it follows."""
    item = get_item(load_registry(), item_id)
    files = [SAMPLE_0601, SAMPLE_0609] if "RasterPathBefore" in item.inputs else [SAMPLE_0601]
    req = ChatRequest(prompt=prompt, attachments=_attachments(*files))
    expected = EXPECTED[prompt]

    if classify_intent(req)["intent"] == IntentType.CONVERSATION:
        assert expected == CONVERSATION, f"{prompt!r} after {item_id} {variant} became a conversation reply"
        return
    assert expected != CONVERSATION, f"{prompt!r} after {item_id} {variant} became an analysis"
    out = orchestrate(req)
    assert out["ok"], out.get("error")
    reg = load_registry()
    steps = [s for s in out["plan"]["steps"] if get_item(reg, s["uses"]).kind in ("algorithm", "model")]
    uses, params = expected
    assert [s["uses"] for s in steps] == [uses], f"{prompt!r} after {item_id} {variant}"
    assert {k: {**steps[0]["binds"], **steps[0]["config"]}.get(k) for k in params} == params


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(flow="test", steps=[PlanStep(uses="input/file"), *steps], outputs=["publish: chat"])


@pytest.mark.parametrize(
    ("step", "first"),
    [
        (PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndwi"}), "Compare with NDVI"),
        (PlanStep(uses="algorithms/change-detection", binds={"IndexType": "NDVI"}), "Compare with NDWI"),
        (PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndsi"}), "Compare with NDVI"),
        (PlanStep(uses="algorithms/change-detection"), "Compare with NDVI"),
        (PlanStep(uses="algorithms/band-math-change", config={"name": "NDMI"}), "Redo the moisture map"),
        (PlanStep(uses="algorithms/band-math-change", config={"name": "SAVI"}), "Compare with NDVI"),
        (PlanStep(uses="algorithms/band-math", config={"name": "NDMI"}), "Classify land cover"),
        (PlanStep(uses="algorithms/ndvi"), "Use EVI"),
    ],
)
def test_followups_pick_the_variant(step: PlanStep, first: str) -> None:
    reg = load_registry()
    out = followups(_plan(step, PlanStep(uses="post-processing/agent-analysis")), reg)

    assert out[0].startswith(first)
    assert 2 <= len(out) <= 3


def test_followups_default_and_none() -> None:
    reg = load_registry()

    assert followups(_plan(PlanStep(uses="models/ndvi-convlstm")), reg) == DEFAULT_FOLLOWUPS  # no ui.followups
    assert followups(_plan(), reg) == DEFAULT_FOLLOWUPS  # no analysis step
    assert followups(None, reg) == []  # a conversation reply


# --- done.followups on a real job ----------------------------------------------------


@pytest.fixture
async def queue() -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    yield q
    await q.stop()


async def _until_terminal(q: JobQueue, job_id: str) -> list[dict[str, Any]]:
    subscription = q.subscribe(job_id)
    assert subscription is not None
    events, live = subscription
    try:
        while live is not None and (not events or events[-1]["type"] not in ("done", "failed", "cancelled")):
            events.append(await asyncio.wait_for(live.get(), 120))
    finally:
        if live is not None:
            q.unsubscribe(job_id, live)
    return events


async def test_ndwi_change_job_on_the_sample_ends_with_followups_and_water(queue: JobQueue) -> None:
    files = [
        {"id": Path(p).name, "filename": Path(p).name, "mime_type": "image/tiff", "path": p}
        for p in (SAMPLE_0601, SAMPLE_0609)
    ]
    job_id = await queue.submit_job("Compare water before and after the dam breach", attachments=files)
    await queue.start()

    events = await _until_terminal(queue, job_id)

    done = events[-1]
    assert done["type"] == "done", done
    assert done["data"]["followups"] == [
        "Compare with NDVI change between the same dates",
        "Map moisture change with NDMI between the two dates",
        "What else can you do with this data?",
    ]
    job = await queue.get_job(job_id)
    assert job is not None and job.result is not None
    assert job.result["followups"] == done["data"]["followups"]
    stats = job.result["execution"]["artifacts"]["ChangeMap"]["statistics"]
    assert stats["water"]["gained_km2"] == 156.761
    assert stats["areas_km2"]["increase"] == 325.227
