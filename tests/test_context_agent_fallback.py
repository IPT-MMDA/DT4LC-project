"""Hybrid-deterministic fallback of the context agent (JOSS reviewer 2 finding 4).

The context agent uses an LLM reply only when it passes the structural check
``_accept`` (well-formed JSON, the ContextUnderstanding schema, non-blank goal,
registry types only, string keywords). When every provider fails, or every
reply is rejected, it returns ``_deterministic_context``, built from the
registry triggers, so the job path keeps working without any LLM.

No test here can reach a live provider: ``_providers_down`` points the real
router at a closed Ollama port and refuses to run if it holds anything else,
even when a ``.env`` with API keys has been loaded.
"""

import asyncio
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dta.config import ROOT_DIR
from dta.dti.coe import context_agent
from dta.dti.coe.context_agent import analyze
from dta.dti.coe.llm import LLMResponse, get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.orchestrator import _inject_file_bindings, orchestrate
from dta.dti.coe.plan_validator import validate
from dta.dti.coe.planner import plan
from dta.dti.coe.triggers import get_trigger_index
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding
from server.jobs import JobQueue, JobStatus, SQLiteJobStore

BEFORE = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
AFTER = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_CLOSED_PORT = "http://127.0.0.1:9"
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Real router with Ollama on a closed port as its only provider: every generate() raises."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    # load_dotenv() never overrides set variables, so a later server.app import cannot undo these.
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", _CLOSED_PORT)
    reset_llm_router()
    providers = get_llm_router().providers
    assert providers and all(isinstance(p, OllamaProvider) and p.base_url == _CLOSED_PORT for p in providers)
    yield
    reset_llm_router()


def _attachments(*paths: str) -> list[Attachment]:
    return [Attachment(id=f"a{i}", filename=Path(p).name, mime_type="image/tiff", path=p) for i, p in enumerate(paths)]


def _ctx_from(payload: dict[str, Any]) -> ContextUnderstanding:
    return ContextUnderstanding(**payload)


class _ReplyRouter:
    """Router stub: every generate() returns the same text, like an available provider."""

    providers: list[Any] = []

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text=self.text, model="stub", provider="stub")


# --- U1: every provider down -------------------------------------------------


def test_outage_returns_deterministic_context(caplog: pytest.LogCaptureFixture) -> None:
    reg = load_registry()
    req = ChatRequest(prompt="calculate NDVI", attachments=_attachments(BEFORE))

    with caplog.at_level("WARNING", logger="dta.dti.coe.context_agent"):
        ctx = analyze(req, reg.types)

    ndvi = get_item(reg, "algorithms/ndvi")
    assert ndvi.triggers is not None
    assert ctx == ContextUnderstanding(
        goal="calculate NDVI",
        desired_outputs=["NDVIMap"],
        required_inputs=["RasterPath"],
        hints={"keywords": ndvi.triggers.keywords},
    )
    assert "All LLM providers failed: ollama" in caplog.text


@pytest.mark.parametrize(
    ("prompt", "files", "steps", "index_type"),
    [
        ("calculate NDVI", [BEFORE], ["input/file", "algorithms/ndvi", "post-processing/agent-analysis"], None),
        (
            "detect changes",
            [BEFORE, AFTER],
            ["input/file-before", "input/file-after", "algorithms/change-detection", "post-processing/agent-analysis"],
            "ndvi",  # no index named in the goal: the registry default
        ),
        (
            "detect water changes",
            [BEFORE, AFTER],
            ["input/file-before", "input/file-after", "algorithms/change-detection", "post-processing/agent-analysis"],
            "ndwi",  # "water" in the goal, via the change-detection index_keyword_map
        ),
        (
            "detect field boundaries in this image",
            [BEFORE],
            ["input/file", "models/delineate-anything", "post-processing/agent-analysis"],
            None,
        ),
    ],
)
def test_outage_orchestrate_plans(prompt: str, files: list[str], steps: list[str], index_type: str | None) -> None:
    out = orchestrate(ChatRequest(prompt=prompt, attachments=_attachments(*files)))

    assert out["ok"], out.get("error")
    plan_steps = out["plan"]["steps"]
    assert [s["uses"] for s in plan_steps] == steps
    binds = {k: v for s in plan_steps for k, v in s["binds"].items()}
    if len(files) == 2:
        assert binds["RasterPathBefore"] == BEFORE and binds["RasterPathAfter"] == AFTER
        assert binds["IndexType"] == index_type
    else:
        assert binds["RasterPath"] == BEFORE


# --- U2: a reply that fails the structural check -----------------------------

_VALID = {
    "goal": "calculate NDVI",
    "desired_outputs": ["NDVIMap"],
    "required_inputs": ["RasterPath"],
    "hints": {"keywords": ["ndvi"]},
}


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        pytest.param("Sure, I will calculate NDVI for you.", "no JSON object", id="prose"),
        pytest.param("{goal: calculate NDVI, desired_outputs: [NDVIMap]}", "malformed JSON", id="non-json-braces"),
        pytest.param(json.dumps({**_VALID, "desired_outputs": [{"type": "NDVIMap"}]}), "schema", id="wrong-types"),
        pytest.param(json.dumps({"goal": "calculate NDVI"}), "schema", id="missing-keys"),
        pytest.param(json.dumps({**_VALID, "hints": ["ndvi"]}), "schema", id="hints-list"),
        pytest.param(json.dumps({**_VALID, "hints": {"keywords": [1, 2]}}), "keywords", id="non-str-keywords"),
        pytest.param(json.dumps({**_VALID, "desired_outputs": ["PrettyMap"]}), "PrettyMap", id="unknown-type"),
        pytest.param(json.dumps({**_VALID, "required_inputs": ["Raster Path"]}), "Raster Path", id="unknown-input"),
        pytest.param(json.dumps({**_VALID, "goal": "  "}), "blank goal", id="blank-goal"),
        pytest.param(json.dumps([{"goal": "calculate NDVI"}]), "schema", id="json-array"),
    ],
)
def test_rejected_reply_returns_deterministic_context(
    reply: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    reg = load_registry()
    req = ChatRequest(prompt="calculate NDVI", attachments=_attachments(BEFORE))

    with (
        patch("dta.dti.coe.context_agent.get_llm_router", return_value=_ReplyRouter(reply)),
        caplog.at_level("WARNING", logger="dta.dti.coe.context_agent"),
    ):
        ctx = analyze(req, reg.types)

    assert ctx == context_agent._deterministic_context(req)
    assert reason in caplog.text
    assert "using deterministic context" in caplog.text


def test_extra_keys_are_accepted_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    """Approved decision: keys outside the schema do not reject an otherwise valid reply."""
    reg = load_registry()
    reply = json.dumps({**_VALID, "confidence": 0.9, "explanation": "NDVI needs red and NIR"})

    with (
        patch("dta.dti.coe.context_agent.get_llm_router", return_value=_ReplyRouter(reply)),
        caplog.at_level("WARNING", logger="dta.dti.coe.context_agent"),
    ):
        ctx = analyze(ChatRequest(prompt="please calculate NDVI", attachments=[]), reg.types)

    assert ctx == _ctx_from(_VALID)
    assert "confidence" not in ctx.model_dump()
    assert caplog.text == ""


def test_accept_rejects_a_json_array() -> None:
    assert context_agent._accept([_VALID], load_registry().types) is None
    assert context_agent._accept(_VALID, load_registry().types) == _ctx_from(_VALID)


# --- P1: delta_t, every trigger phrase planned without any LLM ----------------
# A phrase that cannot reach an accepted plan with an analysis step counts towards delta_t;
# mark it with an explicit strict xfail and its reason rather than loosening the rules.


def _p1_cases() -> Iterator[Any]:
    for item in get_trigger_index().user_runnable_items:
        if item.triggers is None:
            continue
        files = [BEFORE, AFTER] if "RasterPathBefore" in item.inputs else [BEFORE]
        for phrase in dict.fromkeys(item.triggers.keywords + item.triggers.action_phrases):
            yield pytest.param(phrase, files, id=f"{item.id}:{phrase}")


@pytest.mark.parametrize(("phrase", "files"), list(_p1_cases()))
def test_trigger_phrase_plans_an_analysis_step_without_llm(phrase: str, files: list[str]) -> None:
    reg = load_registry()
    req = ChatRequest(prompt=phrase, attachments=_attachments(*files))

    candidate = plan(analyze(req, reg.types), reg)
    _inject_file_bindings(candidate, req.attachments)
    final = validate(candidate, reg)

    assert any(get_item(reg, s.uses).kind in ("algorithm", "model") for s in final.steps), final.steps


# --- I1: the whole job path with every provider down --------------------------


@pytest.mark.parametrize(
    ("prompt", "files", "output"),
    [("calculate NDVI", [BEFORE], "NDVIMap"), ("detect changes", [BEFORE, AFTER], "ChangeMap")],
)
async def test_job_completes_with_every_provider_down(
    tmp_path: Path, prompt: str, files: list[str], output: str
) -> None:
    queue = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    await queue.start()
    try:
        attachments = [a.model_dump() for a in _attachments(*files)]
        job_id = await queue.submit_job(prompt, attachments=attachments)
        for _ in range(240):
            job = await queue.get_job(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            await asyncio.sleep(0.25)

        job = await queue.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED, job.error
        assert job.result is not None
        rasters = job.result["execution"]["artifacts"][output]["rasters"]
        assert rasters and all(r["kind"] == "geotiff" and Path(r["path"]).is_file() for r in rasters.values())
    finally:
        await queue.stop()
