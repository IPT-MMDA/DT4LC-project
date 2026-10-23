"""Plan proposal and confirmation (v0.6.0 story).

A plan that is not a single registered template (band math, or a new combination
of steps) is offered as a proposal before it runs. Named indices no built-in
analysis computes (NDMI, NBR, ...) are composed from a table without an LLM, or
refused when the files lack a band: the committed sample has B2–B6, so NBR (B7)
is refused and NDMI (B5, B6) is proposed.

LLM paths use a scripted router; every other test points the real router at a
closed Ollama port, so no test reaches a live provider.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta
import functools
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import rasterio

from dta.config import ROOT_DIR
from dta.dti.coe import planner as planner_mod
from dta.dti.coe.llm import LLMResponse, get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.named_indices import NAMED_INDICES, match_named_index, proposal_text
from dta.dti.coe.orchestrator import NO_ANALYSIS_STEP, orchestrate
from dta.dti.raster import describe_raster
from dta.dti.registry import load_registry
from dta.dti.schemas import Attachment, ChatRequest, ExecutionPlan
from server import jobs as jobs_mod
from server.job_events import KEEPALIVE_FRAME, sse_event_frame
from server.jobs import Job, JobQueue, JobStatus, SQLiteJobStore
from server.routes import jobs as job_routes

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")

NDMI_PROMPT = "Map moisture change with NDMI between the two dates"
NDMI_TEXT = (
    "I don't have a moisture-change algorithm. I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) "
    "from B5 and B6 for both dates and difference them — run that?"
)
NDMI_FORMULA = "(B5 - B6) / (B5 + B6)"
BURN_MESSAGE = (
    "Burn severity uses NBR = (NIR − SWIR2)/(NIR + SWIR2), which needs band B7 (SWIR2). "
    "The sample has B2–B6, so I can't compute it here."
)
POST = "post-processing/agent-analysis"


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


def _attachments(*paths: str) -> list[Attachment]:
    return [Attachment(id=f"a{i}", filename=Path(p).name, mime_type="image/tiff", path=p) for i, p in enumerate(paths)]


class ScriptedLLM:
    """Router stub answering each COE stage from its system prompt: intent, context, plan."""

    providers: list[Any] = []

    def __init__(
        self, plan_steps: list[dict[str, Any]], context: dict[str, Any] | None = None, intent: str = "pipeline"
    ) -> None:
        self.plan_steps = plan_steps
        self.intent = intent
        self.context = context or {"goal": "x", "desired_outputs": [], "required_inputs": ["RasterPath"]}
        self.stages: list[str] = []

    def generate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResponse:
        system = messages[0].content
        if system.startswith("You are an Intent Classifier"):
            stage, text = "intent", json.dumps({"intent": self.intent, "reason": "scripted", "response": "Chat."})
        elif system.startswith("You are a Context Understanding Agent"):
            stage, text = "context", json.dumps({**self.context, "hints": {"keywords": []}})
        elif system.startswith("You are a pipeline planner"):
            stage, text = "plan", json.dumps({"steps": self.plan_steps, "reasoning": "scripted"})
        else:
            stage, text = "other", "Scripted interpretation."
        self.stages.append(stage)
        return LLMResponse(text=text, model="scripted", provider="scripted")


def _with_llm(llm: ScriptedLLM) -> Any:
    """Patch every COE stage's router with ``llm``."""
    from contextlib import ExitStack

    stack = ExitStack()
    for module in ("intent_classifier", "context_agent", "llm_planner"):
        stack.enter_context(patch(f"dta.dti.coe.{module}.get_llm_router", return_value=llm))
    return stack


NBR_PLAN = [
    {"uses": "input/file-before"},
    {"uses": "input/file-after"},
    {"uses": "algorithms/band-math-change", "config": {"formula": "(B5 - B7) / (B5 + B7)", "name": "NBR"}},
    {"uses": POST},
]


# --- The named-index table -------------------------------------------------------------


def test_named_index_table() -> None:
    table = {i.name: i.formula.format(red="B4", nir="B5", swir1="B6", swir2="B7") for i in NAMED_INDICES}
    assert table == {
        "NDMI": "(B5 - B6) / (B5 + B6)",
        "NBR": "(B5 - B7) / (B5 + B7)",
        "SAVI": "1.5 * (B5 - B4) / (B5 + B4 + 0.5)",
        "NDBI": "(B6 - B5) / (B6 + B5)",
    }


@pytest.mark.parametrize(
    ("prompt", "name"),
    [
        (NDMI_PROMPT, "NDMI"),
        ("Map soil moisture", "NDMI"),
        ("Map burn severity", "NBR"),
        ("Where did it burn?", "NBR"),
        ("compute NBR", "NBR"),
        ("Map moisture with NDWI", None),  # a built-in index is named: not a composed request
        ("Detect water changes between these images", None),
        ("Map the burnished roofs", None),  # whole words only
    ],
)
def test_prompt_names_an_index(prompt: str, name: str | None) -> None:
    index = match_named_index(prompt)
    assert (index.name if index else None) == name


# --- Proposals from the table (no LLM) ---------------------------------------------------


def test_ndmi_on_the_two_sample_files_is_a_band_math_change_proposal() -> None:
    out = orchestrate(ChatRequest(prompt=NDMI_PROMPT, attachments=_attachments(SAMPLE_0601, SAMPLE_0609)))

    assert out["ok"], out.get("error")
    assert out["proposal"] == NDMI_TEXT
    plan = out["plan"]
    assert plan["composed"] is True
    assert [s["uses"] for s in plan["steps"]] == [
        "input/file-before",
        "input/file-after",
        "algorithms/band-math-change",
        POST,
    ]
    assert plan["steps"][0]["binds"] == {"RasterPathBefore": SAMPLE_0601}
    assert plan["steps"][1]["binds"] == {"RasterPathAfter": SAMPLE_0609}
    assert plan["steps"][2]["config"] == {"formula": NDMI_FORMULA, "name": "NDMI", "threshold": 0.2}
    assert proposal_text(ExecutionPlan(**plan), load_registry()) == NDMI_TEXT


def test_ndmi_on_one_file_is_a_single_date_proposal() -> None:
    out = orchestrate(ChatRequest(prompt="Map moisture with NDMI", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["proposal"] == (
        "I don't have a moisture algorithm. "
        "I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) from B5 and B6 — run that?"
    )
    assert [s["uses"] for s in out["plan"]["steps"]] == ["input/file", "algorithms/band-math", POST]
    assert out["plan"]["steps"][1]["config"] == {"formula": NDMI_FORMULA, "name": "NDMI"}


def test_ndmi_is_an_analysis_even_if_the_llm_would_chat() -> None:
    """Live, Gemini classified the NDMI prompt as conversation: no registry keyword matched it."""
    llm = ScriptedLLM([], intent="conversation")
    with _with_llm(llm):
        out = orchestrate(ChatRequest(prompt=NDMI_PROMPT, attachments=_attachments(SAMPLE_0601, SAMPLE_0609)))
        question = orchestrate(ChatRequest(prompt="What is NDMI?", attachments=_attachments(SAMPLE_0601)))

    assert out["intent"] == "pipeline"
    assert out["proposal"] == NDMI_TEXT
    assert question["intent"] == "conversation"  # a question about an index is still answered


# --- Burn severity is refused (B7) ------------------------------------------------------


@pytest.mark.parametrize("files", [[SAMPLE_0601], [SAMPLE_0601, SAMPLE_0609]], ids=["one-file", "two-files"])
def test_burn_severity_on_the_sample_is_refused_without_llm(files: list[str]) -> None:
    with patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(ChatRequest(prompt="Map burn severity", attachments=_attachments(*files)))

    assert out["ok"] is False
    assert out["error"] == BURN_MESSAGE
    assert "NBR" in json.dumps(out["candidate"])
    template.assert_not_called()  # refused before planning: never an unrelated template


def test_burn_severity_is_refused_with_an_llm_that_proposes_nbr() -> None:
    llm = ScriptedLLM(NBR_PLAN)
    with _with_llm(llm), patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(ChatRequest(prompt="Map burn severity", attachments=_attachments(SAMPLE_0601, SAMPLE_0609)))

    assert out["ok"] is False
    assert out["error"] == BURN_MESSAGE
    assert "plan" not in llm.stages  # the table refuses before the LLM planner
    template.assert_not_called()


def test_llm_plan_naming_a_missing_band_stops_without_template_retry() -> None:
    """An LLM PlanError for a band the file lacks is final: no template plan runs instead."""
    llm = ScriptedLLM(NBR_PLAN)
    with _with_llm(llm), patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(
            ChatRequest(prompt="Compute (B5 - B7) / (B5 + B7)", attachments=_attachments(SAMPLE_0601, SAMPLE_0609))
        )

    assert "plan" in llm.stages
    assert out["ok"] is False
    assert out["error"].startswith("Step algorithms/band-math-change: Band B7 is not in this file (B2–B6).")
    template.assert_not_called()


# --- Composed LLM plans ----------------------------------------------------------------------


def test_llm_band_math_plan_is_a_proposal_with_a_generic_text() -> None:
    steps = [
        {"uses": "input/file"},
        {"uses": "algorithms/band-math", "config": {"formula": "(B4 - B3) / (B4 + B3)"}},
        {"uses": POST},
    ]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="Compute (B4 - B3) / (B4 + B3)", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is True
    assert out["proposal"] == (
        "No built-in analysis matches this. I can compute (B4 - B3) / (B4 + B3) from B4 and B3 — run that?"
    )


def test_llm_plan_combining_two_analyses_is_a_proposal() -> None:
    steps = [{"uses": "input/file"}, {"uses": "algorithms/ndvi"}, {"uses": "algorithms/statistics"}, {"uses": POST}]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="NDVI then the statistics", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is True
    assert out["proposal"].startswith("No single built-in analysis matches this. I can run ")


def test_llm_plan_matching_a_template_runs_without_proposal() -> None:
    steps = [{"uses": "input/file"}, {"uses": "algorithms/ndvi"}, {"uses": POST}]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="Show plant vigour", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is False
    assert "proposal" not in out


# --- A plan with no analysis step is not accepted -------------------------------------------


def test_plan_without_analysis_step_is_not_accepted() -> None:
    out = orchestrate(ChatRequest(prompt="Show me the landscape", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"] is False
    assert out["error"] == NO_ANALYSIS_STEP
    assert [s["uses"] for s in out["candidate"]["steps"]] == ["input/file", POST]


def test_skeletal_llm_plan_falls_back_to_the_template() -> None:
    """HD fallback: an LLM plan that only reads and summarises is replaced by the template plan."""
    ctx = {"goal": "calculate NDVI", "desired_outputs": ["NDVIMap"], "required_inputs": ["RasterPath"]}
    with _with_llm(ScriptedLLM([{"uses": "input/file"}, {"uses": POST}], context=ctx)):
        out = orchestrate(ChatRequest(prompt="calculate NDVI please", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert [s["uses"] for s in out["plan"]["steps"]] == ["input/file", "algorithms/ndvi", POST]
    assert out["plan"]["composed"] is False


# --- The job API: proposal, confirm, cancel, quota ----------------------------------------


@functools.cache
def _sample_attachment(path: str) -> dict[str, Any]:
    """The attachment the UI sends for a sample file: what ``POST /v1/upload`` returns."""
    name = Path(path).name
    with rasterio.open(path) as src:
        meta = describe_raster(src, name, os.path.getsize(path)).model_dump()
    return {"id": name, "filename": name, "mime_type": "image/tiff", "path": path, "metadata": meta}


SAMPLE_PAIR = [SAMPLE_0601, SAMPLE_0609]


@pytest.fixture
async def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    await q.start()
    yield q
    await q.stop()


def _client() -> httpx.AsyncClient:
    """A browser on the app (no lifespan): its own cookie jar, so its own session."""
    from server.app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _events_until(q: JobQueue, job_id: str, last: set[str], timeout: float = 120) -> list[dict[str, Any]]:
    """The job's events up to the first whose type is in ``last``, as a stream subscriber sees them."""
    subscription = q.subscribe(job_id)
    assert subscription is not None
    events, live = subscription
    try:
        while not any(e["type"] in last for e in events):
            assert live is not None, [e["type"] for e in events]
            events.append(await asyncio.wait_for(live.get(), timeout))
    finally:
        if live is not None:
            q.unsubscribe(job_id, live)
    return events[: next(i for i, e in enumerate(events) if e["type"] in last) + 1]


async def _submit(client: httpx.AsyncClient, prompt: str, paths: list[str]) -> dict[str, Any]:
    r = await client.post("/v1/jobs", json={"prompt": prompt, "attachments": [_sample_attachment(p) for p in paths]})
    assert r.status_code == 202, r.text
    return r.json()  # type: ignore[no-any-return]


async def _proposal(q: JobQueue, client: httpx.AsyncClient, prompt: str = NDMI_PROMPT) -> str:
    job_id = (await _submit(client, prompt, SAMPLE_PAIR))["id"]
    assert (await _events_until(q, job_id, {"proposal", "failed", "done"}))[-1]["type"] == "proposal"
    return str(job_id)


async def _used(client: httpx.AsyncClient) -> int:
    return int((await client.get("/v1/session")).json()["analyses"]["used"])


async def test_ndmi_job_proposes_then_confirm_runs_it_on_the_same_stream(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setattr(job_routes, "KEEPALIVE_S", 0.05)
    async with _client() as client:
        submitted = await _submit(client, NDMI_PROMPT, SAMPLE_PAIR)
        job_id = submitted["id"]
        assert submitted["status"] == "pending"
        assert [e["type"] for e in submitted["events"]] == ["intent"]

        seq = await _events_until(queue, job_id, {"proposal", "failed", "done"})
        assert [e["type"] for e in seq] == ["intent", "planning", "proposal"]
        assert seq[0]["data"]["text"] == "I'll compute NDMI for 1 June and 9 June from the sample."
        assert seq[1]["data"]["phase"] == "started"
        proposal = seq[2]["data"]
        assert proposal["text"] == NDMI_TEXT
        assert proposal["plan"]["composed"] is True
        assert proposal["plan"]["steps"][2]["uses"] == "algorithms/band-math-change"
        assert proposal["plan"]["steps"][2]["config"]["formula"] == NDMI_FORMULA

        record = (await client.get(f"/v1/jobs/{job_id}")).json()
        assert record["status"] == "proposal"
        assert record["proposal"] == {"text": NDMI_TEXT, "plan": proposal["plan"]}
        assert record["plan"]["composed"] is True
        listed = (await client.get("/v1/jobs", params={"status": "proposal"})).json()
        assert [j["status"] for j in listed["jobs"]] == ["proposal"]
        assert await _used(client) == 0  # a proposal is not an analysis

        # The event stream stays open after the proposal: the backlog, then keep-alives.
        subscription = queue.subscribe(job_id)
        assert subscription is not None
        backlog, live = subscription
        assert live is not None
        frames = job_routes._event_frames(queue, job_id, backlog, live)
        assert [await anext(frames) for _ in backlog] == [sse_event_frame(e) for e in seq]
        assert await anext(frames) == KEEPALIVE_FRAME

        r = await client.post(f"/v1/jobs/{job_id}/confirm")
        assert r.status_code == 202, r.text
        assert r.json()["id"] == job_id
        assert await _used(client) == 1  # the confirmation counts

        # The same stream carries the run.
        rest = []
        async for frame in frames:
            if frame != KEEPALIVE_FRAME:
                rest.append(frame)
        types = [f.split(b"\n")[1].removeprefix(b"event: ").decode() for f in rest]
        assert types == ["planning", *["step_started", "step_finished"] * 4, "done"]
        ready = json.loads(rest[0].split(b"\n")[2].removeprefix(b"data: "))
        assert ready["phase"] == "ready"
        assert ready["composed"] is True
        assert [s["id"] for s in ready["steps"]] == [s["uses"] for s in proposal["plan"]["steps"]]

        job = (await client.get(f"/v1/jobs/{job_id}")).json()
    assert job["status"] == "completed", job["error"]
    result = job["result"]["execution"]["artifacts"]["BandMathChangeMap"]
    assert result["formula"] == NDMI_FORMULA
    assert result["name"] == "NDMI"
    assert [e["seq"] for e in job["events"]] == list(range(1, len(job["events"]) + 1))


@pytest.mark.parametrize("with_llm", [False, True], ids=["no-llm", "llm-proposes-nbr"])
async def test_burn_severity_job_fails_with_band_missing(queue: JobQueue, with_llm: bool) -> None:
    from contextlib import nullcontext

    with (
        _with_llm(ScriptedLLM(NBR_PLAN)) if with_llm else nullcontext(),
        patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template),
    ):
        async with _client() as client:
            job_id = (await _submit(client, "Map burn severity", SAMPLE_PAIR))["id"]
            seq = await _events_until(queue, job_id, {"proposal", "failed", "done"})

    assert [e["type"] for e in seq] == ["intent", "planning", "failed"]  # never a ready plan or a step
    failed = seq[-1]["data"]
    assert failed["code"] == "BAND_MISSING"
    assert failed["message"] == BURN_MESSAGE
    assert [(f["label"], f["kind"]) for f in failed["fixes"]] == [
        ("Map moisture change with NDMI instead", "prompt"),
        ("Use my own image", "upload"),
    ]
    assert failed["fixes"][0]["prompt"] == NDMI_PROMPT


async def test_confirm_is_owner_scoped_and_once_only(queue: JobQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    async with _client() as owner, _client() as stranger:
        job_id = await _proposal(queue, owner)

        r = await stranger.post(f"/v1/jobs/{job_id}/confirm")
        assert r.status_code == 404
        assert (await owner.get(f"/v1/jobs/{job_id}")).json()["status"] == "proposal"

        assert (await owner.post(f"/v1/jobs/{job_id}/confirm")).status_code == 202
        r = await owner.post(f"/v1/jobs/{job_id}/confirm")
        assert r.status_code == 409
        assert "is not a proposal" in r.json()["error"]["message"]
        assert await _used(owner) == 1  # the refused second confirmation did not count
        await _events_until(queue, job_id, {"done", "failed"})

        assert (await owner.post("/v1/jobs/nope/confirm")).status_code == 404


async def test_confirm_a_template_job_is_a_conflict(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = (await _submit(client, "calculate NDVI", [SAMPLE_0601]))["id"]
        await _events_until(queue, job_id, {"done", "failed"})
        r = await client.post(f"/v1/jobs/{job_id}/confirm")
    assert r.status_code == 409


async def test_edited_plan_is_validated_and_reads_only_the_jobs_files(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _proposal(queue, client)
        plan = (await client.get(f"/v1/jobs/{job_id}")).json()["proposal"]["plan"]

        nbr = json.loads(json.dumps(plan))
        nbr["steps"][2]["config"] = {"formula": "(B5 - B7) / (B5 + B7)", "name": "NBR"}
        r = await client.post(f"/v1/jobs/{job_id}/confirm", json={"plan": nbr})
        assert r.status_code == 422
        assert r.json()["error"]["message"].startswith(
            "Step algorithms/band-math-change: Band B7 is not in this file (B2–B6)."
        )

        unsafe = json.loads(json.dumps(plan))
        unsafe["steps"][2]["config"]["formula"] = "abs(B5)"
        r = await client.post(f"/v1/jobs/{job_id}/confirm", json={"plan": unsafe})
        assert r.status_code == 422
        assert "Formula not allowed: `abs(B5)`" in r.json()["error"]["message"]

        skeletal = {**plan, "steps": [plan["steps"][0], plan["steps"][3]]}
        r = await client.post(f"/v1/jobs/{job_id}/confirm", json={"plan": skeletal})
        assert r.status_code == 422
        assert r.json()["error"]["message"] == NO_ANALYSIS_STEP

        r = await client.post(f"/v1/jobs/{job_id}/confirm", json={"plan": {"steps": [{"uses": "algorithms/nope"}]}})
        assert r.status_code == 422
        assert (await client.get(f"/v1/jobs/{job_id}")).json()["status"] == "proposal"

        # A valid edit runs; its file inputs are the job's attachments, whatever the body says.
        edited = json.loads(json.dumps(plan))
        edited["steps"][0]["binds"] = {"RasterPathBefore": "/etc/passwd"}
        edited["steps"][2]["config"]["threshold"] = 0.1
        r = await client.post(f"/v1/jobs/{job_id}/confirm", json={"plan": edited})
        assert r.status_code == 202, r.text
        assert r.json()["plan"]["steps"][0]["binds"] == {"RasterPathBefore": SAMPLE_0601}
        assert r.json()["plan"]["steps"][2]["config"]["threshold"] == 0.1
        assert (await _events_until(queue, job_id, {"done", "failed"}))[-1]["type"] == "done"


async def test_cancel_a_proposal(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _proposal(queue, client)
        r = await client.post(f"/v1/jobs/{job_id}/cancel")
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "cancelled"
        seq = await _events_until(queue, job_id, {"cancelled"})
        assert [e["type"] for e in seq] == ["intent", "planning", "proposal", "cancelled"]
        assert (await client.post(f"/v1/jobs/{job_id}/confirm")).status_code == 409


async def test_guest_quota_counts_the_confirmation_not_the_proposal(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setenv("GUEST_ANALYSES_PER_HOUR", "1")
    async with _client() as client:
        first = await _proposal(queue, client)
        second = await _proposal(queue, client)  # proposals never use up the quota
        assert await _used(client) == 0

        assert (await client.post(f"/v1/jobs/{first}/confirm")).status_code == 202
        r = await client.post(f"/v1/jobs/{second}/confirm")
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "GUEST_QUOTA"
        assert (await client.get(f"/v1/jobs/{second}")).json()["status"] == "proposal"  # still confirmable later
        await _events_until(queue, first, {"done", "failed"})


async def test_an_expired_proposal_ends_its_stream(tmp_path: Path) -> None:
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    try:
        job = Job(
            id="p1",
            status=JobStatus.PROPOSAL,
            prompt=NDMI_PROMPT,
            owner_role="guest",
            created_at=datetime.now() - timedelta(hours=25),
            events=[{"seq": 1, "type": "proposal", "data": {"text": NDMI_TEXT, "plan": {}, "t": 0.0}}],
        )
        q._store.save(job)
        subscription = q.subscribe("p1")
        assert subscription is not None and subscription[1] is not None
        live = subscription[1]

        assert await q.cleanup_old_jobs(guest_hours=24) == 1

        event = await asyncio.wait_for(live.get(), 5)
        assert (event["type"], event["data"]["reason"]) == ("cancelled", "expired")
        assert await q.get_job("p1") is None
    finally:
        await q.stop()


@pytest.mark.parametrize(
    ("prompt", "threshold"),
    [
        ("Redo the moisture map with a 0.1 threshold", 0.1),
        ("Map NDMI change with a threshold of 0.15", 0.15),
        ("Map moisture change with NDMI between the two dates", None),
        ("Map NDMI change with a 0 threshold", None),
    ],
)
def test_named_index_change_takes_the_prompt_threshold(prompt: str, threshold: float | None) -> None:
    """The NDMI follow-up "Redo the moisture map with a 0.1 threshold" must not run at the default 0.2."""
    from dta.dti.coe.named_indices import named_index_request, prompt_threshold

    assert prompt_threshold(prompt) == threshold
    request = named_index_request(prompt, _attachments(SAMPLE_0601, SAMPLE_0609), load_registry())
    assert request is not None
    assert request.plan.steps[2].config["threshold"] == (threshold or 0.2)
