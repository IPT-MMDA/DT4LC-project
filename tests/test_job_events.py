"""Job event stream: ``GET /v1/jobs/{id}/events`` (v0.6.0 story "Job event stream").

The jobs run the real orchestrator with every LLM provider down (Ollama on a
closed port, API keys removed), so the hybrid-deterministic fallbacks make the
plan and the event sequence deterministic.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
import functools
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import rasterio

from dta.config import ROOT_DIR
from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.executor import ExecutionError, MissingInputError, PipelineExecutor
from dta.dti.raster import describe_raster
from dta.dti.schemas import PlanStep
from server import jobs as jobs_mod
from server.job_events import KEEPALIVE_FRAME, classify_failure, done_summary, sse_event_frame
from server.jobs import Job, JobQueue, JobStatus, SQLiteJobStore
from server.routes import jobs as job_routes

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_CLOSED_PORT = "http://127.0.0.1:9"
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
_FIX_KINDS = {"prompt", "retry", "upload", "open_map"}
NDVI_STEPS = ["input/file", "algorithms/ndvi", "post-processing/agent-analysis"]


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Real router with Ollama on a closed port as its only provider: every generate() raises."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", _CLOSED_PORT)
    reset_llm_router()
    providers = get_llm_router().providers
    assert providers and all(isinstance(p, OllamaProvider) and p.base_url == _CLOSED_PORT for p in providers)
    yield
    reset_llm_router()


@functools.cache
def _sample_attachment(path: str) -> dict[str, Any]:
    """The attachment the UI sends for a sample file: what ``POST /v1/upload`` returns."""
    name = Path(path).name
    with rasterio.open(path) as src:
        meta = describe_raster(src, name, os.path.getsize(path)).model_dump()
    return {"id": name, "filename": name, "mime_type": "image/tiff", "path": path, "metadata": meta}


@pytest.fixture
async def queue(tmp_path: Path) -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    yield q
    await q.stop()


async def _collect(q: JobQueue, job_id: str, after: int = 0, timeout: float = 60) -> list[dict[str, Any]]:
    """Every event with seq > ``after`` until the terminal one, as a stream subscriber sees them."""
    subscription = q.subscribe(job_id, after)
    assert subscription is not None
    events, live = subscription
    try:
        while live is not None and (not events or events[-1]["type"] not in ("done", "failed", "cancelled")):
            events.append(await asyncio.wait_for(live.get(), timeout))
    finally:
        if live is not None:
            q.unsubscribe(job_id, live)
    return events


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """SSE frames back into ``{seq, type, data}``; comments are dropped."""
    events = []
    for frame in body.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if line and not line.startswith(":"))
        if fields:
            events.append({"seq": int(fields["id"]), "type": fields["event"], "data": json.loads(fields["data"])})
    return events


@pytest.fixture
def api(queue: JobQueue, monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    """HTTP client on the app (no lifespan) with ``queue`` as the job queue."""
    from server.app import app

    monkeypatch.setattr(jobs_mod, "_job_queue", queue)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


# --- Wire format ---------------------------------------------------------------


def test_sse_frame_format() -> None:
    event = {"seq": 7, "type": "step_started", "data": {"index": 2, "id": "algorithms/ndvi", "name": "NDVI · 1 June"}}
    assert sse_event_frame(event) == (
        b'id: 7\nevent: step_started\ndata: {"index":2,"id":"algorithms/ndvi","name":"NDVI \xc2\xb7 1 June"}\n\n'
    )
    assert KEEPALIVE_FRAME == b": keep-alive\n\n"


async def test_keepalive_then_terminal_ends_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_routes, "KEEPALIVE_S", 0.05)
    live: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    owner = MagicMock()
    backlog = [{"seq": 1, "type": "intent", "data": {"text": "I'll compute NDVI."}}]
    frames = job_routes._event_frames(owner, "job1", backlog, live)

    assert await anext(frames) == sse_event_frame(backlog[0])
    assert await anext(frames) == KEEPALIVE_FRAME
    done = {"seq": 2, "type": "done", "data": {"elapsed_s": 1.0, "summary": "1 step · 1 s"}}
    live.put_nowait(done)
    rest = [f async for f in frames if f != KEEPALIVE_FRAME]
    assert rest == [sse_event_frame(done)]
    owner.unsubscribe.assert_called_once_with("job1", live)


def test_done_summary_collapses_repeats() -> None:
    skeleton = [
        {"index": 1, "id": "input/file", "name": "Read a.tif", "kind": "input"},
        {"index": 2, "id": "algorithms/ndwi", "name": "NDWI", "kind": "algorithm"},
        {"index": 3, "id": "algorithms/ndwi", "name": "NDWI", "kind": "algorithm"},
        {"index": 4, "id": "algorithms/change-detection", "name": "Change detection", "kind": "algorithm"},
    ]
    assert done_summary(skeleton, 48.2) == "4 steps · 48 s · NDWI (×2) → Change detection"
    assert done_summary(skeleton[:2], 0.3) == "2 steps · <1 s · NDWI"
    assert done_summary(skeleton[:1], 72) == "1 step · 1.2 min"


# --- The NDVI job ----------------------------------------------------------------


async def test_ndvi_job_yields_full_ordered_sequence(queue: JobQueue) -> None:
    t0 = time.monotonic()
    job_id = await queue.submit_job("calculate NDVI", attachments=[_sample_attachment(SAMPLE_0601)])
    assert time.monotonic() - t0 < 1.0  # the intent is ready on submission
    first = queue.subscribe(job_id)
    assert first is not None
    assert [e["type"] for e in first[0]] == ["intent"]
    assert first[1] is not None
    queue.unsubscribe(job_id, first[1])

    events = await asyncio.gather(_collect(queue, job_id), queue.start())
    seq = events[0]

    assert [e["type"] for e in seq] == [
        "intent",
        "planning",
        "planning",
        *["step_started", "step_finished"] * 3,
        "done",
    ]
    assert [e["seq"] for e in seq] == list(range(1, len(seq) + 1))
    assert all(isinstance(e["data"]["t"], float) for e in seq)
    ts = [e["data"]["t"] for e in seq]
    assert ts == sorted(ts)

    intent, started, ready = seq[0]["data"], seq[1]["data"], seq[2]["data"]
    assert intent["text"] == "I'll compute NDVI for 1 June from the sample."
    assert started["phase"] == "started"
    assert ready["phase"] == "ready"
    assert ready["composed"] is False
    assert ready["steps"] == [
        {"index": 1, "id": "input/file", "name": "Read hlsl_20230601.tif", "kind": "input"},
        {"index": 2, "id": "algorithms/ndvi", "name": "NDVI", "kind": "algorithm"},
        {"index": 3, "id": "post-processing/agent-analysis", "name": "Interpretation", "kind": "postprocess"},
    ]
    for i, step_id in enumerate(NDVI_STEPS, start=1):
        s, f = seq[1 + 2 * i]["data"], seq[2 + 2 * i]["data"]
        assert (s["index"], s["id"], s["name"]) == (i, step_id, ready["steps"][i - 1]["name"])
        assert (f["index"], f["id"]) == (i, step_id)
        assert f["elapsed_s"] >= 0
    done = seq[-1]["data"]
    assert done["summary"].startswith("3 steps · ")
    assert done["summary"].endswith(" · NDVI")
    assert done["elapsed_s"] >= 0

    job = await queue.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 1.0


async def test_replay_after_completion_and_reconnect(queue: JobQueue, api: httpx.AsyncClient, tmp_path: Path) -> None:
    await queue.start()
    job_id = await queue.submit_job("calculate NDVI", attachments=[_sample_attachment(SAMPLE_0601)])
    live = await _collect(queue, job_id)
    assert live[-1]["type"] == "done"

    async with api:
        r = await api.get(f"/v1/jobs/{job_id}/events")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-accel-buffering"] == "no"
        assert r.headers["cache-control"] == "no-cache"
        assert _parse_sse(r.text) == live

        r = await api.get(f"/v1/jobs/{job_id}/events", headers={"Last-Event-ID": "4"})
        assert _parse_sse(r.text) == live[4:]
        r = await api.get(f"/v1/jobs/{job_id}/events", params={"after": 8}, headers={"Last-Event-ID": "2"})
        assert _parse_sse(r.text) == live[8:]
        r = await api.get(f"/v1/jobs/{job_id}/events", params={"after": len(live)})
        assert _parse_sse(r.text) == []

        r = await api.get(f"/v1/jobs/{job_id}")
        assert r.json()["events"] == live
        r = await api.get("/v1/jobs")
        assert all("events" not in j for j in r.json()["jobs"])

        r = await api.get("/v1/jobs/nope/events")
        assert r.status_code == 404

    # The log is persisted with the job: a new store on the same file (a server restart) replays it.
    reopened = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    try:
        assert await _collect(reopened, job_id) == live
    finally:
        await reopened.stop()


async def test_intent_for_change_detection_on_the_sample(queue: JobQueue) -> None:
    job_id = await queue.submit_job(
        "Compare water before and after the dam breach",
        attachments=[_sample_attachment(SAMPLE_0601), _sample_attachment(SAMPLE_0609)],
    )
    job = await queue.get_job(job_id)
    assert job is not None
    assert job.events[0]["data"]["text"] == (
        "I'll compute NDWI for 1 June and 9 June from the sample, then difference them to map where water changed."
    )


async def test_single_image_analysis_on_the_sample_pair_reads_9_june(queue: JobQueue) -> None:
    """D1: the Workspace sends both sample files; "Calculate EVI" runs EVI on the later one, and says so."""
    job_id = await queue.submit_job(
        "Calculate EVI", attachments=[_sample_attachment(SAMPLE_0601), _sample_attachment(SAMPLE_0609)]
    )
    events, _ = await asyncio.gather(_collect(queue, job_id), queue.start())

    assert events[0]["data"]["text"] == "I'll compute EVI for 9 June from the sample."
    assert events[-1]["type"] == "done", events[-1]
    assert events[-1]["data"]["summary"].endswith("EVI")
    job = await queue.get_job(job_id)
    assert job is not None and job.plan is not None
    assert [s.uses for s in job.plan.steps] == ["input/file", "algorithms/evi", "post-processing/agent-analysis"]
    assert job.plan.steps[0].binds == {"RasterPath": SAMPLE_0609}


async def test_a_date_in_the_prompt_picks_that_file_of_the_pair(queue: JobQueue) -> None:
    """ "Calculate EVI for 1 June" on the pair reads 1 June, and the intent sentence names it."""
    job_id = await queue.submit_job(
        "Calculate EVI for 1 June", attachments=[_sample_attachment(SAMPLE_0609), _sample_attachment(SAMPLE_0601)]
    )
    events, _ = await asyncio.gather(_collect(queue, job_id), queue.start())

    assert events[0]["data"]["text"] == "I'll compute EVI for 1 June from the sample."
    assert events[-1]["type"] == "done", events[-1]
    job = await queue.get_job(job_id)
    assert job is not None and job.plan is not None
    assert job.plan.steps[0].binds == {"RasterPath": SAMPLE_0601}


# --- Cancel ----------------------------------------------------------------------


async def test_cancel_mid_run_ends_with_cancelled(queue: JobQueue) -> None:
    step_running = threading.Event()
    release = threading.Event()
    executed: list[str] = []
    original = PipelineExecutor._execute_step

    def blocking_step(self: PipelineExecutor, step: PlanStep, item: Any) -> None:
        executed.append(step.uses)
        if step.uses == "algorithms/ndvi":
            step_running.set()
            assert release.wait(timeout=30)
        original(self, step, item)

    process_done = asyncio.Event()
    original_process = JobQueue._process_job

    async def tracked_process(self: JobQueue, job_id: str, worker_id: int) -> None:
        await original_process(self, job_id, worker_id)
        process_done.set()

    with (
        patch.object(PipelineExecutor, "_execute_step", blocking_step),
        patch.object(JobQueue, "_process_job", tracked_process),
    ):
        await queue.start()
        job_id = await queue.submit_job("calculate NDVI", attachments=[_sample_attachment(SAMPLE_0601)])
        collector = asyncio.create_task(_collect(queue, job_id))
        try:
            assert await asyncio.to_thread(step_running.wait, 30)
            assert await queue.cancel_job(job_id)
            seq = await asyncio.wait_for(collector, 10)
        finally:
            release.set()
        await asyncio.wait_for(process_done.wait(), timeout=30)

    types = [e["type"] for e in seq]
    assert types == ["intent", "planning", "planning", "step_started", "step_finished", "step_started", "cancelled"]
    assert seq[-1]["data"]["at_step"] == 2
    assert executed == ["input/file", "algorithms/ndvi"]  # step 3 never started

    # After the cancel, the step in progress finished, yet nothing was added to the log.
    job = await queue.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.CANCELLED
    assert [e["type"] for e in job.events] == types
    assert await _collect(queue, job_id) == seq


# --- Failures --------------------------------------------------------------------


def _plan_result(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "intent": "pipeline", "plan": {"flow": "test", "steps": list(steps)}}


async def test_band_missing_job_fails_with_code_and_fixes(queue: JobQueue) -> None:
    plan = _plan_result(
        {"uses": "input/file", "binds": {"RasterPath": SAMPLE_0601}},
        {"uses": "algorithms/band-math", "config": {"formula": "(B5 - B7) / (B5 + B7)", "name": "NBR"}},
    )
    with patch("server.jobs.orchestrate", return_value=plan):
        await queue.start()
        job_id = await queue.submit_job("Map burn severity", attachments=[_sample_attachment(SAMPLE_0601)])
        seq = await _collect(queue, job_id)

    assert [e["type"] for e in seq] == [
        "intent",
        "planning",
        "planning",
        "step_started",
        "step_finished",
        "step_started",
        "failed",
    ]
    assert seq[2]["data"]["steps"][1]["name"] == "NBR"
    failed = seq[-1]["data"]
    assert failed["code"] == "BAND_MISSING"
    assert failed["step_index"] == 2
    assert failed["message"].startswith("Band B7 is not in this file (B2–B6).")
    assert "Step algorithms/band-math failed" in failed["raw"]
    assert 1 <= len(failed["fixes"]) <= 3
    assert all(f["label"] and f["kind"] in _FIX_KINDS for f in failed["fixes"])


async def test_planning_failure_is_classified(queue: JobQueue) -> None:
    error = "Step algorithms/band-math: Band B7 is not in this file (B2–B6). Available: B2, B3, B4, B5, B6."
    with patch("server.jobs.orchestrate", return_value={"ok": False, "intent": "pipeline", "error": error}):
        await queue.start()
        job_id = await queue.submit_job("Map burn severity", attachments=[_sample_attachment(SAMPLE_0601)])
        seq = await _collect(queue, job_id)

    assert [e["type"] for e in seq] == ["intent", "planning", "failed"]
    failed = seq[-1]["data"]
    assert failed["code"] == "BAND_MISSING"
    assert "step_index" not in failed
    assert failed["message"].startswith("Band B7 is not in this file")


async def test_missing_input_job_fails_with_input_missing(queue: JobQueue) -> None:
    plan = _plan_result({"uses": "input/file"}, {"uses": "algorithms/ndvi"})
    with patch("server.jobs.orchestrate", return_value=plan):
        await queue.start()
        job_id = await queue.submit_job("calculate NDVI")
        seq = await _collect(queue, job_id)

    failed = seq[-1]["data"]
    assert seq[-1]["type"] == "failed"
    assert failed["code"] == "INPUT_MISSING"
    assert failed["step_index"] == 1
    assert failed["message"].startswith("I need a file")
    assert {f["kind"] for f in failed["fixes"]} == {"upload", "open_map"}


@pytest.mark.parametrize(
    ("error", "stage", "code"),
    [
        (ExecutionError("Step input/file failed: x"), "running", "INTERNAL"),
        (RuntimeError("No plan for this request"), "planning", "PLAN_FAILED"),
        ("EVI requires band 'blue', but the raster (4 bands) doesn't expose it.", "running", "BAND_MISSING"),
        ("Server restarted while job was in progress", "running", "INTERRUPTED"),
    ],
)
def test_classify_failure(error: BaseException | str, stage: str, code: str) -> None:
    data = classify_failure(error, stage)
    assert data["code"] == code
    assert data["message"]
    assert 1 <= len(data["fixes"]) <= 3


def test_classify_failure_by_cause_and_redacts_keys() -> None:
    cause = MissingInputError("input/file", "RasterPath", "I need a file. key sk-proj-abcdEFGH1234 rejected")
    wrapped = ExecutionError(f"Step input/file failed: {cause}")
    wrapped.__cause__ = cause
    data = classify_failure(wrapped, "running", step_index=1)
    assert data["code"] == "INPUT_MISSING"
    assert data["step_index"] == 1
    assert "sk-proj" not in data["raw"] and "sk-proj" not in data["message"]
    assert "***" in data["raw"]


# --- Persistence -----------------------------------------------------------------

_V051_SCHEMA = """
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY, status TEXT NOT NULL, prompt TEXT NOT NULL,
        attachments TEXT NOT NULL DEFAULT '[]', context TEXT, plan TEXT, result TEXT,
        progress REAL NOT NULL DEFAULT 0.0, error TEXT, created_at TEXT NOT NULL,
        started_at TEXT, completed_at TEXT
    )
"""


async def test_old_jobs_db_without_events_column_loads(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(db)
    conn.execute(_V051_SCHEMA)
    rows = [
        ("old1", "completed", "calculate ndvi", None, "2026-10-01T10:00:00", "2026-10-01T10:00:48"),
        ("old2", "running", "calculate ndvi", None, "2026-10-01T10:00:00", None),
    ]
    conn.executemany(
        "INSERT INTO jobs (id, status, prompt, error, started_at, completed_at, created_at) VALUES (?,?,?,?,?,?,?)",
        [(*r, "2026-10-01T10:00:00") for r in rows],
    )
    conn.commit()
    conn.close()

    store = SQLiteJobStore(db)  # adds the column
    old = store.get("old1")
    assert old is not None
    assert old.events == []
    old.prompt = "saved again"
    store.save(old)
    assert store.get("old1") is not None

    q = JobQueue(max_workers=1, store=store)  # marks old2 failed, as after a restart
    try:
        done = await _collect(q, "old1")
        assert [(e["seq"], e["type"]) for e in done] == [(1, "done")]
        assert done[0]["data"]["elapsed_s"] == 48.0
        failed = await _collect(q, "old2")
        assert [(e["seq"], e["type"]) for e in failed] == [(1, "failed")]
        assert failed[0]["data"]["code"] == "INTERRUPTED"
    finally:
        await q.stop()


def test_events_round_trip_through_sqlite(tmp_path: Path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    events = [{"seq": 1, "type": "intent", "data": {"text": "I'll compute NDVI.", "t": 1.0}}]
    store.save(Job(id="j1", status=JobStatus.PENDING, prompt="p", events=events))
    loaded = store.get("j1")
    assert loaded is not None
    assert loaded.events == events
    assert store.list_summaries()[0].events == []
