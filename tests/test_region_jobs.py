"""Region hand-off from the map (v0.6.0 story, #22, #18): ``POST /v1/jobs`` with ``region``.

A region job fetches its imagery from Earth Engine. Without credentials it fails at
planning with ``EE_NOT_CONFIGURED`` and two fixes, before any network call. With Earth
Engine "configured" (mocked) the plan fetches one composite, or two for a change
analysis, before the analysis. Every LLM provider points at a closed port.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
import functools
import os
from pathlib import Path
import socket
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import rasterio

from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.data_sources import gee_common, gee_region
from dta.dti.executor import ExecutionError
from dta.dti.raster import describe_raster
from dta.dti.schemas import ChatRequest, ExecutionPlan, PlanStep
from server import jobs as jobs_mod
from server.job_events import classify_failure
from server.jobs import Job, JobQueue, JobStatus, SQLiteJobStore
from server.routes.jobs import REGION_IGNORED_NOTE
from server.schemas import JobRegion
from server.utils import UPLOAD_DIR

# The sample as attachable files (tests/conftest.py sample_uploads)
pytestmark = pytest.mark.usefixtures("sample_uploads")
SAMPLE_0601 = str(UPLOAD_DIR / "hlsl_20230601.tif")
SAMPLE_0609 = str(UPLOAD_DIR / "hlsl_20230609.tif")
SAMPLE_BOUNDS = [32.388779, 46.455331, 32.691118, 46.663879]  # bounds_wgs84 of the Kakhovka crops
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")

REGION = {
    "bbox": SAMPLE_BOUNDS,
    "name": "Dnipro floodplain",
    "place": "Kherson",
    "dataset": "HLS L30",
    "start_date": "2023-06-01",
    "end_date": "2023-06-09",
    "cloud_max": 20,
}
CHANGE_PROMPT = "Compare water before and after the dam breach"
EE_MESSAGE = "Earth Engine isn't configured on this server, so I can't fetch imagery for this region."
EE_FIXES = [{"label": "Use my own image", "kind": "upload"}, {"label": "Try the Kakhovka sample", "kind": "sample"}]
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
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


@pytest.fixture(autouse=True)
def _ee_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Earth Engine project or key file, and not initialised: ``is_configured()`` is False."""
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setattr(gee_common, "_ee_initialized", False)
    assert gee_common.is_configured() is False


@pytest.fixture
def ee_configured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Earth Engine "configured" and its fetch mocked (the sample crop of the window's date); the fetch configs."""
    calls: list[dict[str, Any]] = []

    def fake_fetch(config: dict[str, Any]) -> str:
        calls.append(config)
        return SAMPLE_0601 if config["start_date"] <= "2023-06-01" else SAMPLE_0609

    monkeypatch.setattr(gee_common, "is_configured", lambda: True)
    monkeypatch.setattr(gee_common, "initialize_gee", lambda: True)
    monkeypatch.setattr(gee_region, "fetch", fake_fetch)
    return calls


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


@functools.cache
def _sample_attachment(path: str) -> dict[str, Any]:
    name = Path(path).name
    with rasterio.open(path) as src:
        meta = describe_raster(src, name, os.path.getsize(path)).model_dump()
    return {"id": name, "filename": name, "mime_type": "image/tiff", "path": path, "metadata": meta}


def _echo(**overrides: Any) -> dict[str, Any]:
    return JobRegion.model_validate({**REGION, **overrides}).echo()  # type: ignore[no-any-return]


# --- Without Earth Engine: accepted, then a clean failure at planning ---------------------------


async def test_region_job_without_earth_engine_fails_at_planning_without_network(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ee

    opened: list[str] = []

    def no_network(*args: object, **kwargs: object) -> None:
        opened.append(repr(args[1:]))
        raise AssertionError("a region job opened a network connection")

    def no_initialize(*args: object, **kwargs: object) -> None:
        opened.append("ee.Initialize")
        raise AssertionError("a region job called ee.Initialize")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(ee, "Initialize", no_initialize)
    with patch.object(jobs_mod, "orchestrate", side_effect=AssertionError("planned a region job without EE")) as plan:
        async with _client() as client:
            r = await client.post("/v1/jobs", json={"prompt": CHANGE_PROMPT, "region": REGION})
            assert r.status_code == 202, r.text
            submitted = r.json()
            assert submitted["status"] == "pending"
            assert [e["type"] for e in submitted["events"]] == ["intent"]
            assert submitted["region"]["line"] == (
                "Region · Dnipro floodplain · Kherson · 537 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud"
            )

            seq = await _events_until(queue, submitted["id"], {"failed", "done", "proposal"})
            record = (await client.get(f"/v1/jobs/{submitted['id']}")).json()

    assert [e["type"] for e in seq] == ["intent", "planning", "failed"]  # no ready skeleton, no step
    assert seq[0]["data"]["text"] == (
        "I'll compute NDWI for 1 June and 9 June from Dnipro floodplain, then difference them to map where water "
        "changed."
    )
    assert seq[1]["data"]["phase"] == "started"
    failed = {k: v for k, v in seq[2]["data"].items() if k != "t"}
    assert failed == {"code": "EE_NOT_CONFIGURED", "message": EE_MESSAGE, "fixes": EE_FIXES, "raw": EE_MESSAGE}
    assert opened == []
    plan.assert_not_called()
    assert record["status"] == "failed"
    assert record["plan"] is None
    assert record["region"] == submitted["region"]


async def test_region_job_counts_like_an_analysis_and_is_owner_scoped(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    async with _client() as owner, _client() as stranger:
        r = await owner.post("/v1/jobs", json={"prompt": CHANGE_PROMPT, "region": REGION})
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        assert r.json()["owner"]
        await _events_until(queue, job_id, {"failed", "done"})
        assert (await owner.get("/v1/session")).json()["analyses"]["used"] == 1
        assert (await stranger.get(f"/v1/jobs/{job_id}")).status_code == 404
        assert (await stranger.get(f"/v1/jobs/{job_id}/events")).status_code == 404


def test_fetch_step_without_credentials_never_initialises_earth_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planned fetch whose server lost its credentials (a confirmed proposal) fails the same way."""
    monkeypatch.setattr(gee_common, "initialize_gee", lambda: pytest.fail("ee.Initialize attempted"))
    config = gee_region.fetch_config(_echo(), "2023-06-01", "2023-06-09")
    with pytest.raises(gee_region.EarthEngineNotConfiguredError) as raised:
        gee_region.run(config)
    wrapped = ExecutionError(f"Step input/earth-engine failed: {raised.value}")
    wrapped.__cause__ = raised.value
    failed = classify_failure(wrapped, "running", step_index=1)
    assert (failed["code"], failed["message"], failed["fixes"]) == ("EE_NOT_CONFIGURED", EE_MESSAGE, EE_FIXES)


# --- The echo: area and the chip line ------------------------------------------------------------


def test_region_echo_has_area_bounds_and_line() -> None:
    echo = _echo()
    # The crops are 768 × 768 px at 30 m (530.8 km² in UTM); their WGS84 bounds enclose a little more.
    assert 530 < echo["area_km2"] < 545
    assert echo["bounds"] == SAMPLE_BOUNDS
    assert echo["bands"] == BANDS  # the dataset's optical bands by default
    assert {k: echo[k] for k in ("geometry", "bbox", "dataset", "start_date", "end_date", "cloud_max")} == {
        "geometry": None,
        "bbox": SAMPLE_BOUNDS,
        "dataset": "HLS L30",
        "start_date": "2023-06-01",
        "end_date": "2023-06-09",
        "cloud_max": 20,
    }
    assert "note" not in echo

    w, s, e, n = SAMPLE_BOUNDS
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    polygon = _echo(bbox=None, geometry={"type": "Polygon", "coordinates": [ring]}, name=None, place=None)
    assert polygon["area_km2"] == echo["area_km2"]
    assert polygon["line"] == "Region · 537 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud"

    small = _echo(bbox=[32.5, 46.5, 32.52, 46.52], place=None, dataset="Sentinel-2", end_date="2024-01-05")
    assert small["line"] == "Region · Dnipro floodplain · 3.4 km² · Sentinel-2 · 2023-06-01 → 2024-01-05 · ≤20 % cloud"
    assert _echo(bands=["SR_B5", "b6", "B5"])["bands"] == ["B5", "B6"]


_TRIANGLE = [[[32.4, 46.5], [32.5, 46.5], [32.5, 46.6], [32.4, 46.5]]]
_BOW_TIE = [[[32.4, 46.5], [32.5, 46.6], [32.5, 46.5], [32.4, 46.6], [32.4, 46.5]]]


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"geometry": {"type": "Polygon", "coordinates": _TRIANGLE}}, id="geometry-and-bbox"),
        pytest.param({"bbox": None}, id="neither"),
        pytest.param({"bbox": [32.7, 46.4, 32.3, 46.6]}, id="west-after-east"),
        pytest.param({"bbox": [32.3, 46.4, 32.7]}, id="bbox-3-values"),
        pytest.param({"bbox": [32.3, 89.9, 32.7, 91.0]}, id="latitude-over-90"),
        pytest.param({"bbox": [30.0, 45.0, 32.0, 47.0]}, id="over-2500-km2"),  # ~34,000 km²
        pytest.param({"start_date": "2023-06-10"}, id="start-after-end"),
        pytest.param({"start_date": "2023-13-01"}, id="bad-date"),
        pytest.param({"dataset": "Planet"}, id="unknown-dataset"),
        pytest.param({"bands": ["B8"]}, id="band-not-in-dataset"),
        pytest.param({"bands": []}, id="no-bands"),
        pytest.param({"cloud_max": 120}, id="cloud-over-100"),
        pytest.param({"name": "x" * 81}, id="long-name"),
        pytest.param({"unknown": 1}, id="unknown-field"),
        pytest.param({"bbox": None, "geometry": {"type": "Point", "coordinates": [32.5, 46.5]}}, id="point"),
        pytest.param({"bbox": None, "geometry": {"type": "Polygon", "coordinates": _BOW_TIE}}, id="self-intersecting"),
        pytest.param(
            {"bbox": None, "geometry": {"type": "Polygon", "coordinates": [[[200, 46.5], [201, 46.5], [201, 46.6]]]}},
            id="longitude-over-180",
        ),
        pytest.param({"bbox": None, "geometry": {"type": "Polygon", "coordinates": "nope"}}, id="bad-coordinates"),
    ],
)
async def test_invalid_regions_are_422(queue: JobQueue, change: dict[str, Any]) -> None:
    async with _client() as client:
        r = await client.post("/v1/jobs", json={"prompt": CHANGE_PROMPT, "region": {**REGION, **change}})
        assert r.status_code == 422, r.text
        assert (await client.get("/v1/jobs")).json()["total"] == 0


def test_change_windows() -> None:
    assert gee_region.change_windows("2023-06-01", "2023-06-09") == (
        ("2023-05-29", "2023-06-04"),
        ("2023-06-06", "2023-06-12"),
    )
    assert gee_region.change_windows("2023-06-01", "2023-07-01") == (
        ("2023-05-24", "2023-06-09"),
        ("2023-06-23", "2023-07-09"),
    )
    assert gee_region.change_windows("2023-06-01", "2023-06-02") == (
        ("2023-06-01", "2023-06-01"),
        ("2023-06-02", "2023-06-02"),
    )


# --- With Earth Engine: the plan fetches before the analysis -------------------------------------


def _fetch(start: str, end: str, label: str) -> dict[str, Any]:
    return {
        "name": f"Fetch HLS L30 · {label}",
        "dataset": "HLS L30",
        "bbox": SAMPLE_BOUNDS,
        "start_date": start,
        "end_date": end,
        "bands": BANDS,
        "cloud_max": 20,
    }


BEFORE = _fetch("2023-05-29", "2023-06-04", "2023-05-29 → 06-04")
AFTER = _fetch("2023-06-06", "2023-06-12", "2023-06-06 → 06-12")


async def test_change_region_job_fetches_before_and_after_then_runs(
    queue: JobQueue, ee_configured: list[dict[str, Any]]
) -> None:
    async with _client() as client:
        r = await client.post("/v1/jobs", json={"prompt": CHANGE_PROMPT, "region": REGION})
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]
        seq = await _events_until(queue, job_id, {"failed", "done", "proposal"})
        job = (await client.get(f"/v1/jobs/{job_id}")).json()

    assert seq[-1]["type"] == "done", seq[-1]
    ready = next(e["data"] for e in seq if e["type"] == "planning" and e["data"]["phase"] == "ready")
    assert [(s["id"], s["name"], s["kind"]) for s in ready["steps"]] == [
        ("input/earth-engine-before", "Fetch HLS L30 · 2023-05-29 → 06-04", "input"),
        ("input/earth-engine-after", "Fetch HLS L30 · 2023-06-06 → 06-12", "input"),
        ("algorithms/change-detection", "Change Detection · NDWI", "algorithm"),
        (POST, "Interpretation", "postprocess"),
    ]
    assert job["plan"]["steps"] == [
        {"uses": "input/earth-engine-before", "binds": {}, "config": BEFORE},
        {"uses": "input/earth-engine-after", "binds": {}, "config": AFTER},
        {"uses": "algorithms/change-detection", "binds": {"IndexType": "ndwi"}, "config": {}},
        {"uses": POST, "binds": {}, "config": {}},
    ]
    assert ee_configured == [BEFORE, AFTER]
    assert "ChangeMap" in job["result"]["execution"]["artifacts"]


WHOLE_RANGE = _fetch("2023-06-01", "2023-06-09", "2023-06-01 → 06-09")


def test_single_date_region_plan_fetches_one_composite_over_the_range(ee_configured: list[dict[str, Any]]) -> None:
    result = orchestrate(ChatRequest(prompt="Calculate NDVI for this region", metadata={"region": _echo()}))
    assert result["ok"], result
    assert result["plan"]["steps"] == [
        {"uses": "input/earth-engine", "binds": {}, "config": WHOLE_RANGE},
        {"uses": "algorithms/ndvi", "binds": {}, "config": {}},
        {"uses": POST, "binds": {}, "config": {}},
    ]
    assert ee_configured == []  # planning fetches nothing


async def test_attachments_win_over_the_region(queue: JobQueue) -> None:
    """Earth Engine is not configured, yet the job runs: it reads the file and notes that the region is unused."""
    async with _client() as client:
        r = await client.post(
            "/v1/jobs",
            json={"prompt": "calculate NDVI", "attachments": [_sample_attachment(SAMPLE_0601)], "region": REGION},
        )
        assert r.status_code == 202, r.text
        submitted = r.json()
        seq = await _events_until(queue, submitted["id"], {"failed", "done", "proposal"})

    assert submitted["region"]["note"] == REGION_IGNORED_NOTE
    assert seq[-1]["type"] == "done", seq[-1]
    ready = next(e["data"] for e in seq if e["type"] == "planning" and e["data"]["phase"] == "ready")
    assert [s["id"] for s in ready["steps"]] == ["input/file", "algorithms/ndvi", POST]


def test_an_edited_plan_fetches_only_the_jobs_region(queue: JobQueue) -> None:
    """Confirming an edited plan rewrites its fetch configs from the job's region, whatever the edit says."""
    job = Job(id="r1", status=JobStatus.PROPOSAL, prompt=CHANGE_PROMPT, region=_echo())
    elsewhere = {**BEFORE, "bbox": [0.0, 0.0, 10.0, 10.0], "dataset": "MODIS", "bands": ["sur_refl_b01"]}
    edited = ExecutionPlan(
        steps=[
            PlanStep(uses="input/earth-engine-before", config=elsewhere),
            PlanStep(uses="input/file-after", binds={"RasterPathAfter": "/etc/passwd"}),
            PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndwi"}),
        ]
    )
    plan = queue._rebind_and_validate(edited, job)
    assert [(s.uses, s.binds, s.config) for s in plan.steps[:2]] == [
        ("input/earth-engine-before", {}, BEFORE),
        ("input/earth-engine-after", {}, AFTER),
    ]
