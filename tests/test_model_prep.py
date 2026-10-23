"""Model preparation as a job step (v0.6.0 story "Model preparation as a step").

A missing model becomes a "Preparing model" step for roles that may download it
(``MODEL_DOWNLOAD_ON_DEMAND``, default ``members``); other roles get ``MODEL_GATED``
(or ``MODEL_NOT_INSTALLED`` when nobody may download) before anything is fetched.
The model manager is a fake: no download and no network. The model step itself is stubbed.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import dta.config
from dta.config import ROOT_DIR
import dta.dti.models
from dta.dti.models import AVAILABLE_MODELS, DownloadProgress, ModelManager, ModelStatus
from dta.dti.schemas import ExecutionPlan, PlanStep, RegistryItem
from server import jobs as jobs_mod
from server.job_events import classify_failure, step_skeleton
from server.jobs import JobQueue, MemoryJobStore

PRITHVI = "prithvi-eo-v1-100m"
PRITHVI_STEP = "models/prithvi-reconstruction"
PRITHVI_BYTES = AVAILABLE_MODELS[PRITHVI].size_mb * 1024 * 1024
SAMPLE = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_FIX_KINDS = {"prompt", "retry", "upload", "open_map"}


class FakeModelManager:
    """``is_model_available`` / ``get_model_path`` / ``download`` like :class:`ModelManager`, without I/O."""

    def __init__(self, cache_dir: Path, installed: tuple[str, ...] = ()) -> None:
        self.cache_dir = cache_dir
        self.installed = set(installed)
        self.downloads: list[str] = []

    def is_model_available(self, model_id: str) -> bool:
        return model_id in self.installed

    def get_model_path(self, model_id: str) -> Path:
        return self.cache_dir / model_id

    def download(self, model_id: str, progress_callback: Any = None, is_cancelled: Any = None) -> DownloadProgress:
        self.downloads.append(model_id)
        total = AVAILABLE_MODELS[model_id].size_mb * 1024 * 1024
        for fraction in (0.25, 0.75):
            progress_callback(
                DownloadProgress(
                    model_id=model_id,
                    status=ModelStatus.DOWNLOADING,
                    progress=fraction,
                    downloaded_bytes=int(total * fraction),
                    total_bytes=total,
                    speed_bps=10_000_000,
                )
            )
        self.installed.add(model_id)
        return DownloadProgress(
            model_id=model_id, status=ModelStatus.AVAILABLE, progress=1.0, downloaded_bytes=total, total_bytes=total
        )


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeModelManager]:
    """The fake as the global model manager. ``DT4LC_MODEL_CACHE`` is read once at import, so its
    resolved value (``dta.config.MODELS_PATH``) points under ``tmp_path`` too, in case anything
    builds a real :class:`ModelManager`."""
    cache = tmp_path / "models"
    monkeypatch.setattr(dta.config, "MODELS_PATH", cache)
    fake = FakeModelManager(cache)
    monkeypatch.setattr(dta.dti.models, "get_model_manager", lambda: fake)
    yield fake


@pytest.fixture(autouse=True)
def _stub_model_steps() -> Iterator[None]:
    """Model steps "run" instantly and output a placeholder; other python steps run for real."""
    from dta.dti.executor import PipelineExecutor

    real = PipelineExecutor._run_python

    def run(self: PipelineExecutor, step: PlanStep, item: RegistryItem) -> None:
        if item.kind == "model":
            self.artifacts.update({out: {"stub": step.uses} for out in item.outputs})
        else:
            real(self, step, item)

    with patch.object(PipelineExecutor, "_run_python", run):
        yield


@pytest.fixture
async def queue() -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    await q.start()
    yield q
    await q.stop()


def _prithvi_plan(raster: str) -> dict[str, Any]:
    steps = [{"uses": "input/file", "binds": {"RasterPath": raster}}, {"uses": PRITHVI_STEP}]
    return {"ok": True, "intent": "pipeline", "plan": {"flow": "test", "steps": steps}}


async def _run(q: JobQueue, raster: str, role: str | None, timeout: float = 30) -> list[dict[str, Any]]:
    """Submit a Prithvi job for ``role`` and return its event log once it ends."""
    with patch("server.jobs.orchestrate", return_value=_prithvi_plan(raster)):
        job_id = await q.submit_job("Fill the cloud gaps", owner_role=role)
        return await _wait(q, job_id, timeout)


async def _wait(q: JobQueue, job_id: str, timeout: float = 30) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        job = await q.get_job(job_id)
        assert job is not None
        if job.events and job.events[-1]["type"] in ("done", "failed", "cancelled"):
            return job.events
        assert asyncio.get_running_loop().time() < deadline, f"job did not end: {job.events}"
        await asyncio.sleep(0.05)


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [e["type"] for e in events]


def _assert_fixes(fixes: list[dict[str, Any]]) -> None:
    assert 1 <= len(fixes) <= 3
    assert all(f["label"] and f["kind"] in _FIX_KINDS for f in fixes)


# --- Download allowed ------------------------------------------------------------


async def test_member_with_missing_weights_gets_a_preparing_model_step(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str
) -> None:
    events = await _run(queue, synthetic_raster_path, "member")

    assert _types(events) == [
        "intent",
        "planning",
        "planning",
        "step_started",
        "step_finished",
        "step_started",
        "step_progress",
        "step_progress",
        "step_progress",
        "step_finished",
        "step_started",
        "step_finished",
        "done",
    ]
    ready = events[2]["data"]
    assert [(r["index"], r["id"], r["kind"]) for r in ready["steps"]] == [
        (1, "input/file", "input"),
        (2, f"prepare/{PRITHVI}", "prepare"),
        (3, PRITHVI_STEP, "model"),
    ]
    assert ready["steps"][1]["name"] == "Preparing model · Prithvi EO v1 (100M)"
    assert ready["steps"][1]["model_id"] == PRITHVI

    started = events[5]["data"]
    assert started["index"] == 2 and started["id"] == f"prepare/{PRITHVI}"
    assert started["name"].startswith("Preparing model")
    progress = [e["data"] for e in events[6:9]]
    assert all(p["index"] == 2 for p in progress)
    assert [p["bytes"] for p in progress] == [0, PRITHVI_BYTES // 4, int(PRITHVI_BYTES * 0.75)]
    assert all(p["total_bytes"] == PRITHVI_BYTES for p in progress)
    assert [p["fraction"] for p in progress] == [0.0, 0.25, 0.75]
    assert all(isinstance(p["eta_s"], int) and p["eta_s"] > 0 for p in progress)
    assert progress[0]["message"] == "Downloading Prithvi EO v1 (100M) · 450 MB"
    assert events[9]["data"]["index"] == 2 and events[9]["data"]["id"] == f"prepare/{PRITHVI}"
    assert events[10]["data"] == {**events[10]["data"], "index": 3, "id": PRITHVI_STEP}
    assert events[11]["data"]["index"] == 3
    assert manager.downloads == [PRITHVI]


async def test_installed_weights_need_no_preparation(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str
) -> None:
    manager.installed.add(PRITHVI)
    events = await _run(queue, synthetic_raster_path, "guest")

    assert events[-1]["type"] == "done"
    assert [r["kind"] for r in events[2]["data"]["steps"]] == ["input", "model"]
    assert not any(e["data"].get("name", "").startswith("Preparing model") for e in events)
    assert "step_progress" not in _types(events)
    assert manager.downloads == []


async def test_policy_all_lets_a_guest_download(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "all")
    events = await _run(queue, synthetic_raster_path, "guest")

    assert events[-1]["type"] == "done"
    assert any(e["type"] == "step_started" and e["data"]["id"] == f"prepare/{PRITHVI}" for e in events)
    assert manager.downloads == [PRITHVI]


# --- Download refused ------------------------------------------------------------


async def test_guest_with_missing_weights_is_gated_before_any_download(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str
) -> None:
    events = await _run(queue, synthetic_raster_path, "guest")

    assert _types(events) == ["intent", "planning", "failed"]
    failed = events[-1]["data"]
    assert failed["code"] == "MODEL_GATED"
    assert failed["message"] == (
        "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator."
    )
    assert [f["label"] for f in failed["fixes"]] == ["See what I can do", "Use a built-in analysis instead"]
    _assert_fixes(failed["fixes"])
    assert "step_index" not in failed
    assert manager.downloads == []
    # The queue's ModelNotInstalledError branch ran: the job result names the model.
    job = await queue.get_job((await queue.list_jobs())[0].id)
    assert job is not None and job.result is not None and job.result["model_required"]["id"] == PRITHVI


async def test_guest_job_through_the_api_is_gated(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.app import app

    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setattr(jobs_mod, "_job_queue", queue)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with patch("server.jobs.orchestrate", return_value=_prithvi_plan(synthetic_raster_path)):
            response = await client.post("/v1/jobs", json={"prompt": "Fill the cloud gaps"})
            assert response.status_code == 202
            events = await _wait(queue, response.json()["id"])

    assert events[-1]["data"]["code"] == "MODEL_GATED"
    assert manager.downloads == []


async def test_members_are_gated_when_only_operators_may_download(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "operators")
    events = await _run(queue, synthetic_raster_path, "member")

    assert events[-1]["data"]["code"] == "MODEL_GATED"
    assert "isn't available for members" in events[-1]["data"]["message"]
    assert manager.downloads == []


async def test_policy_none_fails_with_model_not_installed(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "none")
    events = await _run(queue, synthetic_raster_path, "operator")

    assert _types(events) == ["intent", "planning", "failed"]
    failed = events[-1]["data"]
    assert failed["code"] == "MODEL_NOT_INSTALLED"
    assert (
        failed["message"]
        == "Prithvi EO v1 (100M) isn't installed on this server (450 MB) — ask an operator to install it."
    )
    _assert_fixes(failed["fixes"])
    assert manager.downloads == []


# --- Cheap preconditions first -----------------------------------------------------


async def test_prithvi_on_the_sample_fails_on_bands_before_any_download(
    queue: JobQueue, manager: FakeModelManager
) -> None:
    events = await _run(queue, SAMPLE, "member")

    assert events[-1]["type"] == "failed"
    assert "step_started" not in _types(events)
    failed = events[-1]["data"]
    assert failed["code"] == "BAND_MISSING"
    assert failed["message"] == "Prithvi needs six HLS bands (B2–B7); this file has five (B2–B6, no B7)."
    assert failed["fixes"] == [
        {"label": "Use my own image", "kind": "upload"},
        {"label": "Run NDWI change instead", "kind": "prompt", "prompt": "Map NDWI change between the two dates"},
    ]
    assert manager.downloads == []


# --- Executor, skeleton and failure table ----------------------------------------------


def test_executor_without_downloads_raises_model_not_installed_unwrapped(
    manager: FakeModelManager, synthetic_raster_path: str
) -> None:
    from dta.dti.executor import ModelNotInstalledError, PipelineExecutor

    plan = ExecutionPlan(**_prithvi_plan(synthetic_raster_path)["plan"])
    with pytest.raises(ModelNotInstalledError) as info:
        PipelineExecutor().execute(plan)
    assert type(info.value) is ModelNotInstalledError
    assert info.value.model_id == PRITHVI
    assert manager.downloads == []


def test_model_not_installed_inside_a_step_keeps_its_type(manager: FakeModelManager) -> None:
    """The step-level check (weights deleted after the preflight) is not wrapped in ExecutionError."""
    from dta.dti.executor import ModelNotInstalledError, PipelineExecutor

    plan = ExecutionPlan(
        steps=[PlanStep(uses="input/file", binds={"RasterPath": "/x.tif"}), PlanStep(uses=PRITHVI_STEP)]
    )
    executor = PipelineExecutor()
    with patch.object(PipelineExecutor, "models_to_prepare", return_value={}):
        with pytest.raises(ModelNotInstalledError) as info:
            executor.execute(plan)
    assert type(info.value) is ModelNotInstalledError


def test_skeleton_inserts_one_preparation_row_per_model() -> None:
    from dta.dti.registry import load_registry

    plan = ExecutionPlan(steps=[PlanStep(uses="input/file"), PlanStep(uses=PRITHVI_STEP), PlanStep(uses=PRITHVI_STEP)])
    rows = step_skeleton(plan, load_registry(), prepare={2: PRITHVI})
    assert [(r["index"], r["kind"]) for r in rows] == [(1, "input"), (2, "prepare"), (3, "model"), (4, "model")]
    assert [r["index"] for r in step_skeleton(plan, load_registry())] == [1, 2, 3]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (
            "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator.",
            "MODEL_GATED",
        ),
        (
            "Prithvi EO v1 (100M) isn't installed on this server (450 MB) — ask an operator to install it.",
            "MODEL_NOT_INSTALLED",
        ),
        ("Prithvi needs six HLS bands (B2–B7); this file has five (B2–B6, no B7).", "BAND_MISSING"),
    ],
)
def test_stored_error_text_reclassifies(text: str, code: str) -> None:
    """A job reloaded without its terminal event is classified from ``job.error`` alone."""
    assert classify_failure(text)["code"] == code


# --- Model manager ---------------------------------------------------------------------


def test_real_manager_download_reports_bytes_and_finishes(tmp_path: Path) -> None:
    manager = ModelManager(cache_dir=tmp_path / "models")

    def fake_hf_hub_download(repo_id: str, filename: str, local_dir: Path, **kwargs: object) -> str:
        dest = Path(local_dir) / filename
        dest.write_bytes(b"w" * 1000)
        return str(dest)

    seen: list[int] = []
    with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
        result = manager.download(PRITHVI, progress_callback=lambda p: seen.append(p.downloaded_bytes), poll_s=0.01)

    assert result.status == ModelStatus.AVAILABLE
    assert seen and seen[-1] == result.total_bytes
    assert manager.is_model_available(PRITHVI)


def test_real_manager_download_failure_is_returned(tmp_path: Path) -> None:
    manager = ModelManager(cache_dir=tmp_path / "models")
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        result = manager.download(PRITHVI, poll_s=0.01)
    assert result.status == ModelStatus.FAILED
    assert result.error == "offline"


def test_model_states(tmp_path: Path) -> None:
    manager = ModelManager(cache_dir=tmp_path / "models")
    states = {s["id"]: s for s in manager.model_states()}

    assert set(states) == set(AVAILABLE_MODELS)
    prithvi = states[PRITHVI]
    assert prithvi == {
        "id": PRITHVI,
        "display_name": "Prithvi EO v1 (100M)",
        "installed": False,
        "size_bytes": PRITHVI_BYTES,
        "est_download_s": 75,
        "licence": "Apache-2.0",
        "licence_url": AVAILABLE_MODELS[PRITHVI].license_url,
        "steps": [PRITHVI_STEP],
    }


async def test_cancel_while_preparing_ends_with_cancelled(
    queue: JobQueue, manager: FakeModelManager, synthetic_raster_path: str
) -> None:
    def slow_download(model_id: str, progress_callback: Any = None, is_cancelled: Any = None) -> DownloadProgress:
        manager.downloads.append(model_id)
        while not is_cancelled():
            time.sleep(0.01)
        return DownloadProgress(model_id=model_id, status=ModelStatus.DOWNLOADING)

    manager.download = slow_download  # type: ignore[method-assign]
    with patch("server.jobs.orchestrate", return_value=_prithvi_plan(synthetic_raster_path)):
        job_id = await queue.submit_job("Fill the cloud gaps", owner_role="member")
        for _ in range(500):
            job = await queue.get_job(job_id)
            assert job is not None
            if any(e["type"] == "step_started" and e["data"]["index"] == 2 for e in job.events):
                break
            await asyncio.sleep(0.01)
        assert await queue.cancel_job(job_id)
        events = await _wait(queue, job_id)

    assert events[-1] == {**events[-1], "type": "cancelled", "data": {**events[-1]["data"], "at_step": 2}}
    assert not any(e["type"] == "step_started" and e["data"]["index"] == 3 for e in events)
    await asyncio.sleep(0.2)  # the worker thread notices the cancel and stops
    assert PRITHVI not in manager.installed
