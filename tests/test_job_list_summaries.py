"""``GET /v1/jobs`` lists summaries without loading ``result`` (#67 bug 2, v0.5.1 item 6).

The list and ``cleanup_old_jobs`` read ``JobStore.list_summaries()``; only
``GET /v1/jobs/{id}`` loads and decodes the full record.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from dta.dti.schemas import ExecutionPlan, PlanStep
import server.jobs as jobs_mod
from server.jobs import Job, JobQueue, JobStatus, JobStore, MemoryJobStore, SQLiteJobStore

_PLAN = ExecutionPlan(
    steps=[PlanStep(uses="input/file"), PlanStep(uses="algorithms/ndvi", binds={"RasterPath": "input"})],
    outputs=["NDVIMap"],
)


class _UndecodableResult(dict[str, Any]):
    """A result that fails as soon as anything serialises it (MemoryJobStore analogue of a corrupt row)."""

    def items(self) -> Any:
        raise AssertionError("listing touched job.result")


def _jobs(n: int) -> list[Job]:
    base = datetime(2026, 10, 22, 0, 0, 0)
    statuses = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    return [
        Job(
            id=f"job{i:03d}",
            status=statuses[i % 3],
            prompt=f"prompt {i}",
            attachments=[{"name": f"f{i}.tif", "path": f"/x/f{i}.tif"}],
            context={"i": i},
            plan=_PLAN if i % 2 else None,
            result={"values": [0.5, 0.25], "statistics": {"mean": 0.5}},
            progress=1.0 if i % 4 else 0.75,
            error="boom" if statuses[i % 3] == JobStatus.FAILED else None,
            created_at=base + timedelta(seconds=i),
            started_at=base + timedelta(seconds=i, milliseconds=5),
            completed_at=base + timedelta(seconds=i + 1),
        )
        for i in range(n)
    ]


def _summary(job: Job) -> dict[str, Any]:
    """The v0.5.0 list item: ``to_dict()`` of the full job minus ``result`` (and the v0.6.0 ``events``)."""
    d = job.to_dict()
    del d["result"]
    del d["events"]
    return d


def _make_store(kind: str, tmp_path: Path) -> JobStore:
    return SQLiteJobStore(tmp_path / "jobs.db") if kind == "sqlite" else MemoryJobStore()


def _break_results(store: JobStore, tmp_path: Path) -> None:
    """Make every stored ``result`` fail if anything decodes or serialises it."""
    if isinstance(store, SQLiteJobStore):
        con = sqlite3.connect(tmp_path / "jobs.db")
        con.execute("UPDATE jobs SET result = '{not json'")
        con.commit()
        con.close()
    else:
        for job in store.list_all():
            job.result = _UndecodableResult(x=1)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient on the app without its lifespan (no workers); tests install the queue."""
    from server.app import app

    monkeypatch.setattr(jobs_mod, "_job_queue", None)
    yield TestClient(app)
    if jobs_mod._job_queue is not None:
        jobs_mod._job_queue._thread_pool.shutdown(wait=False)


@pytest.mark.parametrize("kind", ["sqlite", "memory"])
def test_list_never_decodes_result(kind: str, tmp_path: Path, client: TestClient) -> None:
    store = _make_store(kind, tmp_path)
    jobs = _jobs(12)
    for job in jobs:
        store.save(job)
    jobs_mod._job_queue = JobQueue(max_workers=1, store=store)
    expected = [_summary(j) for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)][:10]

    _break_results(store, tmp_path)
    r = client.get("/v1/jobs", params={"limit": 10})

    assert r.status_code == 200, r.text
    assert r.json() == {"jobs": expected, "total": 12, "limit": 10, "offset": 0}


@pytest.mark.parametrize("kind", ["sqlite", "memory"])
@pytest.mark.asyncio
async def test_cleanup_never_decodes_result(kind: str, tmp_path: Path) -> None:
    store = _make_store(kind, tmp_path)
    old, recent = _jobs(2)
    old.completed_at = datetime.now() - timedelta(hours=2)
    recent.completed_at = datetime.now()
    store.save(old)
    store.save(recent)
    queue = JobQueue(max_workers=1, retention_hours=1, store=store)

    _break_results(store, tmp_path)
    try:
        assert await queue.cleanup_old_jobs() == 1
    finally:
        queue._thread_pool.shutdown(wait=False)
    assert store.keys() == [recent.id]


def test_get_job_returns_full_result(tmp_path: Path, client: TestClient) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    job = _jobs(2)[1]
    store.save(job)
    jobs_mod._job_queue = JobQueue(max_workers=1, store=store)

    listed = client.get("/v1/jobs").json()["jobs"]
    record = client.get(f"/v1/jobs/{job.id}").json()

    assert "result" not in listed[0]
    assert record == job.to_dict()
    assert record["result"] == {"values": [0.5, 0.25], "statistics": {"mean": 0.5}}


@pytest.mark.asyncio
async def test_list_shows_live_state_of_in_flight_jobs(tmp_path: Path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    queue = JobQueue(max_workers=1, store=store)
    try:
        job_id = await queue.submit_job("running job")
        live = queue._jobs_cache[job_id]
        live.status = JobStatus.RUNNING
        live.progress = 0.4

        (listed,) = await queue.list_jobs()
    finally:
        queue._thread_pool.shutdown(wait=False)
    assert (listed.id, listed.status, listed.progress) == (job_id, JobStatus.RUNNING, 0.4)
    assert store.get(job_id).progress == 0.0  # type: ignore[union-attr]


@pytest.mark.slow  # seeds a 200 MB SQLite file (~4 s)
def test_list_200_jobs_with_1mb_results_under_200ms(tmp_path: Path, client: TestClient) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    for job in _jobs(200):
        store.save(job)
    big = json.dumps({"values": [0.123456789 + j * 1e-9 for j in range(80_000)]})
    assert len(big) > 1_000_000
    con = sqlite3.connect(tmp_path / "jobs.db")
    con.execute("UPDATE jobs SET result = ?", (big,))
    con.commit()
    con.close()
    jobs_mod._job_queue = JobQueue(max_workers=1, store=store)

    client.get("/v1/jobs", params={"limit": 20})  # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        r = client.get("/v1/jobs", params={"limit": 20})
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
        assert len(r.json()["jobs"]) == 20

    assert statistics.median(times) < 200, f"GET /v1/jobs?limit=20 took {times} ms"
