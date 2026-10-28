"""``DELETE /v1/jobs/{id}`` (v0.6.0 Library): owner-scoped, removes the row and ``OUTPUTS_PATH/<job_id>/``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from dta.dti import artifact_files
from server import jobs as jobs_mod
from server import sessions
from server.jobs import Job, JobQueue, JobStatus, SQLiteJobStore


@pytest.fixture
async def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[JobQueue]:
    """A queue without workers: tests put jobs in the states they need."""
    monkeypatch.setenv("DEV_ROLE", "guest")
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    yield q
    q._thread_pool.shutdown(wait=False)


def _client() -> httpx.AsyncClient:
    from server.app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _owner(client: httpx.AsyncClient) -> str:
    await client.get("/v1/session")
    session = sessions.decode_cookie(client.cookies[sessions.COOKIE_NAME])
    assert session is not None
    return session.id


def _stored(q: JobQueue, job_id: str, status: JobStatus, owner: str) -> Path:
    """A stored job with one output file; returns its output directory."""
    q._store.save(Job(id=job_id, status=status, prompt="calculate ndvi", owner=owner, owner_role="guest"))
    out = artifact_files.output_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "1_NDVIMap_ndvi.tif").write_bytes(b"tif")
    return out


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED])
async def test_delete_a_finished_job_removes_row_and_outputs(queue: JobQueue, status: JobStatus) -> None:
    async with _client() as client:
        owner = await _owner(client)
        out = _stored(queue, f"del-{status.value}", status, owner)

        r = await client.delete(f"/v1/jobs/del-{status.value}")

        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "id": f"del-{status.value}"}
        assert queue._store.get(f"del-{status.value}") is None
        assert not out.exists()
        assert (await client.get(f"/v1/jobs/del-{status.value}")).status_code == 404
        assert (await client.get("/v1/jobs")).json()["jobs"] == []
        assert (await client.delete(f"/v1/jobs/del-{status.value}")).status_code == 404


async def test_a_running_or_pending_job_must_be_cancelled_first(queue: JobQueue) -> None:
    async with _client() as client:
        owner = await _owner(client)
        out = _stored(queue, "busy", JobStatus.RUNNING, owner)
        pending_id = await queue.submit_job("calculate ndvi", owner=owner, owner_role="guest")

        running = await client.delete("/v1/jobs/busy")
        pending = await client.delete(f"/v1/jobs/{pending_id}")

        assert running.status_code == 409
        assert running.json()["error"]["message"] == "Job busy is running: cancel it first"
        assert pending.status_code == 409
        assert queue._store.get("busy") is not None and out.is_dir()
        assert (await queue.get_job(pending_id)) is not None

        assert (await client.post(f"/v1/jobs/{pending_id}/cancel")).status_code == 200
        assert (await client.delete(f"/v1/jobs/{pending_id}")).status_code == 200


async def test_a_proposal_may_be_deleted_and_its_stream_ends(queue: JobQueue) -> None:
    async with _client() as client:
        owner = await _owner(client)
        _stored(queue, "prop", JobStatus.PROPOSAL, owner)
        subscription = queue.subscribe("prop")
        assert subscription is not None and subscription[1] is not None
        live = subscription[1]

        assert (await client.delete("/v1/jobs/prop")).status_code == 200

        event = await asyncio.wait_for(live.get(), 5)
        assert (event["type"], event["data"]["reason"]) == ("cancelled", "deleted")
        assert queue._store.get("prop") is None


async def test_another_owners_job_is_404(queue: JobQueue) -> None:
    async with _client() as alice, _client() as bob:
        out = _stored(queue, "alices", JobStatus.COMPLETED, await _owner(alice))
        await _owner(bob)

        assert (await bob.delete("/v1/jobs/alices")).status_code == 404
        assert (await bob.delete("/v1/jobs/nojob")).status_code == 404

        assert queue._store.get("alices") is not None and out.is_dir()
        assert (await alice.delete("/v1/jobs/alices")).status_code == 200
