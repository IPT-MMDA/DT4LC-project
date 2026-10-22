"""Tests for job store implementations.

Tests MemoryJobStore, SQLiteJobStore, and JobQueue integration with SQLiteJobStore.
"""

from datetime import datetime
from pathlib import Path

import pytest

from dta.dti.schemas import ExecutionPlan, PlanStep
from server.jobs import Job, JobQueue, JobStatus, MemoryJobStore, SQLiteJobStore


def _make_job(
    job_id: str = "test123",
    status: JobStatus = JobStatus.PENDING,
    prompt: str = "calculate ndvi",
) -> Job:
    return Job(id=job_id, status=status, prompt=prompt)


def _make_full_job() -> Job:
    """Create a job with all fields populated for round-trip testing."""
    plan = ExecutionPlan(
        flow="auto",
        steps=[PlanStep(uses="algorithms/ndvi", binds={"RasterPath": "input"})],
        outputs=["NDVIMap"],
    )
    return Job(
        id="full123",
        status=JobStatus.COMPLETED,
        prompt="calculate ndvi on satellite data",
        attachments=[{"id": "a1", "filename": "test.tif", "mime_type": "image/tiff", "path": "/tmp/test.tif"}],
        context={"previous_attachments": [], "session_id": "s1"},
        plan=plan,
        result={"intent": "pipeline", "execution": {"status": "ok"}},
        progress=1.0,
        error=None,
        created_at=datetime(2026, 1, 15, 10, 30, 0),
        started_at=datetime(2026, 1, 15, 10, 30, 1),
        completed_at=datetime(2026, 1, 15, 10, 30, 5),
    )


class TestMemoryJobStore:
    """Tests for in-memory job store."""

    def test_save_and_get(self) -> None:
        store = MemoryJobStore()
        job = _make_job()
        store.save(job)
        assert store.get("test123") is job

    def test_get_missing(self) -> None:
        store = MemoryJobStore()
        assert store.get("nonexistent") is None

    def test_delete(self) -> None:
        store = MemoryJobStore()
        store.save(_make_job())
        assert store.delete("test123") is True
        assert store.get("test123") is None

    def test_delete_missing(self) -> None:
        store = MemoryJobStore()
        assert store.delete("nonexistent") is False

    def test_list_all(self) -> None:
        store = MemoryJobStore()
        store.save(_make_job("j1"))
        store.save(_make_job("j2"))
        jobs = store.list_all()
        assert len(jobs) == 2
        assert {j.id for j in jobs} == {"j1", "j2"}

    def test_count(self) -> None:
        store = MemoryJobStore()
        assert store.count() == 0
        store.save(_make_job())
        assert store.count() == 1

    def test_keys(self) -> None:
        store = MemoryJobStore()
        store.save(_make_job("a"))
        store.save(_make_job("b"))
        assert set(store.keys()) == {"a", "b"}

    def test_recover_interrupted_is_noop(self) -> None:
        store = MemoryJobStore()
        assert store.recover_interrupted() == 0


class TestSQLiteJobStore:
    """Tests for SQLite-backed job store."""

    def test_table_auto_created(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = SQLiteJobStore(db_path)
        assert store.count() == 0

    def test_save_and_get(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        job = _make_job()
        store.save(job)

        loaded = store.get("test123")
        assert loaded is not None
        assert loaded.id == "test123"
        assert loaded.status == JobStatus.PENDING
        assert loaded.prompt == "calculate ndvi"

    def test_get_missing(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        assert store.get("nonexistent") is None

    def test_round_trip_all_fields(self, tmp_path: Path) -> None:
        """Verify all Job fields survive serialization round-trip."""
        store = SQLiteJobStore(tmp_path / "test.db")
        original = _make_full_job()
        store.save(original)

        loaded = store.get("full123")
        assert loaded is not None
        assert loaded.id == original.id
        assert loaded.status == original.status
        assert loaded.prompt == original.prompt
        assert loaded.attachments == original.attachments
        assert loaded.context == original.context
        assert loaded.plan is not None
        assert loaded.plan.model_dump() == original.plan.model_dump()
        assert loaded.result == original.result
        assert loaded.progress == original.progress
        assert loaded.error == original.error
        assert loaded.created_at == original.created_at
        assert loaded.started_at == original.started_at
        assert loaded.completed_at == original.completed_at

    def test_save_updates_existing(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        job = _make_job()
        store.save(job)

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        store.save(job)

        loaded = store.get("test123")
        assert loaded is not None
        assert loaded.status == JobStatus.RUNNING
        assert loaded.started_at is not None

    def test_delete(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        store.save(_make_job())
        assert store.delete("test123") is True
        assert store.get("test123") is None

    def test_delete_missing(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        assert store.delete("nonexistent") is False

    def test_list_all(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        store.save(_make_job("j1"))
        store.save(_make_job("j2"))
        store.save(_make_job("j3"))
        jobs = store.list_all()
        assert len(jobs) == 3
        assert {j.id for j in jobs} == {"j1", "j2", "j3"}

    def test_count(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        assert store.count() == 0
        store.save(_make_job("a"))
        store.save(_make_job("b"))
        assert store.count() == 2

    def test_keys(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        store.save(_make_job("x"))
        store.save(_make_job("y"))
        assert set(store.keys()) == {"x", "y"}

    def test_recover_interrupted(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = SQLiteJobStore(db_path)
        store.save(_make_job("p1", status=JobStatus.PENDING))
        store.save(_make_job("r1", status=JobStatus.RUNNING))
        store.save(_make_job("c1", status=JobStatus.COMPLETED))

        # Simulate restart by creating a new store on the same db.
        store2 = SQLiteJobStore(db_path)
        recovered = store2.recover_interrupted()
        assert recovered == 2

        p1 = store2.get("p1")
        assert p1 is not None
        assert p1.status == JobStatus.FAILED
        assert p1.error == "Server restarted while job was in progress"

        r1 = store2.get("r1")
        assert r1 is not None
        assert r1.status == JobStatus.FAILED

        c1 = store2.get("c1")
        assert c1 is not None
        assert c1.status == JobStatus.COMPLETED

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"

        store1 = SQLiteJobStore(db_path)
        job = _make_job(status=JobStatus.COMPLETED)
        job.completed_at = datetime.now()
        store1.save(job)

        store2 = SQLiteJobStore(db_path)
        loaded = store2.get("test123")
        assert loaded is not None
        assert loaded.prompt == "calculate ndvi"
        assert loaded.status == JobStatus.COMPLETED


class TestJobQueueWithSQLiteStore:
    """Integration tests for JobQueue backed by SQLiteJobStore.

    Workers are not started to avoid race conditions between submit and assertions.
    These tests verify store integration only.
    """

    @pytest.mark.asyncio
    async def test_submit_and_get(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        queue = JobQueue(max_workers=1, store=store)

        job_id = await queue.submit_job("test prompt")
        job = await queue.get_job(job_id)
        assert job is not None
        assert job.prompt == "test prompt"
        assert job.status == JobStatus.PENDING

        db_job = store.get(job_id)
        assert db_job is not None
        assert db_job.prompt == "test prompt"

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        queue = JobQueue(max_workers=1, store=store)

        job_id = await queue.submit_job("test prompt")
        cancelled = await queue.cancel_job(job_id)
        assert cancelled

        db_job = store.get(job_id)
        assert db_job is not None
        assert db_job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_total_count_and_job_ids(self, tmp_path: Path) -> None:
        store = SQLiteJobStore(tmp_path / "test.db")
        queue = JobQueue(max_workers=1, store=store)

        id1 = await queue.submit_job("job 1")
        id2 = await queue.submit_job("job 2")

        assert await queue.get_total_count() == 2
        ids = await queue.get_job_ids()
        assert set(ids) == {id1, id2}
