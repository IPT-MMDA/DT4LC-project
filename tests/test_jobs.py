"""Tests for async job queue.

Tests job submission, processing, cancellation, and queue management.
"""

import asyncio
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import PlanStep
from server.jobs import Job, JobQueue, JobStatus, SQLiteJobStore


class TestJobQueueInitialization:
    """Tests for job queue initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self) -> None:
        """Test job queue initialization."""
        queue = JobQueue(max_workers=2, max_queue_size=10, retention_hours=1)

        assert queue.max_workers == 2
        assert queue.max_queue_size == 10
        assert queue.retention_hours == 1
        assert not queue._running

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping job queue."""
        queue = JobQueue(max_workers=2)

        await queue.start()
        assert queue._running
        assert len(queue._workers) == 2

        await queue.stop()
        assert not queue._running
        assert len(queue._workers) == 0


class TestJobSubmission:
    """Tests for job submission."""

    @pytest.mark.asyncio
    async def test_submit_job(self) -> None:
        """Test job submission."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            job_id = await queue.submit_job("calculate ndvi")
            assert job_id is not None
            assert len(job_id) == 8

            job = await queue.get_job(job_id)
            assert job is not None
            assert job.prompt == "calculate ndvi"
            assert job.status == JobStatus.PENDING
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_queue_full(self) -> None:
        """Test queue full behavior."""
        queue = JobQueue(max_workers=1, max_queue_size=2)
        await queue.start()

        try:
            await queue.submit_job("job 1")
            await queue.submit_job("job 2")

            with pytest.raises(RuntimeError, match="queue is full"):
                await queue.submit_job("job 3")
        finally:
            await queue.stop()


class TestJobProcessing:
    """Tests for job processing."""

    @pytest.mark.llm
    @pytest.mark.asyncio
    async def test_job_processing(self) -> None:
        """Test job processing through queue."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            job_id = await queue.submit_job("calculate ndvi on kahovka data")

            for _ in range(60):
                job = await queue.get_job(job_id)
                if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
                await asyncio.sleep(0.5)

            job = await queue.get_job(job_id)
            assert job is not None

            assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED)

            if job.status == JobStatus.COMPLETED:
                assert job.result is not None
                if job.result.get("intent") == "pipeline":
                    assert job.plan is not None
                assert job.progress == 1.0
                assert job.completed_at is not None
        finally:
            await queue.stop()


class TestJobCancellation:
    """Tests for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_job(self) -> None:
        """Test job cancellation."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            job_id = await queue.submit_job("slow operation")

            cancelled = await queue.cancel_job(job_id)
            assert cancelled

            job = await queue.get_job(job_id)
            assert job is not None
            assert job.status == JobStatus.CANCELLED
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self) -> None:
        """Test that completed jobs cannot be cancelled."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            job_id = await queue.submit_job("calculate ndvi on kahovka data")

            for _ in range(60):
                job = await queue.get_job(job_id)
                if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
                await asyncio.sleep(0.5)

            cancelled = await queue.cancel_job(job_id)
            assert not cancelled
        finally:
            await queue.stop()


class TestJobListing:
    """Tests for job listing."""

    @pytest.mark.asyncio
    async def test_list_jobs(self) -> None:
        """Test job listing."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            await queue.submit_job("job 1")
            await queue.submit_job("job 2")
            job3_id = await queue.submit_job("job 3")

            jobs = await queue.list_jobs(limit=10)
            assert len(jobs) >= 3

            job_ids = [j.id for j in jobs]
            assert job_ids[0] == job3_id
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(self) -> None:
        """Test job listing with pagination."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            for i in range(5):
                await queue.submit_job(f"job {i}")

            page1 = await queue.list_jobs(limit=2, offset=0)
            assert len(page1) == 2

            page2 = await queue.list_jobs(limit=2, offset=2)
            assert len(page2) == 2

            page1_ids = {j.id for j in page1}
            page2_ids = {j.id for j in page2}
            assert page1_ids.isdisjoint(page2_ids)
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status(self) -> None:
        """Test filtering jobs by status."""
        queue = JobQueue(max_workers=1)
        await queue.start()

        try:
            job1_id = await queue.submit_job("job 1")
            await queue.cancel_job(job1_id)

            job2_id = await queue.submit_job("job 2")

            cancelled_jobs = await queue.list_jobs(status=JobStatus.CANCELLED)
            assert len(cancelled_jobs) >= 1
            assert all(j.status == JobStatus.CANCELLED for j in cancelled_jobs)

            pending_jobs = await queue.list_jobs(status=JobStatus.PENDING)
            assert any(j.id == job2_id for j in pending_jobs)
        finally:
            await queue.stop()


class TestJobStatistics:
    """Tests for job statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test queue statistics."""
        queue = JobQueue(max_workers=2)
        await queue.start()

        try:
            await queue.submit_job("job 1")
            await queue.submit_job("job 2")

            stats = queue.get_stats()
            assert stats["total_jobs"] >= 2
            assert stats["workers"] == 2
            assert stats["max_workers"] == 2
            assert "by_status" in stats
        finally:
            await queue.stop()


class TestJobSerialization:
    """Tests for job serialization."""

    @pytest.mark.asyncio
    async def test_job_to_dict(self) -> None:
        """Test job serialization."""
        job = Job(
            id="test123",
            status=JobStatus.COMPLETED,
            prompt="test prompt",
            progress=1.0,
        )

        data = job.to_dict()
        assert data["id"] == "test123"
        assert data["status"] == "completed"
        assert data["prompt"] == "test prompt"
        assert data["progress"] == 1.0
        assert "created_at" in data


class TestCancelRunningJob:
    """Cancelling a running multi-step job (#67 bug 1)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cancel_during_step", "step_fails"),
        [(1, False), (3, False), (3, True)],
        ids=["first-step", "last-step", "last-step-fails"],
    )
    async def test_cancel_during_step_stays_cancelled(
        self, tmp_path: Path, cancel_during_step: int, step_fails: bool
    ) -> None:
        """Cancel while a step runs: status stays cancelled, remaining steps are skipped."""
        step_running = threading.Event()
        release_step = threading.Event()
        executed: list[str] = []

        def slow_step(_self: PipelineExecutor, step: PlanStep, _item: object) -> None:
            executed.append(step.uses)
            if len(executed) == cancel_during_step:
                step_running.set()
                assert release_step.wait(timeout=10)
                if step_fails:
                    raise RuntimeError("step failed after cancel")

        plan_result = {
            "ok": True,
            "intent": "pipeline",
            "plan": {"flow": "test", "steps": [{"uses": f"algorithms/step{i}"} for i in range(1, 4)]},
        }
        process_done = asyncio.Event()
        original_process = JobQueue._process_job

        async def tracked_process(self: JobQueue, job_id: str, worker_id: int) -> None:
            await original_process(self, job_id, worker_id)
            process_done.set()

        store = SQLiteJobStore(tmp_path / "jobs.db")
        queue = JobQueue(max_workers=1, store=store)
        with (
            patch("server.jobs.orchestrate", return_value=plan_result),
            patch("dta.dti.executor.get_item", return_value=MagicMock()),
            patch.object(PipelineExecutor, "_execute_step", slow_step),
            patch.object(JobQueue, "_process_job", tracked_process),
        ):
            await queue.start()
            try:
                job_id = await queue.submit_job("multi-step job")
                assert await asyncio.to_thread(step_running.wait, 10)

                assert await queue.cancel_job(job_id)
                job = await queue.get_job(job_id)
                assert job is not None
                assert job.status == JobStatus.CANCELLED

                release_step.set()
                await asyncio.wait_for(process_done.wait(), timeout=10)

                job = await queue.get_job(job_id)
                assert job is not None
                assert job.status == JobStatus.CANCELLED
                assert len(executed) == cancel_during_step
                row = store.get(job_id)
                assert row is not None
                assert row.status == JobStatus.CANCELLED
            finally:
                release_step.set()
                await queue.stop()
