"""Phase 4 Tests - Async Job Queue."""

import asyncio

import pytest

from server.jobs import Job, JobQueue, JobStatus


@pytest.mark.asyncio
async def test_job_queue_initialization() -> None:
    """Test job queue initialization."""
    queue = JobQueue(max_workers=2, max_queue_size=10, retention_hours=1)

    assert queue.max_workers == 2
    assert queue.max_queue_size == 10
    assert queue.retention_hours == 1
    assert not queue._running


@pytest.mark.asyncio
async def test_job_queue_start_stop() -> None:
    """Test starting and stopping job queue."""
    queue = JobQueue(max_workers=2)

    await queue.start()
    assert queue._running
    assert len(queue._workers) == 2

    await queue.stop()
    assert not queue._running
    assert len(queue._workers) == 0


@pytest.mark.asyncio
async def test_submit_job() -> None:
    """Test job submission."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        job_id = await queue.submit_job("calculate ndvi")
        assert job_id is not None
        assert len(job_id) == 8  # UUID truncated to 8 chars

        job = await queue.get_job(job_id)
        assert job is not None
        assert job.prompt == "calculate ndvi"
        assert job.status == JobStatus.PENDING
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_job_processing() -> None:
    """Test job processing through queue."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        job_id = await queue.submit_job("calculate ndvi on kahovka data")

        # Wait for job to complete (max 30 seconds)
        for _ in range(60):
            job = await queue.get_job(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.5)

        job = await queue.get_job(job_id)
        assert job is not None

        # Should complete successfully (or fail with valid error)
        assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED)

        if job.status == JobStatus.COMPLETED:
            assert job.result is not None
            assert job.plan is not None
            assert job.progress == 1.0
            assert job.completed_at is not None
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_cancel_job() -> None:
    """Test job cancellation."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        job_id = await queue.submit_job("slow operation")

        # Cancel immediately
        cancelled = await queue.cancel_job(job_id)
        assert cancelled

        job = await queue.get_job(job_id)
        assert job is not None
        assert job.status == JobStatus.CANCELLED
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_cancel_completed_job() -> None:
    """Test that completed jobs cannot be cancelled."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        job_id = await queue.submit_job("calculate ndvi on kahovka data")

        # Wait for completion
        for _ in range(60):
            job = await queue.get_job(job_id)
            if job and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.5)

        # Try to cancel
        cancelled = await queue.cancel_job(job_id)
        assert not cancelled  # Cannot cancel finished job
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_list_jobs() -> None:
    """Test job listing."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        # Submit multiple jobs
        job1_id = await queue.submit_job("job 1")
        job2_id = await queue.submit_job("job 2")
        job3_id = await queue.submit_job("job 3")

        # List all jobs
        jobs = await queue.list_jobs(limit=10)
        assert len(jobs) >= 3

        # Check order (most recent first)
        job_ids = [j.id for j in jobs]
        assert job_ids[0] == job3_id  # Most recent
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_list_jobs_pagination() -> None:
    """Test job listing with pagination."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        # Submit 5 jobs
        for i in range(5):
            await queue.submit_job(f"job {i}")

        # Get first page
        page1 = await queue.list_jobs(limit=2, offset=0)
        assert len(page1) == 2

        # Get second page
        page2 = await queue.list_jobs(limit=2, offset=2)
        assert len(page2) == 2

        # No overlap
        page1_ids = {j.id for j in page1}
        page2_ids = {j.id for j in page2}
        assert page1_ids.isdisjoint(page2_ids)
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_list_jobs_filter_by_status() -> None:
    """Test filtering jobs by status."""
    queue = JobQueue(max_workers=1)
    await queue.start()

    try:
        # Submit and immediately cancel one job
        job1_id = await queue.submit_job("job 1")
        await queue.cancel_job(job1_id)

        # Submit another job
        job2_id = await queue.submit_job("job 2")

        # Filter by cancelled
        cancelled_jobs = await queue.list_jobs(status=JobStatus.CANCELLED)
        assert len(cancelled_jobs) >= 1
        assert all(j.status == JobStatus.CANCELLED for j in cancelled_jobs)

        # Filter by pending
        pending_jobs = await queue.list_jobs(status=JobStatus.PENDING)
        assert any(j.id == job2_id for j in pending_jobs)
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_queue_full() -> None:
    """Test queue full behavior."""
    queue = JobQueue(max_workers=1, max_queue_size=2)
    await queue.start()

    try:
        # Fill queue
        await queue.submit_job("job 1")
        await queue.submit_job("job 2")

        # Third job should raise error
        with pytest.raises(RuntimeError, match="queue is full"):
            await queue.submit_job("job 3")
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_get_stats() -> None:
    """Test queue statistics."""
    queue = JobQueue(max_workers=2)
    await queue.start()

    try:
        # Submit some jobs
        await queue.submit_job("job 1")
        await queue.submit_job("job 2")

        stats = queue.get_stats()
        assert stats["total_jobs"] >= 2
        assert stats["workers"] == 2
        assert stats["max_workers"] == 2
        assert "by_status" in stats
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_job_to_dict() -> None:
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
