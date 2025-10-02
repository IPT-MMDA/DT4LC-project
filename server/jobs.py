"""Async job queue system for DTA.

Provides background job processing with status tracking and result caching.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any
import uuid

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.registry import load_registry
from dta.dti.schemas import ChatRequest as COEChatRequest
from dta.dti.schemas import ExecutionPlan

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job model with status tracking."""

    id: str
    status: JobStatus
    prompt: str
    attachments: list[dict[str, Any]] = field(default_factory=list)  # File attachments
    plan: ExecutionPlan | None = None
    result: dict[str, Any] | None = None
    progress: float = 0.0  # 0.0 to 1.0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "status": self.status.value,
            "prompt": self.prompt,
            "plan": self.plan.model_dump() if self.plan else None,
            "result": self.result,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @property
    def duration_seconds(self) -> float | None:
        """Get job duration in seconds.

        Returns:
            Duration or None if not started/completed
        """
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class JobQueue:
    """Async job queue with background workers."""

    def __init__(
        self,
        max_workers: int = 3,
        max_queue_size: int = 100,
        retention_hours: int = 1,
    ) -> None:
        """Initialize job queue.

        Args:
            max_workers: Maximum concurrent workers
            max_queue_size: Maximum queue size
            retention_hours: Job retention time in hours
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.retention_hours = retention_hours

        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

        # Initialize components
        self._registry = load_registry()
        self._executor = PipelineExecutor(self._registry)

    async def start(self) -> None:
        """Start worker pool."""
        if self._running:
            return

        self._running = True
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_workers)]
        logger.info(f"Started job queue with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop worker pool gracefully."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped job queue")

    async def submit_job(
        self,
        prompt: str,
        mode: str = "hybrid",
        attachments: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Submit a new job.

        Args:
            prompt: User prompt
            mode: Planning mode (hybrid/llm/template)
            attachments: Optional file attachments
            context: Optional context

        Returns:
            Job ID

        Raises:
            RuntimeError: If queue is full
        """
        if self._queue.full():
            raise RuntimeError("Job queue is full")

        # Create job
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            prompt=prompt,
            attachments=attachments or [],
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Add to queue
        await self._queue.put(job_id)
        logger.info(f"Submitted job {job_id}: {prompt}")

        return job_id

    async def get_job(self, job_id: str) -> Job | None:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job or None if not found
        """
        async with self._lock:
            return self._jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False otherwise
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False  # Already finished

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            logger.info(f"Cancelled job {job_id}")
            return True

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of jobs
        """
        async with self._lock:
            jobs = list(self._jobs.values())

            # Filter by status
            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by created_at descending
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Paginate
            return jobs[offset : offset + limit]

    async def cleanup_old_jobs(self) -> int:
        """Remove jobs older than retention period.

        Returns:
            Number of jobs removed
        """
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        removed = 0

        async with self._lock:
            # Find old jobs
            old_job_ids = [
                job_id for job_id, job in self._jobs.items() if job.completed_at and job.completed_at < cutoff
            ]

            # Remove them
            for job_id in old_job_ids:
                del self._jobs[job_id]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} old jobs")

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self._jobs)
        by_status = {}

        for job in self._jobs.values():
            status = job.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_jobs": total,
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "max_workers": self.max_workers,
            "by_status": by_status,
        }

    async def _worker(self, worker_id: int) -> None:
        """Background worker that processes jobs.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get job from queue (with timeout)
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    # Queue is empty, continue polling
                    continue

                # Process job
                await self._process_job(job_id, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    async def _process_job(self, job_id: str, worker_id: int) -> None:
        """Process a single job.

        Args:
            job_id: Job identifier
            worker_id: Worker identifier
        """
        # Get job
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            # Check if cancelled
            if job.status == JobStatus.CANCELLED:
                return

            # Mark as running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

        logger.info(f"Worker {worker_id} processing job {job_id}")

        try:
            # Plan execution
            job.progress = 0.2

            # Convert job attachments to COE format
            from dta.dti.schemas import Attachment as COEAttachment

            coe_attachments = [
                COEAttachment(
                    id=att.get("id", ""),
                    filename=att.get("filename", ""),
                    mime_type=att.get("mime_type", "image/tiff"),
                    path=att.get("path"),
                )
                for att in job.attachments
            ]

            coe_req = COEChatRequest(prompt=job.prompt, attachments=coe_attachments)
            plan_result = orchestrate(coe_req)

            if not plan_result.get("ok"):
                raise RuntimeError(plan_result.get("error", "Planning failed"))

            job.plan = ExecutionPlan(**plan_result["plan"])
            job.progress = 0.4

            # Execute plan
            execution_result = self._executor.execute(job.plan)
            job.progress = 0.8

            # Build result
            result = {
                "plan": job.plan.model_dump(),
                "execution": execution_result,
            }

            # Mark completed
            async with self._lock:
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.completed_at = datetime.now()

            logger.info(f"Worker {worker_id} completed job {job_id} in {job.duration_seconds:.2f}s")

        except Exception as e:
            # Mark failed
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()

            logger.error(f"Worker {worker_id} failed job {job_id}: {e}", exc_info=True)


# Global job queue instance
_job_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """Get global job queue instance.

    Returns:
        Job queue instance
    """
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(max_workers=3, max_queue_size=100, retention_hours=1)
    return _job_queue
