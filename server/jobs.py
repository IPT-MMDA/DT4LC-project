"""Async job queue system for DTA.

Provides background job processing with status tracking and result caching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
import functools
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from typing import Any
import uuid

import numpy as np

from dta.dti import artifact_files
from dta.dti.coe.context_agent import _deterministic_context
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.coe.planner import plan_template
from dta.dti.executor import PipelineExecutor
from dta.dti.registry import load_registry
from dta.dti.schemas import Attachment as COEAttachment
from dta.dti.schemas import ChatRequest as COEChatRequest
from dta.dti.schemas import ExecutionPlan

from .job_events import TERMINAL_EVENTS, classify_failure, done_summary, intent_text, step_skeleton

logger = logging.getLogger(__name__)


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable objects to JSON-safe types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer | np.floating):
        val = float(obj)
        # Handle NaN and Inf values (not JSON compliant)
        if np.isnan(val):
            return None
        elif np.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        # Handle Python float NaN and Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


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
    context: dict[str, Any] | None = None  # Chat context including previous attachments
    plan: ExecutionPlan | None = None
    result: dict[str, Any] | None = None
    progress: float = 0.0  # 0.0 to 1.0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # Event log, ``{"seq", "type", "data"}`` per event (see server/job_events.py); metadata only, no arrays.
    events: list[dict[str, Any]] = field(default_factory=list)
    # Session id of the submitter (a stable user id once sign-in exists) and its role then;
    # ``None`` for jobs from before v0.6.0, which only operators see.
    owner: str | None = None
    owner_role: str | None = None

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
            "result": _make_json_serializable(self.result) if self.result else None,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "events": list(self.events),
            "owner": self.owner,
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


class JobStore(ABC):
    """Abstract storage backend for jobs."""

    @abstractmethod
    def save(self, job: Job) -> None:
        """Persist a job (insert or update)."""

    @abstractmethod
    def get(self, job_id: str) -> Job | None:
        """Retrieve a single job by ID."""

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """Delete a job. Returns True if it existed."""

    @abstractmethod
    def list_all(self) -> list[Job]:
        """Return all jobs."""

    @abstractmethod
    def list_summaries(self) -> list[Job]:
        """Return all jobs without ``result``, ``context``, ``attachments`` and ``events`` (``GET /v1/jobs`` fields).

        The returned jobs are partial: never ``save()`` them.
        """

    def get_owner(self, job_id: str) -> str | None:
        """The job's ``owner`` (``None`` if unknown or ownerless) without loading its result."""
        job = self.get(job_id)
        return job.owner if job else None

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored jobs."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Return all job IDs."""

    @abstractmethod
    def recover_interrupted(self) -> int:
        """Mark any PENDING/RUNNING jobs as FAILED on startup.

        Returns:
            Number of recovered jobs.
        """


class MemoryJobStore(JobStore):
    """In-memory job storage using OrderedDict."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()

    def save(self, job: Job) -> None:
        self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    def list_all(self) -> list[Job]:
        return list(self._jobs.values())

    def list_summaries(self) -> list[Job]:
        return [replace(j, result=None, context=None, attachments=[], events=[]) for j in self._jobs.values()]

    def count(self) -> int:
        return len(self._jobs)

    def keys(self) -> list[str]:
        return list(self._jobs.keys())

    def recover_interrupted(self) -> int:
        return 0


class SQLiteJobStore(JobStore):
    """SQLite-backed persistent job storage."""

    _ADDED_COLUMNS = {"events": "TEXT NOT NULL DEFAULT '[]'", "owner": "TEXT", "owner_role": "TEXT"}

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id           TEXT PRIMARY KEY,
                status       TEXT NOT NULL,
                prompt       TEXT NOT NULL,
                attachments  TEXT NOT NULL DEFAULT '[]',
                context      TEXT,
                plan         TEXT,
                result       TEXT,
                progress     REAL NOT NULL DEFAULT 0.0,
                error        TEXT,
                created_at   TEXT NOT NULL,
                started_at   TEXT,
                completed_at TEXT
            )
        """)
        # Columns added after v0.5.1: an older jobs.db gets them with their default on open.
        have = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        for name, decl in self._ADDED_COLUMNS.items():
            if name not in have:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
        self._conn.commit()

    def _serialize_job(self, job: Job) -> tuple[str | float | None, ...]:
        return (
            job.id,
            job.status.value,
            job.prompt,
            json.dumps(job.attachments),
            json.dumps(job.context) if job.context is not None else None,
            json.dumps(job.plan.model_dump()) if job.plan is not None else None,
            json.dumps(_make_json_serializable(job.result)) if job.result is not None else None,
            job.progress,
            job.error,
            job.created_at.isoformat(),
            job.started_at.isoformat() if job.started_at else None,
            job.completed_at.isoformat() if job.completed_at else None,
            json.dumps(list(job.events)),
            job.owner,
            job.owner_role,
        )

    def _deserialize_row(self, row: sqlite3.Row) -> Job:
        plan_data = json.loads(row["plan"]) if row["plan"] else None
        return Job(
            id=row["id"],
            status=JobStatus(row["status"]),
            prompt=row["prompt"],
            attachments=json.loads(row["attachments"]),
            context=json.loads(row["context"]) if row["context"] else None,
            plan=ExecutionPlan(**plan_data) if plan_data else None,
            result=json.loads(row["result"]) if row["result"] else None,
            progress=row["progress"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            events=json.loads(row["events"]),
            owner=row["owner"],
            owner_role=row["owner_role"],
        )

    def save(self, job: Job) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, status, prompt, attachments, context, plan, result,
                progress, error, created_at, started_at, completed_at, events, owner, owner_role)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._serialize_job(job),
        )
        self._conn.commit()

    def get(self, job_id: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._deserialize_row(row) if row else None

    def delete(self, job_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_all(self) -> list[Job]:
        rows = self._conn.execute("SELECT * FROM jobs").fetchall()
        return [self._deserialize_row(row) for row in rows]

    def list_summaries(self) -> list[Job]:
        rows = self._conn.execute(
            "SELECT id, status, prompt, plan, progress, error, created_at, started_at, completed_at, owner, owner_role"
            " FROM jobs"
        ).fetchall()
        return [
            Job(
                id=row["id"],
                status=JobStatus(row["status"]),
                prompt=row["prompt"],
                plan=ExecutionPlan(**json.loads(row["plan"])) if row["plan"] else None,
                progress=row["progress"],
                error=row["error"],
                created_at=datetime.fromisoformat(row["created_at"]),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                owner=row["owner"],
                owner_role=row["owner_role"],
            )
            for row in rows
        ]

    def get_owner(self, job_id: str) -> str | None:
        row = self._conn.execute("SELECT owner FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["owner"] if row else None

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0])

    def keys(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM jobs").fetchall()
        return [row[0] for row in rows]

    def recover_interrupted(self) -> int:
        cursor = self._conn.execute(
            "UPDATE jobs SET status = ?, error = ?, completed_at = ? WHERE status IN (?, ?)",
            (
                JobStatus.FAILED.value,
                "Server restarted while job was in progress",
                datetime.now().isoformat(),
                JobStatus.PENDING.value,
                JobStatus.RUNNING.value,
            ),
        )
        self._conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Recovered {count} interrupted jobs (marked as FAILED)")
        return count


class JobQueue:
    """Async job queue with background workers."""

    def __init__(
        self,
        max_workers: int = 3,
        max_queue_size: int = 100,
        retention_hours: float | None = 1,
        store: JobStore | None = None,
    ) -> None:
        """Initialize job queue.

        Args:
            max_workers: Maximum concurrent workers
            max_queue_size: Maximum queue size
            retention_hours: Job retention time in hours; ``None`` keeps jobs (guest jobs
                still expire, see :meth:`cleanup_old_jobs`)
            store: Job storage backend (defaults to MemoryJobStore)
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.retention_hours = retention_hours

        self._store: JobStore = store or MemoryJobStore()
        self._jobs_cache: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._lock = asyncio.Lock()
        # Guards every job's event log and the subscribers; worker threads emit events too.
        self._events_lock = threading.Lock()
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]]] = {}

        # Thread pool for running blocking operations
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job_worker")

        # Initialize components - registry is thread-safe (read-only), executor is created per-job
        self._registry = load_registry()

        # Recover any interrupted jobs from previous run
        self._store.recover_interrupted()

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

        # Shutdown thread pool
        self._thread_pool.shutdown(wait=False)
        logger.info("Stopped job queue")

    async def submit_job(
        self,
        prompt: str,
        mode: str = "hybrid",
        attachments: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        owner: str | None = None,
        owner_role: str | None = None,
    ) -> str:
        """Submit a new job.

        Args:
            prompt: User prompt
            mode: Planning mode (hybrid/llm/template)
            attachments: Optional file attachments
            context: Optional context
            owner: Submitter's session id
            owner_role: Submitter's role (guest jobs expire with the guest session)

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
            context=context,
            owner=owner,
            owner_role=owner_role,
        )
        self._emit(job, "intent", text=self._intent_text(job))

        async with self._lock:
            self._store.save(job)
            self._jobs_cache[job_id] = job

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
            cached = self._jobs_cache.get(job_id)
            if cached is not None:
                return cached
            return self._store.get(job_id)

    async def get_owner(self, job_id: str) -> str | None:
        """The job's ``owner`` without loading its result; ``None`` for an unknown or ownerless job."""
        async with self._lock:
            cached = self._jobs_cache.get(job_id)
            if cached is not None:
                return cached.owner
            return self._store.get_owner(job_id)

    # --- Event log -------------------------------------------------------------

    def _emit(self, job: Job, event_type: str, **data: Any) -> dict[str, Any] | None:
        """Append an event to ``job.events`` and hand it to the job's subscribers; safe from any thread.

        ``None`` values are dropped from ``data``. Nothing is appended after a terminal event.
        """
        payload = {k: v for k, v in data.items() if v is not None}
        payload["t"] = round(time.time(), 3)
        with self._events_lock:
            if job.events and job.events[-1]["type"] in TERMINAL_EVENTS:
                return None
            event = {"seq": len(job.events) + 1, "type": event_type, "data": payload}
            job.events.append(event)
            for loop, queue in self._subscribers.get(job.id, []):
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                except RuntimeError:  # the subscriber's loop is closed
                    pass
        return event

    @staticmethod
    def _current_step(job: Job) -> int | None:
        """Index of the step started but not finished, per the event log."""
        running: int | None = None
        for event in job.events:
            if event["type"] == "step_started":
                running = event["data"]["index"]
            elif event["type"] == "step_finished" and event["data"]["index"] == running:
                running = None
        return running

    def _done_data(self, job: Job) -> dict[str, Any]:
        """``done`` data: ``elapsed_s`` and the one-line summary (a conversation reply ran no steps)."""
        elapsed = job.duration_seconds or 0.0
        if job.plan is None:
            summary = "Answered without running an analysis"
        else:
            summary = done_summary(step_skeleton(job.plan, self._registry), elapsed)
        return {"elapsed_s": round(elapsed, 1), "summary": summary}

    def _terminal_event(self, job: Job) -> None:
        """Append the terminal event a finished job lacks (a job from before the event log, or one
        marked failed by ``recover_interrupted``), so every finished job's stream ends."""
        if job.status == JobStatus.COMPLETED:
            self._emit(job, "done", **self._done_data(job))
        elif job.status == JobStatus.FAILED:
            stage = "running" if job.plan is not None else "planning"
            self._emit(job, "failed", **classify_failure(job.error or "", stage, self._current_step(job)))
        elif job.status == JobStatus.CANCELLED:
            self._emit(job, "cancelled", at_step=self._current_step(job))

    def subscribe(
        self, job_id: str, after: int = 0
    ) -> tuple[list[dict[str, Any]], asyncio.Queue[dict[str, Any]] | None] | None:
        """Events with ``seq > after`` so far, and a queue for the live tail.

        Call from the event loop. Returns ``None`` for an unknown job, and no queue
        when the log already ends with its terminal event. Pass the queue to
        :meth:`unsubscribe` when done.
        """
        loop = asyncio.get_running_loop()
        with self._events_lock:
            job = self._jobs_cache.get(job_id) or self._store.get(job_id)
            if job is None:
                return None
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self._terminal_event(job)
        with self._events_lock:
            backlog = [e for e in job.events if e["seq"] > after]
            if job.events and job.events[-1]["type"] in TERMINAL_EVENTS:
                return backlog, None
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            self._subscribers.setdefault(job_id, []).append((loop, queue))
            return backlog, queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Stop handing ``job_id``'s events to ``queue``."""
        with self._events_lock:
            subscribers = [s for s in self._subscribers.get(job_id, []) if s[1] is not queue]
            if subscribers:
                self._subscribers[job_id] = subscribers
            else:
                self._subscribers.pop(job_id, None)

    def _intent_text(self, job: Job) -> str:
        """The ``intent`` sentence: the template plan for the deterministic context, no LLM call."""
        attachments = [a for a in job.attachments if a.get("path")]
        if not attachments and job.context:
            attachments = [a for a in job.context.get("previous_attachments", []) if a.get("path")]
        try:
            req = COEChatRequest(
                prompt=job.prompt,
                attachments=[
                    COEAttachment(
                        id=a.get("id", ""),
                        filename=a.get("filename", ""),
                        mime_type=a.get("mime_type", "image/tiff"),
                        path=a["path"],
                    )
                    for a in attachments
                ],
            )
            return intent_text(plan_template(_deterministic_context(req), self._registry), self._registry, attachments)
        except Exception as e:
            logger.warning(f"Job {job.id}: no intent sentence from the template plan ({e})")
            return "I'll work out which analysis fits this request."

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False otherwise
        """
        async with self._lock:
            job = self._jobs_cache.get(job_id) or self._store.get(job_id)
            if not job:
                return False

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False  # Already finished

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            self._emit(job, "cancelled", at_step=self._current_step(job))
            self._store.save(job)
            self._jobs_cache.pop(job_id, None)
            logger.info(f"Cancelled job {job_id}")
            return True

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
        owner: str | None = None,
    ) -> list[Job]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum results
            offset: Offset for pagination
            owner: Only this owner's jobs; ``None`` lists every job (operators)

        Returns:
            List of jobs; finished jobs come without ``result`` (fetch one by id for it)
        """
        async with self._lock:
            # Merge store and cache (cache has fresher state for in-flight jobs)
            all_jobs: dict[str, Job] = {j.id: j for j in self._store.list_summaries()}
            all_jobs.update(self._jobs_cache)
            jobs = list(all_jobs.values())

            if owner is not None:
                jobs = [j for j in jobs if j.owner == owner]

            # Filter by status
            if status:
                jobs = [j for j in jobs if j.status == status]

            # Sort by created_at descending
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Paginate
            return jobs[offset : offset + limit]

    async def cleanup_old_jobs(self, guest_hours: float | None = None) -> int:
        """Remove old finished jobs with their ``OUTPUTS_PATH/<job_id>/`` artifact files.

        A job is old when it finished more than ``retention_hours`` ago (unless that is
        ``None``) or, with ``guest_hours``, when a guest submitted it more than
        ``guest_hours`` ago. The rows go under the lock; the directories are deleted in
        a thread so the event loop keeps serving.

        Returns:
            Number of jobs removed
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=self.retention_hours) if self.retention_hours is not None else None
        guest_cutoff = now - timedelta(hours=guest_hours) if guest_hours is not None else None

        def expired(job: Job) -> bool:
            if job.completed_at is None:
                return False
            if cutoff is not None and job.completed_at < cutoff:
                return True
            return guest_cutoff is not None and job.owner_role == "guest" and job.created_at < guest_cutoff

        async with self._lock:
            old_job_ids = [j.id for j in self._store.list_summaries() if expired(j)]
            for job_id in old_job_ids:
                self._store.delete(job_id)
                self._jobs_cache.pop(job_id, None)

        removed = len(old_job_ids)
        dirs_removed = await asyncio.to_thread(self._remove_output_dirs, old_job_ids) if old_job_ids else 0
        if removed > 0:
            logger.info(f"Cleaned up {removed} old jobs and {dirs_removed} output directories")

        return removed

    @staticmethod
    def _remove_output_dirs(job_ids: list[str]) -> int:
        """Delete ``OUTPUTS_PATH/<job_id>/`` for each id, never anything outside OUTPUTS_PATH. Blocking."""
        outputs_root = artifact_files.OUTPUTS_PATH.resolve()
        dirs_removed = 0
        for job_id in job_ids:
            job_dir = artifact_files.output_dir(job_id).resolve()
            if outputs_root not in job_dir.parents:
                logger.warning(f"Job {job_id}: output directory {job_dir} is outside {outputs_root}, not removed")
            elif job_dir.is_dir():
                shutil.rmtree(job_dir, ignore_errors=True)
                if not job_dir.exists():
                    dirs_removed += 1
        return dirs_removed

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Statistics dictionary
        """
        # Merge store and cache for accurate stats
        all_jobs: dict[str, Job] = {j.id: j for j in self._store.list_all()}
        all_jobs.update(self._jobs_cache)

        by_status: dict[str, int] = {}
        for job in all_jobs.values():
            s = job.status.value
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_jobs": len(all_jobs),
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "max_workers": self.max_workers,
            "by_status": by_status,
        }

    async def get_job_ids(self) -> list[str]:
        """Return all job IDs.

        Returns:
            List of job IDs
        """
        async with self._lock:
            ids = set(self._store.keys())
            ids.update(self._jobs_cache.keys())
            return list(ids)

    async def get_total_count(self, owner: str | None = None) -> int:
        """Return total job count.

        Args:
            owner: Count only this owner's jobs; ``None`` counts every job

        Returns:
            Total number of jobs
        """
        async with self._lock:
            if owner is not None:
                jobs: dict[str, Job] = {j.id: j for j in self._store.list_summaries()}
                jobs.update(self._jobs_cache)
                return sum(1 for j in jobs.values() if j.owner == owner)
            ids = set(self._store.keys())
            ids.update(self._jobs_cache.keys())
            return len(ids)

    async def _worker(self, worker_id: int) -> None:
        """Background worker that processes jobs.

        Args:
            worker_id: Worker identifier
        """
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get job from queue (with timeout)
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                    # Queue is empty, continue polling
                    continue

                logger.debug(f"Worker {worker_id} picked up job {job_id}")
                # Process job
                await self._process_job(job_id, worker_id)

            except asyncio.CancelledError:
                break
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                # Timeout during shutdown - ignore
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    async def _check_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled
        """
        async with self._lock:
            job = self._jobs_cache.get(job_id)
            return job is not None and job.status == JobStatus.CANCELLED

    def _run_blocking_job(self, job: Job, job_id: str, worker_id: int) -> dict[str, Any]:
        """Run blocking job operations in a thread.

        This runs in a separate thread to not block the async event loop.

        Args:
            job: Job object
            job_id: Job identifier
            worker_id: Worker identifier

        Returns:
            Result dictionary or raises exception
        """
        from dta.dti.executor import CancellationError

        # Convert job attachments to COE format
        logger.debug(f"Job {job_id}: Starting processing with {len(job.attachments)} attachments")
        logger.debug(f"Job {job_id}: Raw attachments: {job.attachments}")
        coe_attachments = []
        for att in job.attachments:
            path = att.get("path")
            if path:
                coe_attachments.append(
                    COEAttachment(
                        id=att.get("id", ""),
                        filename=att.get("filename", ""),
                        mime_type=att.get("mime_type", "image/tiff"),
                        path=path,
                    )
                )
            else:
                logger.warning(f"Job {job_id}: Attachment missing path: {att.get('filename', 'unknown')}")

        # If no current attachments, check context for previous attachments
        context_status = "present" if job.context else "None"
        logger.debug(f"Job {job_id}: {len(coe_attachments)} direct attachments with paths, context={context_status}")
        if not coe_attachments and job.context:
            context_attachments = job.context.get("previous_attachments", [])
            logger.debug(f"Job {job_id}: Found {len(context_attachments)} previous attachments in context")
            for att in context_attachments:
                if att.get("path"):
                    coe_attachments.append(
                        COEAttachment(
                            id=att.get("id", ""),
                            filename=att.get("filename", ""),
                            mime_type=att.get("mime_type", "image/tiff"),
                            path=att.get("path"),
                        )
                    )
            if coe_attachments:
                logger.debug(f"Job {job_id}: Using {len(coe_attachments)} attachment(s) from context")
            else:
                logger.warning(f"Job {job_id}: No valid attachments found in context")

        # Check for cancellation before planning
        if job.status == JobStatus.CANCELLED:
            raise CancellationError("Job cancelled before planning")

        job.progress = 0.2
        self._emit(job, "planning", phase="started")
        logger.debug(f"Job {job_id}: Calling orchestrate with {len(coe_attachments)} attachments")
        coe_req = COEChatRequest(prompt=job.prompt, attachments=coe_attachments)
        plan_result = orchestrate(coe_req)
        logger.debug(f"Job {job_id}: Orchestrate returned ok={plan_result.get('ok')}")

        if not plan_result.get("ok"):
            raise RuntimeError(plan_result.get("error", "Planning failed"))

        # Check for cancellation after planning
        # mypy can't see that job.status may have been mutated by another coroutine
        # (the cancel endpoint) during the await above, so it thinks the comparison is dead.
        if job.status == JobStatus.CANCELLED:  # type: ignore[comparison-overlap]
            raise CancellationError("Job cancelled after planning")

        # Handle conversational intent - no pipeline execution needed
        if plan_result.get("intent") == "conversation":
            return {
                "intent": "conversation",
                "response": plan_result.get("response", ""),
                "reason": plan_result.get("reason", ""),
            }

        # Pipeline intent - execute the plan
        job.plan = ExecutionPlan(**plan_result["plan"])
        job.progress = 0.4
        logger.debug(f"Executing plan with {len(job.plan.steps)} steps")
        filenames = {a.path: a.filename for a in coe_attachments if a.path}
        skeleton = step_skeleton(job.plan, self._registry, filenames)
        names = {row["index"]: row["name"] for row in skeleton}
        self._emit(
            job, "planning", phase="ready", steps=skeleton, composed=bool(plan_result["plan"].get("composed", False))
        )
        step_t0: dict[int, float] = {}

        def on_progress(event: dict[str, Any]) -> None:
            """Executor step events -> job events; keeps ``job.progress`` between 0.4 and 0.8."""
            index = event.get("index")
            if not isinstance(index, int):
                return
            if event.get("event") == "step_start":
                step_t0[index] = time.monotonic()
                self._emit(job, "step_started", index=index, id=event["step_id"], name=names.get(index))
            elif event.get("event") == "step_complete":
                job.progress = 0.4 + 0.4 * index / max(len(skeleton), 1)
                elapsed = time.monotonic() - step_t0.get(index, time.monotonic())
                self._emit(job, "step_finished", index=index, id=event["step_id"], elapsed_s=round(elapsed, 2))
            elif event.get("event") == "step_progress":
                keys = ("fraction", "message", "bytes", "total_bytes", "eta_s")
                self._emit(job, "step_progress", index=index, **{k: event.get(k) for k in keys})

        # Create synchronous cancellation checker
        def check_cancelled() -> bool:
            """Check if job was cancelled (sync version for executor)."""
            return job.status == JobStatus.CANCELLED

        # Create a fresh executor for this job (executor has mutable state - artifacts dict)
        executor = PipelineExecutor(self._registry)

        # Execute plan with cancellation support; artifact files go to OUTPUTS_PATH/<job_id>/
        execution_result = executor.execute(
            job.plan,
            on_progress=on_progress,
            is_cancelled=check_cancelled,
            output_dir=artifact_files.output_dir(job_id),
        )

        job.progress = 0.8

        # Build result
        return {
            "intent": "pipeline",
            "plan": job.plan.model_dump(),
            "execution": execution_result,
        }

    async def _process_job(self, job_id: str, worker_id: int) -> None:
        """Process a single job.

        Args:
            job_id: Job identifier
            worker_id: Worker identifier
        """
        from dta.dti.executor import CancellationError, ModelNotInstalledError

        # Get job
        async with self._lock:
            job = self._jobs_cache.get(job_id)
            if not job:
                return

            # Check if cancelled
            if job.status == JobStatus.CANCELLED:
                return

            # Mark as running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._store.save(job)

        logger.info(f"Worker {worker_id} processing job {job_id}")

        try:
            # Run blocking operations in thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._thread_pool,
                functools.partial(self._run_blocking_job, job, job_id, worker_id),
            )

            # Mark completed, unless cancel_job ran while the last step was executing
            async with self._lock:
                if job.status == JobStatus.CANCELLED:
                    raise CancellationError("Job cancelled during its last step")
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.completed_at = datetime.now()
                self._emit(job, "done", **self._done_data(job))
                self._store.save(job)
                self._jobs_cache.pop(job_id, None)

            logger.info(f"Worker {worker_id} completed job {job_id} in {job.duration_seconds:.2f}s")

        except CancellationError:
            logger.info(f"Job {job_id} execution cancelled")
            async with self._lock:
                self._store.save(job)
                self._jobs_cache.pop(job_id, None)

        except ModelNotInstalledError as e:
            # Model not installed - mark as failed with special error info
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.result = e.to_dict()  # Include model info for frontend
                job.completed_at = datetime.now()
                self._emit(job, "failed", **classify_failure(e, "running", self._current_step(job)))
                self._store.save(job)
                self._jobs_cache.pop(job_id, None)

            logger.warning(f"Job {job_id} requires model '{e.model_id}' which is not installed")

        except Exception as e:
            # Mark failed, unless the job was cancelled while its step was running
            async with self._lock:
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now()
                    stage = "running" if job.plan is not None else "planning"
                    self._emit(job, "failed", **classify_failure(e, stage, self._current_step(job)))
                self._store.save(job)
                self._jobs_cache.pop(job_id, None)

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
        storage_mode = os.environ.get("JOB_STORAGE", "sqlite").lower()
        store: JobStore
        if storage_mode == "sqlite":
            from dta.config import JOBS_DB_PATH

            store = SQLiteJobStore(JOBS_DB_PATH)
            logger.info(f"Using SQLite job storage at {JOBS_DB_PATH}")
        else:
            store = MemoryJobStore()
            logger.info("Using in-memory job storage")
        # retention_hours=None: members' and operators' jobs are kept; guest jobs expire
        # after GUEST_SESSION_HOURS (the periodic cleanup in server/app.py).
        _job_queue = JobQueue(max_workers=3, max_queue_size=100, retention_hours=None, store=store)
    return _job_queue
