"""Async job queue endpoints."""

import asyncio
from collections.abc import AsyncIterator
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..job_events import KEEPALIVE_FRAME, TERMINAL_EVENTS, sse_event_frame
from ..jobs import JobQueue, JobStatus, get_job_queue
from ..schemas import (
    CancelJobResponse,
    JobListResponse,
    JobRecord,
    JobSubmitRequest,
    JobSummary,
    QueueStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["jobs"])


def _job_record(job_dict: dict[str, Any]) -> JobRecord:
    return JobRecord.model_validate(job_dict)  # type: ignore[no-any-return]


@router.post(  # type: ignore[misc]
    "/jobs",
    status_code=202,
    response_model=JobRecord,
    summary="Submit async job",
    response_description="Created job queued for background processing",
)
async def submit_job(req: JobSubmitRequest) -> JobRecord:
    """Submit a new async job.

    The job is queued and processed in the background.
    Poll ``GET /v1/jobs/{job_id}`` for status and results.
    """
    try:
        queue = get_job_queue()

        attachments_dict = [att.model_dump() for att in req.attachments]
        logger.info(f"Job submit: {len(req.attachments)} attachments received: {attachments_dict}")

        job_id = await queue.submit_job(
            prompt=req.prompt, mode=req.mode, attachments=attachments_dict, context=req.context
        )

        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Job creation failed")

        return _job_record(job.to_dict())
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"Invalid job response: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {e}") from e


@router.get(  # type: ignore[misc]
    "/jobs/{job_id}",
    response_model=JobRecord,
    summary="Get job status",
)
async def get_job_status(job_id: str) -> JobRecord:
    """Return job status, progress, plan, results (when completed) and the event log (``events``)."""
    try:
        queue = get_job_queue()
        job = await queue.get_job(job_id)

        if not job:
            available_jobs = await queue.get_job_ids()
            logger.warning(f"Job {job_id} not found. Available jobs: {available_jobs}")
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return _job_record(job.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e}") from e


# Seconds between keep-alive comments on an idle event stream.
KEEPALIVE_S = 15.0


async def _event_frames(
    queue: JobQueue, job_id: str, backlog: list[dict[str, Any]], live: asyncio.Queue[dict[str, Any]] | None
) -> AsyncIterator[bytes]:
    """SSE frames: the backlog, then the live tail until the terminal event, with keep-alives."""
    try:
        for event in backlog:
            yield sse_event_frame(event)
        if live is None:
            return
        while True:
            try:
                event = await asyncio.wait_for(live.get(), timeout=KEEPALIVE_S)
            except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
                yield KEEPALIVE_FRAME
                continue
            yield sse_event_frame(event)
            if event["type"] in TERMINAL_EVENTS:
                return
    finally:
        if live is not None:
            queue.unsubscribe(job_id, live)


@router.get(  # type: ignore[misc]
    "/jobs/{job_id}/events",
    summary="Stream job events (SSE)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"}},
)
async def stream_job_events(
    job_id: str,
    after: int | None = Query(None, ge=0, description="Replay only events with a larger id (wins over Last-Event-ID)"),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream the job's events as ``text/event-stream``: the full log so far, then the live tail.

    Frames are ``id: <seq>``, ``event: <type>``, ``data: <json>``; an idle stream gets a
    ``: keep-alive`` comment every 15 s. The stream ends after ``done``, ``failed`` or
    ``cancelled``. To resume, send ``Last-Event-ID`` (browsers do on reconnect) or
    ``?after=<seq>``. Event payloads are documented in ``server/job_events.py``.
    """
    if after is None:
        try:
            after = max(int(last_event_id or 0), 0)
        except ValueError:
            after = 0
    queue = get_job_queue()
    subscription = queue.subscribe(job_id, after)
    if subscription is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    backlog, live = subscription
    return StreamingResponse(
        _event_frames(queue, job_id, backlog, live),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(  # type: ignore[misc]
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    summary="Cancel job",
)
async def cancel_job(job_id: str) -> CancelJobResponse:
    """Cancel a running or pending job.

    The status becomes ``cancelled`` immediately, but a running job stops only at the
    next step boundary: the step in progress (e.g. a subprocess) runs to completion,
    then the remaining steps are skipped and the job stays ``cancelled``.
    """
    try:
        queue = get_job_queue()
        cancelled = await queue.cancel_job(job_id)

        if not cancelled:
            job = await queue.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            raise HTTPException(status_code=400, detail=f"Job {job_id} cannot be cancelled (already finished)")

        job = await queue.get_job(job_id)
        payload = job.to_dict() if job else {"id": job_id, "cancelled": True}
        return CancelJobResponse.model_validate(payload)  # type: ignore[no-any-return]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}") from e


@router.get(  # type: ignore[misc]
    "/jobs",
    response_model=JobListResponse,
    summary="List jobs",
)
async def list_jobs(
    status: str | None = Query(None, description="Filter by status: pending, running, completed, failed, cancelled"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> JobListResponse:
    """List job summaries (no ``result``) with optional status filter and pagination."""
    try:
        queue = get_job_queue()

        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Must be one of: {', '.join(s.value for s in JobStatus)}",
                ) from None

        jobs = await queue.list_jobs(status=status_filter, limit=limit, offset=offset)
        total = await queue.get_total_count()

        return JobListResponse(
            jobs=[JobSummary.model_validate(j.to_dict()) for j in jobs],
            total=total,
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}") from e


@router.get(  # type: ignore[misc]
    "/queue/stats",
    response_model=QueueStatsResponse,
    summary="Queue statistics",
)
async def get_queue_stats() -> QueueStatsResponse:
    """Return job queue statistics (counts by status, workers, etc.)."""
    try:
        queue = get_job_queue()
        return QueueStatsResponse.model_validate(queue.get_stats())  # type: ignore[no-any-return]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}") from e
