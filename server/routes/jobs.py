"""Async job queue endpoints."""

import asyncio
from collections.abc import AsyncIterator
import functools
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from dta.dti.coe.plan_validator import PlanError
from dta.dti.schemas import ExecutionPlan

from .. import library
from ..job_events import KEEPALIVE_FRAME, TERMINAL_EVENTS, sse_event_frame
from ..jobs import JobConflictError, JobQueue, JobStatus, get_job_queue
from ..schemas import (
    CancelJobResponse,
    JobConfirmRequest,
    JobDeleteResponse,
    JobListResponse,
    JobRecord,
    JobSubmitRequest,
    JobSummary,
    QueueStatsResponse,
)
from ..sessions import CurrentSession, Session, enforce_analysis_quota, release_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["jobs"])

REGION_IGNORED_NOTE = "Files are attached, so the analysis reads them and doesn't fetch this region."


def _job_record(job_dict: dict[str, Any]) -> JobRecord:
    return JobRecord.model_validate(job_dict)  # type: ignore[no-any-return]


async def _ensure_visible(queue: JobQueue, job_id: str, session: Session) -> None:
    """404, as for an unknown job, unless ``session`` may see ``job_id`` (its owner, or an operator)."""
    if not session.operator and not session.can_see(await queue.get_owner(job_id)):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@router.post(  # type: ignore[misc]
    "/jobs",
    status_code=202,
    response_model=JobRecord,
    summary="Submit async job",
    response_description="Created job queued for background processing",
)
async def submit_job(req: JobSubmitRequest, session: CurrentSession) -> JobRecord:
    """Submit a new async job.

    The job is queued and processed in the background.
    Poll ``GET /v1/jobs/{job_id}`` for status and results.
    A guest beyond ``GUEST_ANALYSES_PER_HOUR`` gets 429 with code ``GUEST_QUOTA``.

    When planning composes a plan that is not a single built-in analysis (band math, or a new
    combination of steps), the job stops at status ``proposal`` with a ``proposal`` event
    ``{text, plan}`` and waits for ``POST /v1/jobs/{job_id}/confirm``; a proposal does not count
    against the guest quota, its confirmation does.

    ``region`` (a region sent from the map) is fetched from Earth Engine when no file is attached;
    the job echoes it with ``bounds``, ``area_km2`` and the chip ``line``. Attached files win: the
    region is not fetched and its echo carries ``note``. Without Earth Engine credentials a region
    job fails at planning with ``EE_NOT_CONFIGURED``. A region job counts like any analysis. An
    invalid region is a 422.
    """
    counted_at = enforce_analysis_quota(session)
    try:
        queue = get_job_queue()

        attachments_dict = [att.model_dump() for att in req.attachments]
        logger.info(f"Job submit: {len(req.attachments)} attachments received: {attachments_dict}")

        region = None
        if req.region is not None:
            attached = any(att.path for att in req.attachments)
            region = req.region.echo(REGION_IGNORED_NOTE if attached else None)

        job_id = await queue.submit_job(
            prompt=req.prompt,
            mode=req.mode,
            attachments=attachments_dict,
            context=req.context,
            owner=session.owner,
            owner_role=session.role,
            uncount=functools.partial(release_analysis, session, counted_at),
            region=region,
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
async def get_job_status(job_id: str, session: CurrentSession) -> JobRecord:
    """Return job status, progress, plan, results (when completed) and the event log (``events``)."""
    try:
        queue = get_job_queue()
        job = await queue.get_job(job_id)

        if not job or not session.can_see(job.owner):
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
    session: CurrentSession,
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
    await _ensure_visible(queue, job_id, session)
    subscription = queue.subscribe(job_id, after)
    if subscription is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    backlog, live = subscription
    return StreamingResponse(
        _event_frames(queue, job_id, backlog, live),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete(  # type: ignore[misc]
    "/jobs/{job_id}",
    response_model=JobDeleteResponse,
    summary="Delete job",
)
async def delete_job(job_id: str, session: CurrentSession) -> JobDeleteResponse:
    """Delete a finished job (completed, failed, cancelled) or a proposal, with its output files.

    409 for a pending or running job (cancel it first); 404 for an unknown job or another
    session's. A deleted proposal's open event stream ends with ``cancelled {reason: "deleted"}``.
    """
    queue = get_job_queue()
    await _ensure_visible(queue, job_id, session)
    try:
        deleted = await queue.delete_job(job_id)
    except JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobDeleteResponse(id=job_id)


@router.post(  # type: ignore[misc]
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    summary="Cancel job",
)
async def cancel_job(job_id: str, session: CurrentSession) -> CancelJobResponse:
    """Cancel a running or pending job.

    The status becomes ``cancelled`` immediately, but a running job stops only at the
    next step boundary: the step in progress (e.g. a subprocess) runs to completion,
    then the remaining steps are skipped and the job stays ``cancelled``.
    """
    try:
        queue = get_job_queue()
        await _ensure_visible(queue, job_id, session)
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


@router.post(  # type: ignore[misc]
    "/jobs/{job_id}/confirm",
    status_code=202,
    response_model=JobRecord,
    summary="Confirm a proposed plan",
    response_description="The job, queued to run the confirmed plan",
)
async def confirm_job(job_id: str, session: CurrentSession, req: JobConfirmRequest | None = None) -> JobRecord:
    """Run a job's proposal (status ``proposal``): its plan, or ``plan`` from the body (the user's edit).

    The job keeps its id and event stream: ``planning`` ready (``composed: true``), the steps,
    then ``done`` or ``failed``. The confirmation counts against the guest quota (429
    ``GUEST_QUOTA``). 404 for an unknown job or another session's, 409 if the job is not a
    proposal (e.g. confirmed already), 422 with the validator's message for an invalid edited plan.
    """
    queue = get_job_queue()
    await _ensure_visible(queue, job_id, session)
    plan = None
    if req is not None and req.plan is not None:
        try:
            plan = ExecutionPlan.model_validate(req.plan)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid plan: {e.error_count()} error(s): {e}") from e
    try:
        job = await queue.confirm_job(job_id, plan, count=functools.partial(enforce_analysis_quota, session))
    except JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PlanError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_record(job.to_dict())


@router.get(  # type: ignore[misc]
    "/jobs/{job_id}/statistics.csv",
    summary="Download a job's statistics as CSV",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}, "description": "CSV with a header row"}},
)
async def job_statistics_csv(job_id: str, session: CurrentSession) -> Response:
    """The main analysis's statistics as a CSV attachment ``<slug of the card title>-statistics.csv``.

    The class table (``class,area_km2,percent,pixels[,mean_change]``) for change detection, band-math
    change, land cover and snow; one row per band (``band,min,max,mean,std,median,p5,p25,p75,p95,
    valid_pixels,nodata_pixels``) for raster statistics; one row of index statistics
    (``index,min,max,mean,…``) for a single index. 404 for an unknown job, another session's, or a
    job without statistics (not finished, failed, or an analysis that reports none).
    """
    queue = get_job_queue()
    job = await queue.get_job(job_id)
    if not job or not session.can_see(job.owner):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    text = library.statistics_csv(job, queue.registry) if job.status == JobStatus.COMPLETED else None
    if text is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} has no statistics")
    name = library.slug(str((job.card or {}).get("title") or "")) or f"job-{job_id}"
    return Response(
        text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}-statistics.csv"'},
    )


@router.get(  # type: ignore[misc]
    "/jobs",
    response_model=JobListResponse,
    summary="List jobs",
)
async def list_jobs(
    session: CurrentSession,
    status: str | None = Query(
        None, description="Filter by status: pending, proposal, running, completed, failed, cancelled"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> JobListResponse:
    """List the caller's job summaries (no ``result``) with optional status filter and pagination.

    Operators see every job, including those from before v0.6.0 that have no owner.
    """
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

        owner = None if session.operator else session.owner
        jobs = await queue.list_jobs(status=status_filter, limit=limit, offset=offset, owner=owner)
        total = await queue.get_total_count(owner=owner, status=status_filter)

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
