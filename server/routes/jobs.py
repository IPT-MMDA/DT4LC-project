"""Async job queue endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from ..jobs import JobStatus, get_job_queue
from ..schemas import CancelJobResponse, JobListResponse, JobRecord, JobSubmitRequest, QueueStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["jobs"])


def _job_record(job_dict: dict) -> JobRecord:
    return JobRecord.model_validate(job_dict)


@router.post(
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


@router.get(
    "/jobs/{job_id}",
    response_model=JobRecord,
    summary="Get job status",
)
async def get_job_status(job_id: str) -> JobRecord:
    """Return job status, progress, plan, and results (when completed)."""
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


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    summary="Cancel job",
)
async def cancel_job(job_id: str) -> CancelJobResponse:
    """Cancel a running or pending job."""
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
        return CancelJobResponse.model_validate(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}") from e


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List jobs",
)
async def list_jobs(
    status: str | None = Query(None, description="Filter by status: pending, running, completed, failed, cancelled"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> JobListResponse:
    """List jobs with optional status filter and pagination."""
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
            jobs=[_job_record(j.to_dict()) for j in jobs],
            total=total,
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}") from e


@router.get(
    "/queue/stats",
    response_model=QueueStatsResponse,
    summary="Queue statistics",
)
async def get_queue_stats() -> QueueStatsResponse:
    """Return job queue statistics (counts by status, workers, etc.)."""
    try:
        queue = get_job_queue()
        return QueueStatsResponse.model_validate(queue.get_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}") from e
