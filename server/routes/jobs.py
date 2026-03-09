"""Async job queue endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..jobs import JobStatus, get_job_queue
from ..schemas import JobSubmitRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["jobs"])


@router.post("/jobs")  # type: ignore[misc]
async def submit_job(req: JobSubmitRequest) -> JSONResponse:
    """Submit a new async job.

    The job will be queued and processed in the background.
    Use GET /v1/jobs/{job_id} to check status and retrieve results.
    """
    try:
        queue = get_job_queue()

        # Convert attachments to dict for storage
        attachments_dict = [att.model_dump() for att in req.attachments]
        logger.info(f"Job submit: {len(req.attachments)} attachments received: {attachments_dict}")

        job_id = await queue.submit_job(
            prompt=req.prompt, mode=req.mode, attachments=attachments_dict, context=req.context
        )

        job = await queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Job creation failed")

        return JSONResponse(job.to_dict(), status_code=202)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job submission failed: {e}") from e


@router.get("/jobs/{job_id}")  # type: ignore[misc]
async def get_job_status(job_id: str) -> JSONResponse:
    """Get job status and results.

    Returns job details including status, progress, plan, and results (if completed).
    """
    try:
        queue = get_job_queue()
        job = await queue.get_job(job_id)

        if not job:
            # Log available jobs for debugging
            available_jobs = list(queue._jobs.keys())
            logger.warning(f"Job {job_id} not found. Available jobs: {available_jobs}")
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JSONResponse(job.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e}") from e


@router.post("/jobs/{job_id}/cancel")  # type: ignore[misc]
async def cancel_job(job_id: str) -> JSONResponse:
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
        return JSONResponse(job.to_dict() if job else {"id": job_id, "cancelled": True})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}") from e


@router.get("/jobs")  # type: ignore[misc]
async def list_jobs(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> JSONResponse:
    """List jobs with optional filtering and pagination."""
    try:
        queue = get_job_queue()

        # Parse status filter
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
        total = len(queue._jobs)

        return JSONResponse(
            {
                "jobs": [job.to_dict() for job in jobs],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}") from e


@router.get("/queue/stats")  # type: ignore[misc]
async def get_queue_stats() -> JSONResponse:
    """Get job queue statistics."""
    try:
        queue = get_job_queue()
        stats = queue.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}") from e
