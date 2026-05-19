"""Async job queue endpoints."""

import logging

from fastapi import APIRouter, Query
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
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "internal_error",
                        "message": "Job creation failed",
                        "details": {},
                    },
                },
            )

        return JSONResponse(job.to_dict(), status_code=202)
    except RuntimeError as e:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": {
                    "code": "bad_request",
                    "message": str(e),
                    "details": {"type": "RuntimeError"},
                },
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": f"Job submission failed: {e}",
                    "details": {},
                },
            },
        )


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
            available_jobs = await queue.get_job_ids()
            logger.warning(f"Job {job_id} not found. Available jobs: {available_jobs}")
            return JSONResponse(
                status_code=404,
                content={
                    "ok": False,
                    "error": {
                        "code": "not_found",
                        "message": f"Job {job_id} not found",
                        "details": {"available_job_ids": available_jobs},
                    },
                },
            )

        return JSONResponse(job.to_dict())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": f"Failed to get job: {e}",
                    "details": {},
                },
            },
        )


@router.post("/jobs/{job_id}/cancel")  # type: ignore[misc]
async def cancel_job(job_id: str) -> JSONResponse:
    """Cancel a running or pending job."""
    try:
        queue = get_job_queue()
        cancelled = await queue.cancel_job(job_id)

        if not cancelled:
            job = await queue.get_job(job_id)
            if not job:
                return JSONResponse(
                    status_code=404,
                    content={
                        "ok": False,
                        "error": {
                            "code": "not_found",
                            "message": f"Job {job_id} not found",
                            "details": {},
                        },
                    },
                )
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": {
                        "code": "bad_request",
                        "message": f"Job {job_id} cannot be cancelled (already finished)",
                        "details": {"state": job.status.value},
                    },
                },
            )

        job = await queue.get_job(job_id)
        return JSONResponse(job.to_dict() if job else {"id": job_id, "cancelled": True})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": f"Failed to cancel job: {e}",
                    "details": {},
                },
            },
        )


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
                return JSONResponse(
                    status_code=400,
                    content={
                        "ok": False,
                        "error": {
                            "code": "validation_error",
                            "message": f"Invalid status: {status}. \
                                        Must be one of: {', '.join(s.value for s in JobStatus)}",
                            "details": {"provided_status": status, "allowed_statuses": [s.value for s in JobStatus]},
                        },
                    },
                )

        jobs = await queue.list_jobs(status=status_filter, limit=limit, offset=offset)
        total = await queue.get_total_count()

        return JSONResponse(
            {
                "jobs": [job.to_dict() for job in jobs],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": f"Failed to list jobs: {e}",
                    "details": {},
                },
            },
        )


@router.get("/queue/stats")  # type: ignore[misc]
async def get_queue_stats() -> JSONResponse:
    """Get job queue statistics."""
    try:
        queue = get_job_queue()
        stats = queue.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": f"Failed to get stats: {e}",
                    "details": {},
                },
            },
        )
