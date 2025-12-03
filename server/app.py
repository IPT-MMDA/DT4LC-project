import base64
from collections.abc import AsyncIterator
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

# Configure logging BEFORE any loggers are used
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from dta.config import UPLOADS_PATH
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.metrics import get_metrics_collector
from dta.dti.models.registry import get_model_registry
from dta.dti.registry import load_registry
from dta.dti.schemas import ChatRequest as COEChatRequest

from .jobs import JobStatus, get_job_queue
from .model_routes import router as model_router
from .schemas import ChatRequest, JobSubmitRequest

app = FastAPI(title="DT4LC API", version="1.0.0")
HEARTBEAT_SECS = 15

# Register routers
app.include_router(model_router)

# CORS configuration
# In production, restrict to specific origins via CORS_ORIGINS environment variable
cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory from centralized config (resources/.cache/uploads/)
UPLOAD_DIR = UPLOADS_PATH


def sse_frame(payload: dict[str, Any]) -> bytes:
    """Format payload as Server-Sent Event frame.

    Args:
        payload: Data to encode as JSON

    Returns:
        Bytes with SSE format: "data: <json>\\n\\n"
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


@app.get("/v1/health")  # type: ignore[misc]
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"ok": True, "service": "DT4LC", "version": "1.0.0"}


@app.post("/v1/plan")  # type: ignore[misc]
async def create_plan(req: ChatRequest) -> JSONResponse:
    """Generate an execution plan from a user prompt.

    This endpoint uses the COE to analyze the prompt and generate a plan
    without executing it.
    """
    try:
        # Convert server ChatRequest to COE ChatRequest
        # For now, use the last message as prompt
        if not req.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        prompt = req.messages[-1].content
        coe_req = COEChatRequest(prompt=prompt, attachments=[])

        # Orchestrate (generate plan)
        result = orchestrate(coe_req)

        if not result.get("ok"):
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": result.get("error", "Plan generation failed"),
                    "candidate": result.get("candidate"),
                },
            )

        return JSONResponse(
            {
                "ok": True,
                "plan": result["plan"],
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {e}") from e


@app.post("/v1/execute")  # type: ignore[misc]
async def execute_plan(req: ChatRequest) -> JSONResponse:
    """Generate and execute a pipeline plan.

    This is the main endpoint that combines COE planning with DTA execution.
    """
    try:
        # Convert server ChatRequest to COE ChatRequest
        if not req.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        prompt = req.messages[-1].content
        coe_req = COEChatRequest(prompt=prompt, attachments=[])

        # Step 1: Generate plan via COE
        orch_result = orchestrate(coe_req)

        if not orch_result.get("ok"):
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": orch_result.get("error", "Plan generation failed"),
                    "candidate": orch_result.get("candidate"),
                },
            )

        plan_dict = orch_result["plan"]

        # Step 2: Execute plan via DTA
        from dta.dti.schemas import ExecutionPlan

        plan = ExecutionPlan(**plan_dict)
        executor = PipelineExecutor()

        progress_events: list[dict[str, Any]] = []

        def on_progress(event: dict[str, Any]) -> None:
            progress_events.append(event)

        exec_result = executor.execute(plan, on_progress=on_progress)

        return JSONResponse(
            {
                "ok": True,
                "plan": plan_dict,
                "result": exec_result,
                "progress": progress_events,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}") from e


@app.post("/v1/chat")  # type: ignore[misc]
async def chat(req: ChatRequest) -> StreamingResponse:
    """Legacy chat endpoint - redirects to execute endpoint.

    For MVP, this simply calls execute and streams the result.
    In future, this can support true streaming execution.
    """

    async def gen() -> AsyncIterator[bytes]:
        try:
            # Convert to COE request
            if not req.messages:
                yield sse_frame({"error": "No messages provided"})
                yield sse_frame({"done": True})
                return

            prompt = req.messages[-1].content
            coe_req = COEChatRequest(prompt=prompt, attachments=[])

            # Generate plan
            yield sse_frame({"event": "planning", "message": "Generating execution plan..."})

            orch_result = orchestrate(coe_req)

            if not orch_result.get("ok"):
                yield sse_frame(
                    {
                        "error": orch_result.get("error", "Planning failed"),
                        "candidate": orch_result.get("candidate"),
                    }
                )
                yield sse_frame({"done": True})
                return

            yield sse_frame({"event": "plan_ready", "plan": orch_result["plan"]})

            # Execute plan
            from dta.dti.schemas import ExecutionPlan

            plan = ExecutionPlan(**orch_result["plan"])
            executor = PipelineExecutor()

            def on_progress(event: dict[str, Any]) -> None:
                # Can't directly yield from callback, so we'll skip for now
                pass

            yield sse_frame({"event": "executing", "message": "Running pipeline..."})

            exec_result = executor.execute(plan, on_progress=on_progress)

            yield sse_frame({"event": "complete", "result": exec_result})
            yield sse_frame({"done": True})

        except Exception as e:
            yield sse_frame({"error": str(e)})
            yield sse_frame({"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/upload")  # type: ignore[misc]
async def upload_geotiff(file: UploadFile = File) -> JSONResponse:
    # Basic checks
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Please upload a .tif/.tiff GeoTIFF.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Save file to temp directory
    import uuid

    file_id = str(uuid.uuid4())[:8]
    saved_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    saved_path.write_bytes(raw)

    # Read in-memory with rasterio for preview
    try:
        with MemoryFile(raw) as mem, mem.open() as src:
            # Read first band as masked array
            band1 = src.read(1, masked=True)  # (H, W) masked ndarray
            h, w = band1.shape
            bounds = src.bounds  # left, bottom, right, top
            crs = src.crs.to_string() if src.crs else None

            # Simple percentile stretch to 8-bit for preview
            # Convert to float FIRST, then fill with NaN (can't fill uint8 with NaN)
            data = band1.astype("float64").filled(np.nan)
            finite = np.isfinite(data)
            if not finite.any():
                raise HTTPException(status_code=400, detail="All pixels are nodata.")

            # percentiles on finite pixels only
            p2, p98 = np.percentile(data[finite], [2, 98])
            if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                p2, p98 = float(np.nanmin(data[finite])), float(np.nanmax(data[finite]))

            # Handle edge case where p2 == p98
            if p98 - p2 < 1e-10:
                scaled = np.zeros_like(data)
            else:
                scaled = (data - p2) / (p98 - p2)

            # Replace NaN/inf with 0 BEFORE converting to uint8
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
            scaled = np.clip(scaled, 0.0, 1.0)
            scaled = (scaled * 255.0 + 0.5).astype("uint8")

            # Convert to PNG (grayscale, mode "L")
            img = Image.fromarray(scaled, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return JSONResponse(
            {
                "id": file_id,
                "filename": file.filename,
                "path": str(saved_path),  # File path for use in execution
                "size": [int(w), int(h)],
                "crs": crs,
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "preview_png_base64": b64,  # data:image/png;base64,<this>
            }
        )
    except Exception as e:
        # Clean up saved file on error
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF: {e}") from e


@app.get("/v1/capabilities")  # type: ignore[misc]
async def list_capabilities() -> JSONResponse:
    """List all available components from the registry.

    Returns models, algorithms, and other registered components.
    """
    try:
        from dta.dti.registry import load_registry

        registry = load_registry()
        return JSONResponse(
            {
                "version": registry.version,
                "types": registry.types,
                "instances": [item.model_dump() for item in registry.instances],
                "count": len(registry.instances),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load registry: {e}") from e


@app.get("/v1/models")  # type: ignore[misc]
async def list_models() -> JSONResponse:
    """List all registered models from the model registry.

    Returns model information including requirements, availability,
    descriptions, author info, and source URLs.
    """
    try:
        registry = get_model_registry()
        models = []

        # List ALL models from Python registry
        for model_id in registry.list_all():
            req = registry.check_requirements(model_id)
            models.append(req)

        # Also include hosted models from YAML registry (models with integration field)
        try:
            yaml_registry = load_registry()
            for item in yaml_registry.instances:
                if item.kind == "model" and item.integration:
                    models.append(
                        {
                            "model_id": item.id,
                            "name": item.id.split("/")[-1].replace("-", " ").title(),
                            "description": item.description or "",
                            "author": item.metadata.get("author", ""),
                            "source_url": item.integration.url,
                            "available": item.integration.status == "active",
                            "missing_requirements": item.integration.requires
                            if item.integration.status == "planned"
                            else [],
                            "gpu_required": False,
                            "integration_type": item.integration.type,
                            "integration_status": item.integration.status,
                            "keywords": item.keywords,
                            "hosting": item.metadata.get("hosting", "external"),
                            "team": item.metadata.get("team", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to load hosted models from YAML registry: {e}")

        return JSONResponse({"models": models, "count": len(models)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}") from e


@app.get("/v1/metrics")  # type: ignore[misc]
async def get_metrics() -> JSONResponse:
    """Get system metrics including execution and LLM stats."""
    try:
        collector = get_metrics_collector()
        stats = collector.get_stats()

        return JSONResponse(
            {
                "total_executions": stats.total_executions,
                "successful_executions": stats.successful_executions,
                "failed_executions": stats.failed_executions,
                "average_duration_seconds": stats.average_duration_seconds,
                "total_llm_calls": stats.total_llm_calls,
                "total_llm_tokens": stats.total_llm_tokens,
                "total_llm_cost": stats.total_llm_cost,
                "llm_by_provider": stats.llm_by_provider,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}") from e


# Async Job Endpoints


@app.on_event("startup")  # type: ignore[misc]
async def startup_event() -> None:
    """Start job queue on app startup."""
    queue = get_job_queue()
    await queue.start()


@app.on_event("shutdown")  # type: ignore[misc]
async def shutdown_event() -> None:
    """Stop job queue on app shutdown."""
    queue = get_job_queue()
    await queue.stop()


@app.post("/v1/jobs")  # type: ignore[misc]
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


@app.get("/v1/jobs/{job_id}")  # type: ignore[misc]
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


@app.post("/v1/jobs/{job_id}/cancel")  # type: ignore[misc]
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


@app.get("/v1/jobs")  # type: ignore[misc]
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


@app.get("/v1/queue/stats")  # type: ignore[misc]
async def get_queue_stats() -> JSONResponse:
    """Get job queue statistics."""
    try:
        queue = get_job_queue()
        stats = queue.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}") from e


@app.get("/v1/download")  # type: ignore[misc]
async def download_file(path: str) -> FileResponse:
    """Download a file from the server.

    Used for downloading generated outputs like GeoPackage files.

    Args:
        path: Path to the file to download

    Returns:
        File response with appropriate content type
    """
    file_path = Path(path)

    # Security: Only allow downloads from specific directories
    allowed_dirs = [
        Path(tempfile.gettempdir()) / "dt4lc_delineate",
        UPLOAD_DIR,
        Path(tempfile.gettempdir()),
    ]

    # Check if path is under an allowed directory
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            file_path.resolve().relative_to(allowed_dir.resolve())
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied: path not in allowed directories")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    # Determine content type based on extension
    suffix = file_path.suffix.lower()
    content_types = {
        ".gpkg": "application/geopackage+sqlite3",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".png": "image/png",
        ".json": "application/json",
    }
    media_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
