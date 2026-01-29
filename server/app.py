import base64
from collections.abc import AsyncIterator
from datetime import datetime
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
import mercantile
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

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


@app.get("/v1/files")  # type: ignore[misc]
async def list_files() -> JSONResponse:
    """List all available GeoTIFF files (uploaded and exported).

    Returns:
        JSON with list of files from both uploads and exports directories
    """
    try:
        from dta.config import CACHE_PATH
        import rasterio

        files = []

        # Scan uploads directory
        upload_files = list(UPLOAD_DIR.glob("*.tif")) + list(UPLOAD_DIR.glob("*.tiff"))
        for file_path in upload_files:
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string() if src.crs else None
                    width, height = src.width, src.height

                    files.append({
                        "id": file_path.stem,
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": [width, height],
                        "crs": crs,
                        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        "size_bytes": file_path.stat().st_size,
                        "source": "upload",
                        "modified": file_path.stat().st_mtime
                    })
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

        # Scan exports directory
        export_dir = CACHE_PATH / "gee_exports"
        if export_dir.exists():
            export_files = list(export_dir.glob("*.tif")) + list(export_dir.glob("*.tiff"))
            for file_path in export_files:
                try:
                    with rasterio.open(file_path) as src:
                        bounds = src.bounds
                        crs = src.crs.to_string() if src.crs else None
                        width, height = src.width, src.height

                        files.append({
                            "id": file_path.stem,
                            "filename": file_path.name,
                            "path": str(file_path),
                            "size": [width, height],
                            "crs": crs,
                            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                            "size_bytes": file_path.stat().st_size,
                            "source": "export",
                            "modified": file_path.stat().st_mtime
                        })
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)

        logger.info(f"Listed {len(files)} files ({len([f for f in files if f['source'] == 'upload'])} uploads, {len([f for f in files if f['source'] == 'export'])} exports)")

        return JSONResponse({
            "ok": True,
            "files": files,
            "count": len(files)
        })

    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail=str(e)) from e


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


def apply_colormap(data: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply colormap to single-band data, returning RGB."""
    from matplotlib import cm
    import matplotlib.pyplot as plt

    # Get the colormap
    cmap = plt.get_cmap(colormap_name)

    # Normalize data to 0-1 range
    data_normalized = data / 255.0

    # Apply colormap (returns RGBA)
    rgba = cmap(data_normalized)

    # Convert to RGB uint8 (drop alpha channel)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    return rgb


def detect_data_type(filename: str) -> str:
    """Detect data type from filename."""
    filename_lower = filename.lower()
    if 'ndvi' in filename_lower:
        return 'ndvi'
    elif 'ndwi' in filename_lower:
        return 'ndwi'
    elif 'ndsi' in filename_lower:
        return 'ndsi'
    elif 'lulc' in filename_lower or 'land' in filename_lower:
        return 'lulc'
    elif 'change' in filename_lower:
        return 'change'
    return 'generic'


@app.get("/v1/tiles/{z}/{x}/{y}")  # type: ignore[misc]
async def get_tile(
    z: int,
    x: int,
    y: int,
    path: str = Query(..., description="Path to GeoTIFF file")
) -> StreamingResponse:
    """Serve GeoTIFF as map tiles in XYZ format.

    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
        path: Path to the GeoTIFF file

    Returns:
        PNG image tile
    """
    try:
        # Get tile bounds in Web Mercator (EPSG:3857)
        tile = mercantile.Tile(x, y, z)
        tile_bounds_wgs84 = mercantile.bounds(tile)

        # Open GeoTIFF
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"GeoTIFF file not found: {path}")

        # Detect data type from filename
        data_type = detect_data_type(file_path.name)

        with rasterio.open(str(file_path)) as src:
            # Transform tile bounds from WGS84 to source CRS
            src_bounds = transform_bounds(
                'EPSG:4326', src.crs,
                tile_bounds_wgs84.west, tile_bounds_wgs84.south,
                tile_bounds_wgs84.east, tile_bounds_wgs84.north
            )

            # Read window from GeoTIFF
            window = from_bounds(*src_bounds, src.transform)

            # Check if window is valid (intersects with raster)
            if window.width <= 0 or window.height <= 0:
                # Return transparent tile for areas outside the raster
                img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                return StreamingResponse(img_bytes, media_type="image/png")

            # Read data based on band count
            if src.count >= 3:
                # RGB data - read first 3 bands
                data = src.read([1, 2, 3], window=window, out_shape=(3, 256, 256))
                data = np.transpose(data, (1, 2, 0))  # CHW to HWC

                # Handle nodata
                if src.nodata is not None:
                    mask = data[:, :, 0] == src.nodata
                    data[mask] = 0

                # Normalize to 0-255
                if data.max() > 255 or data.dtype != np.uint8:
                    data_finite = data[np.isfinite(data)]
                    if len(data_finite) > 0:
                        p2, p98 = np.percentile(data_finite, [2, 98])
                        data = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                    else:
                        data = np.zeros((256, 256, 3), dtype=np.uint8)

                img = Image.fromarray(data, mode='RGB')
            else:
                # Single band data - read first band
                data = src.read(1, window=window, out_shape=(256, 256))

                # Handle nodata values
                if src.nodata is not None:
                    nodata_mask = data == src.nodata
                    data = np.where(nodata_mask, np.nan, data)

                # Apply data-type-specific normalization and colormapping
                data_finite = data[np.isfinite(data)]

                if len(data_finite) == 0:
                    # All nodata - return transparent tile
                    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    return StreamingResponse(img_bytes, media_type="image/png")

                # Normalize based on data type
                if data_type in ['ndvi', 'ndwi', 'ndsi']:
                    # Spectral indices typically range from -1 to 1
                    # Clip to expected range
                    data_clipped = np.clip(data, -1, 1)
                    # Scale to 0-255
                    data_normalized = ((data_clipped + 1) / 2 * 255).astype(np.uint8)

                    # Apply colormap
                    if data_type == 'ndvi':
                        # Green-yellow-red colormap for vegetation
                        rgb_data = apply_colormap(data_normalized, 'RdYlGn')
                    elif data_type == 'ndwi':
                        # Blue colormap for water
                        rgb_data = apply_colormap(data_normalized, 'Blues')
                    elif data_type == 'ndsi':
                        # Cool colormap for snow/ice
                        rgb_data = apply_colormap(data_normalized, 'cool')
                    else:
                        rgb_data = apply_colormap(data_normalized, 'viridis')

                    # Handle nodata in RGB
                    if src.nodata is not None:
                        rgb_data[nodata_mask] = [0, 0, 0]

                    img = Image.fromarray(rgb_data, mode='RGB')
                else:
                    # Generic single-band data - use percentile stretch
                    p2, p98 = np.percentile(data_finite, [2, 98])
                    data_normalized = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

                    # Apply viridis colormap for better visualization
                    rgb_data = apply_colormap(data_normalized, 'viridis')

                    # Handle nodata
                    if src.nodata is not None:
                        rgb_data[nodata_mask] = [0, 0, 0]

                    img = Image.fromarray(rgb_data, mode='RGB')

            # Save as PNG
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            return StreamingResponse(img_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving tile {z}/{x}/{y} for {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tile: {str(e)}") from e


@app.post("/v1/gee/sentinel2")  # type: ignore[misc]
async def fetch_sentinel2_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi, or ndsi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %")
) -> JSONResponse:
    """Fetch Sentinel-2 data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi, ndsi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_sentinel2 import (
            fetch_sentinel2_composite,
            fetch_sentinel2_indices
        )

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

        # Fetch data based on type
        if data_type == 'rgb':
            result = fetch_sentinel2_composite(bbox, start_date, end_date, cloud_cover_max)
        elif data_type in ['ndvi', 'ndwi', 'ndsi']:
            result = fetch_sentinel2_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get('ok'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Sentinel-2 data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@app.post("/v1/gee/modis")  # type: ignore[misc]
async def fetch_modis_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %")
) -> JSONResponse:
    """Fetch MODIS Terra/Aqua data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_modis import (
            fetch_modis_composite,
            fetch_modis_indices
        )

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

        # Fetch data based on type
        if data_type == 'rgb':
            result = fetch_modis_composite(bbox, start_date, end_date, None, cloud_cover_max)
        elif data_type in ['ndvi', 'ndwi']:
            result = fetch_modis_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get('ok'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching MODIS data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@app.post("/v1/gee/landsat")  # type: ignore[misc]
async def fetch_landsat_data(
    bbox: list[float],
    start_date: str,
    end_date: str,
    data_type: str = Query("rgb", description="Data type: rgb, ndvi, ndwi, ndsi"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %")
) -> JSONResponse:
    """Fetch Landsat 8/9 data from Google Earth Engine.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_type: Type of data to fetch (rgb, ndvi, ndwi, ndsi)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with tile URL and metadata
    """
    try:
        from dta.dti.data_sources.gee_landsat import (
            fetch_landsat_composite,
            fetch_landsat_indices
        )

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

        # Fetch data based on type
        if data_type == 'rgb':
            result = fetch_landsat_composite(bbox, start_date, end_date, None, cloud_cover_max)
        elif data_type in ['ndvi', 'ndwi', 'ndsi']:
            result = fetch_landsat_indices(bbox, start_date, end_date, data_type, cloud_cover_max)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")

        if not result.get('ok'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching Landsat data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}") from e


@app.post("/v1/gee/bulk-fetch")  # type: ignore[misc]
async def bulk_fetch_datasets(
    request: dict
) -> JSONResponse:
    """Bulk fetch multiple bands and indices for pre/post periods.

    Request body should contain:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        dataset_id: Dataset ID ('sentinel-2', 'modis', 'landsat-8')
        bands: List of band IDs to fetch
        indices: List of indices to fetch ('ndvi', 'ndwi', 'ndsi')
        pre_start: Pre-period start date YYYY-MM-DD
        pre_end: Pre-period end date YYYY-MM-DD
        post_start: Post-period start date YYYY-MM-DD (optional if use_now=True)
        post_end: Post-period end date YYYY-MM-DD (optional if use_now=True)
        cloud_cover_max: Maximum cloud cover percentage (0-100)
        use_now: If True, calculate post period as last 7 days

    Returns:
        JSON with list of layer metadata for bulk import
    """
    try:
        from dta.dti.data_sources.gee_bulk_fetch import bulk_fetch_data
        from datetime import datetime, timedelta

        # Extract parameters from request body
        bbox = request.get('bbox', [])
        dataset_id = request.get('dataset_id', '')
        bands = request.get('bands', [])
        indices = request.get('indices', [])
        pre_start = request.get('pre_start', '')
        pre_end = request.get('pre_end', '')
        post_start = request.get('post_start')
        post_end = request.get('post_end')
        cloud_cover_max = request.get('cloud_cover_max', 20.0)
        use_now = request.get('use_now', False)

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        # Validate dataset_id
        if dataset_id not in ['sentinel-2', 'modis', 'landsat-8']:
            raise HTTPException(status_code=400, detail=f"Invalid dataset_id: {dataset_id}")

        # Validate at least one band or index selected
        if not bands and not indices:
            raise HTTPException(status_code=400, detail="Must select at least one band or index")

        # Validate date formats
        try:
            datetime.strptime(pre_start, '%Y-%m-%d')
            datetime.strptime(pre_end, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Pre-period dates must be in YYYY-MM-DD format")

        # Calculate post period if use_now is True
        if use_now:
            today = datetime.now().date()
            seven_days_ago = today - timedelta(days=7)
            post_start = seven_days_ago.strftime('%Y-%m-%d')
            post_end = today.strftime('%Y-%m-%d')
        else:
            if not post_start or not post_end:
                raise HTTPException(status_code=400, detail="Post-period dates required when use_now=False")
            try:
                datetime.strptime(post_start, '%Y-%m-%d')
                datetime.strptime(post_end, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Post-period dates must be in YYYY-MM-DD format")

        # Build period dictionaries
        pre_period = {'start': pre_start, 'end': pre_end}
        post_period = {'start': post_start, 'end': post_end}

        # Execute bulk fetch
        result = bulk_fetch_data(
            dataset_id=dataset_id,
            bbox=bbox,
            bands=bands,
            indices=indices,
            pre_period=pre_period,
            post_period=post_period,
            cloud_cover_max=cloud_cover_max
        )

        if not result.get('ok'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk fetch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk fetch data: {str(e)}") from e


@app.get("/v1/gee/datasets")  # type: ignore[misc]
async def list_available_datasets() -> JSONResponse:
    """List all available GEE datasets with metadata.

    Returns:
        JSON with dataset configurations
    """
    datasets = {
        'sentinel-2': {
            'id': 'sentinel-2',
            'name': 'Sentinel-2',
            'description': 'ESA Copernicus Sentinel-2 Surface Reflectance (10m resolution)',
            'collection': 'COPERNICUS/S2_SR_HARMONIZED',
            'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'supported_indices': ['ndvi', 'ndwi', 'ndsi'],
            'spatial_resolution': '10m',
            'temporal_resolution': '5 days'
        },
        'modis': {
            'id': 'modis',
            'name': 'MODIS Terra/Aqua',
            'description': 'NASA MODIS Surface Reflectance 8-Day Composite (250-500m resolution)',
            'collection': 'MODIS/006/MOD09A1',
            'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04'],
            'supported_indices': ['ndvi', 'ndwi'],
            'spatial_resolution': '250-500m',
            'temporal_resolution': '8 days'
        },
        'landsat-8': {
            'id': 'landsat-8',
            'name': 'Landsat 8/9',
            'description': 'USGS Landsat 8/9 Surface Reflectance (30m resolution)',
            'collection': 'LANDSAT/LC08/C02/T1_L2',
            'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            'supported_indices': ['ndvi', 'ndwi', 'ndsi'],
            'spatial_resolution': '30m',
            'temporal_resolution': '16 days'
        }
    }

    return JSONResponse({'ok': True, 'datasets': datasets})


@app.post("/v1/gee/layers/persist")  # type: ignore[misc]
async def persist_layer_metadata(request: dict[str, Any]) -> JSONResponse:
    """Persist layer metadata for future export and chat context.

    Request body should contain:
        layer_id: Unique layer identifier
        layer_name: Display name of the layer
        dataset_id: Dataset ID ('sentinel-2', 'modis', 'landsat-8')
        bands: List of band IDs
        indices: List of spectral indices
        period: Period label ('pre' or 'post')
        bbox: Bounding box [minX, minY, maxX, maxY]
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        tile_url: GEE tile URL for visualization
        cloud_cover_max: Maximum cloud cover percentage

    Returns:
        JSON with success status
    """
    try:
        from server.layer_metadata_store import save_layer_metadata
        from datetime import datetime

        # Extract fields from request
        layer_id = request.get('layer_id')
        if not layer_id:
            raise HTTPException(status_code=400, detail="layer_id is required")

        metadata = {
            'layer_id': layer_id,
            'layer_name': request.get('layer_name', ''),
            'dataset_id': request.get('dataset_id', ''),
            'bands': request.get('bands', []),
            'indices': request.get('indices', []),
            'period': request.get('period', ''),
            'bbox': request.get('bbox', []),
            'start_date': request.get('start_date', ''),
            'end_date': request.get('end_date', ''),
            'tile_url': request.get('tile_url', ''),
            'cloud_cover_max': request.get('cloud_cover_max', 20.0),
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"Persisting layer metadata: {layer_id}")
        save_layer_metadata(layer_id, metadata)

        return JSONResponse({'ok': True, 'layer_id': layer_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to persist layer metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/gee/layers/{layer_id}/export")  # type: ignore[misc]
async def export_layer_for_analysis(
    layer_id: str,
    scale: int = Query(10, description="Resolution in meters"),
    format: str = Query("geotiff", description="Export format"),
    source: str = Query("auto", description="Data source: gee, microsoft, or auto"),
    name: str = Query(None, description="Custom export name (optional)")
) -> JSONResponse:
    """Export a layer to GeoTIFF for AI analysis.

    This retrieves layer metadata and exports from the chosen source:
    - gee: Google Earth Engine (fast, 32MB limit)
    - microsoft: Microsoft Planetary Computer (no limits)
    - auto: Automatically choose based on size

    Args:
        layer_id: Layer ID to export
        scale: Resolution in meters (default: 10m for Sentinel-2)
        format: Export format (currently only 'geotiff' supported)
        source: Data source (gee, microsoft, auto)

    Returns:
        JSON with export status and attachment object
    """
    try:
        from server.layer_metadata_store import get_layer_metadata
        from dta.dti.data_sources.gee_export import (
            export_gee_image_to_geotiff,
            estimate_export_size,
            create_attachment_from_export
        )
        from dta.dti.data_sources import gee_sentinel2, gee_modis, gee_landsat

        # Initialize GEE using the proper initialization function
        if not gee_sentinel2.initialize_gee():
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Google Earth Engine. Check GEE_PROJECT_ID environment variable."
            )

        # Retrieve layer metadata
        metadata = get_layer_metadata(layer_id)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Layer {layer_id} not found")

        dataset_id = metadata['dataset_id']
        bbox = metadata['bbox']
        bands = metadata['bands']
        indices = metadata['indices']
        start_date = metadata['start_date']
        end_date = metadata['end_date']
        cloud_cover_max = metadata.get('cloud_cover_max', 20.0)
        period = metadata.get('period', 'pre')

        logger.info(f"Exporting layer {layer_id} ({dataset_id}, {bands or indices})")

        # Estimate export size
        num_bands = len(bands) if bands else len(indices)
        size_estimate = estimate_export_size(bbox, scale, num_bands)

        # Determine data source based on user choice and size
        if source == "auto":
            # Auto-select: GEE for small, Microsoft for large
            chosen_source = "gee" if size_estimate['can_use_direct_download'] else "microsoft"
            logger.info(
                f"Auto-selecting source: {chosen_source} "
                f"(size estimate: {size_estimate['estimated_size_mb']:.1f} MB)"
            )
        else:
            chosen_source = source
            logger.info(f"User selected source: {chosen_source}")

        # Generate filename: name_source_bands/indices_period_resolution
        if name:
            # Use custom name if provided
            base_name = name.lower().replace(' ', '_')
        else:
            # Generate from metadata
            base_name = f"{dataset_id.replace('-', '_')}"

        # Add source
        source_suffix = "mpc" if chosen_source == "microsoft" else "gee"

        # Add bands or indices
        if indices:
            bands_suffix = "_".join(indices)
        elif bands:
            bands_suffix = "_".join(bands)
        else:
            bands_suffix = "rgb"

        # Add period if available
        period_suffix = f"_{period}" if period else ""

        # Add resolution
        resolution_suffix = f"{scale}m"

        # Combine all parts
        export_filename = f"{base_name}_{source_suffix}_{bands_suffix}{period_suffix}_{resolution_suffix}"

        logger.info(f"Export filename: {export_filename}")

        # Reconstruct GEE image based on dataset
        if indices:
            # Export spectral index
            index_type = indices[0]  # Take first index if multiple

            if dataset_id == 'sentinel-2':
                result = gee_sentinel2.fetch_sentinel2_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            elif dataset_id == 'modis':
                result = gee_modis.fetch_modis_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            elif dataset_id == 'landsat-8':
                result = gee_landsat.fetch_landsat_indices(
                    bbox, start_date, end_date, index_type, cloud_cover_max, return_image=True
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset_id}")

            image = result.get('image')
            export_bands = ['index']  # The index band is renamed to 'index' in GEE functions

        elif bands:
            # Export band composite
            if dataset_id == 'sentinel-2':
                result = gee_sentinel2.fetch_sentinel2_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            elif dataset_id == 'modis':
                result = gee_modis.fetch_modis_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            elif dataset_id == 'landsat-8':
                result = gee_landsat.fetch_landsat_composite(
                    bbox, start_date, end_date, bands, cloud_cover_max, return_image=True
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset_id}")

            image = result.get('image')
            export_bands = bands

        else:
            raise HTTPException(status_code=400, detail="Layer has no bands or indices to export")

        # Export from chosen source
        if chosen_source == "microsoft":
            # Use Microsoft Planetary Computer
            from dta.dti.data_sources.planetary_computer import export_sentinel2_from_mpc

            logger.info(f"Exporting from Microsoft Planetary Computer")

            export_result = export_sentinel2_from_mpc(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                bands=bands if bands else ['B04', 'B03', 'B02'],  # Default RGB
                scale=scale,
                cloud_cover_max=cloud_cover_max,
                filename=export_filename
            )

        else:
            # Use Google Earth Engine
            if not image:
                raise HTTPException(status_code=500, detail="Failed to reconstruct GEE image")

            logger.info(f"Exporting from Google Earth Engine")

            # Check if region is too large for GEE direct download
            if not size_estimate['can_use_direct_download']:
                return JSONResponse({
                    'ok': False,
                    'error': f"Region too large for GEE ({size_estimate['estimated_size_mb']:.1f} MB). "
                             f"Please use 'Microsoft' as data source for large regions, "
                             "or reduce resolution (30m, 100m, 250m)."
                })

            # Direct download for small regions
            export_result = export_gee_image_to_geotiff(
                image=image,
                bbox=bbox,
                bands=export_bands,
                scale=scale,
                filename=export_filename
            )

        if export_result['status'] != 'completed':
            raise HTTPException(
                status_code=500,
                detail=export_result.get('error', 'Export failed')
            )

        # Create attachment object for chat
        attachment = create_attachment_from_export(
            file_path=export_result['file_path'],
            layer_name=metadata['layer_name'],
            metadata=metadata
        )

        logger.info(f"Successfully exported layer {layer_id} to {export_result['file_path']}")

        return JSONResponse({
            'ok': True,
            'status': 'completed',
            'attachment': attachment,
            'size_mb': export_result['size_bytes'] / (1024 * 1024),
            'source': chosen_source  # Include source information for UI
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to export layer {layer_id}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/gee/layers")  # type: ignore[misc]
async def list_persisted_layers() -> JSONResponse:
    """List all persisted GEE layers.

    Returns:
        JSON with list of layer metadata
    """
    try:
        from server.layer_metadata_store import list_all_layers

        layers = list_all_layers()
        return JSONResponse({'ok': True, 'layers': layers, 'count': len(layers)})

    except Exception as e:
        logger.error(f"Failed to list layers: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/v1/gee/layers/{layer_id}")  # type: ignore[misc]
async def delete_persisted_layer(layer_id: str) -> JSONResponse:
    """Delete persisted layer metadata.

    Args:
        layer_id: Layer ID to delete

    Returns:
        JSON with success status
    """
    try:
        from server.layer_metadata_store import delete_layer_metadata

        deleted = delete_layer_metadata(layer_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Layer {layer_id} not found")

        return JSONResponse({'ok': True, 'layer_id': layer_id})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete layer {layer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/gee/dates")  # type: ignore[misc]
async def get_sentinel2_dates(
    bbox: list[float] = Query(..., description="Bounding box [minX, minY, maxX, maxY]"),
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    cloud_cover_max: float = Query(20.0, ge=0, le=100, description="Maximum cloud cover %")
) -> JSONResponse:
    """Get available Sentinel-2 acquisition dates for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        JSON with available dates
    """
    try:
        from dta.dti.data_sources.gee_sentinel2 import get_available_dates

        # Validate bbox
        if len(bbox) != 4:
            raise HTTPException(status_code=400, detail="bbox must have 4 values [minX, minY, maxX, maxY]")

        result = get_available_dates(bbox, start_date, end_date, cloud_cover_max)

        if not result.get('ok'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available dates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dates: {str(e)}") from e
