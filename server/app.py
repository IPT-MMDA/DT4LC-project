import base64
from collections.abc import AsyncIterator
import io
import json
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ChatRequest as COEChatRequest

from .schemas import ChatRequest

app = FastAPI(title="DT4LC API", version="1.0.0")
HEARTBEAT_SECS = 15

# CORS - allow all origins for MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "dt4lc_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def sse_frame(payload: dict[str, Any]) -> bytes:
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
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Please upload a .tif/.tiff GeoTIFF.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Read in-memory with rasterio
    try:
        with MemoryFile(raw) as mem, mem.open() as src:
            # Read first band as masked array
            band1 = src.read(1, masked=True)  # (H, W) masked ndarray
            h, w = band1.shape
            bounds = src.bounds  # left, bottom, right, top
            crs = src.crs.to_string() if src.crs else None

            # Simple percentile stretch to 8-bit for preview
            data = band1.filled(np.nan).astype("float64")
            finite = np.isfinite(data)
            if not finite.any():
                raise HTTPException(status_code=400, detail="All pixels are nodata.")

            # percentiles on finite pixels only
            p2, p98 = np.percentile(data[finite], [2, 98])
            if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                p2, p98 = float(np.min(data[finite])), float(np.max(data[finite]))

            scaled = (data - p2) / (p98 - p2)
            scaled = np.where(np.isfinite(scaled), scaled, 0.0)  # sanitize inf/nan
            scaled = np.clip(scaled, 0.0, 1.0)
            scaled = (scaled * 255.0 + 0.5).astype("uint8")

            # Convert to PNG (grayscale, mode "L")
            img = Image.fromarray(scaled, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return JSONResponse(
            {
                "filename": file.filename,
                "size": [int(w), int(h)],
                "crs": crs,
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "preview_png_base64": b64,  # data:image/png;base64,<this>
            }
        )
    except Exception as e:
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
