import asyncio
import base64
from collections.abc import AsyncIterator
import io
import json
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from digital_twin.agents.gemini_context import GeminiContextUnderstandingAgent
from digital_twin.config import get_settings

from .schemas import ChatRequest

app = FastAPI(title="DT4LC")
HEARTBEAT_SECS = 15

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_frame(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


@app.get("/v1/health")  # type: ignore[misc]
async def health() -> dict[str, Any]:
    return {"ok": True, "model": settings.gemini_model}


@app.post("/v1/chat")  # type: ignore[misc]
async def chat(req: ChatRequest) -> StreamingResponse:
    agent = GeminiContextUnderstandingAgent()

    async def gen() -> AsyncIterator[bytes]:
        # Heartbeat generator
        async def heartbeat() -> AsyncIterator[bytes]:
            try:
                while True:
                    await asyncio.sleep(HEARTBEAT_SECS)
                    yield sse_frame({"ping": True})
            except asyncio.CancelledError:
                return

        hb = heartbeat()
        agent_stream = agent.stream([m.model_dump() for m in req.messages]).__aiter__()

        async def next_agent_chunk() -> str | bytes:
            return await agent_stream.__anext__()

        async def next_heartbeat() -> bytes:
            return await hb.__anext__()

        try:
            while True:
                done, _ = await asyncio.wait(
                    {
                        asyncio.create_task(next_agent_chunk()),
                        asyncio.create_task(next_heartbeat()),
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    try:
                        val = task.result()
                    except StopAsyncIteration:
                        # model finished
                        yield sse_frame({"done": True})
                        return

                    if isinstance(val, (bytes, bytearray)):
                        # heartbeat already framed
                        yield val
                    else:
                        # handle agent text/error marker
                        if isinstance(val, str) and val.startswith("__ERROR__::"):
                            err = json.loads(val.split("::", 1)[1])
                            yield sse_frame(
                                {
                                    "error": err.get("message"),
                                    "retry_after": err.get("retry_after"),
                                    "kind": err.get("type"),
                                }
                            )
                            yield sse_frame({"done": True})
                            return
                        if val:
                            yield sse_frame({"delta": val})

                # rearm happens via calling next_heartbeat() again in next loop
        finally:
            # best-effort close marker
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
