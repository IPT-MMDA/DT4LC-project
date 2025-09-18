from collections.abc import AsyncGenerator
import json
import asyncio
from typing import Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from digital_twin.config import get_settings
from digital_twin.agents.gemini_context import GeminiContextUnderstandingAgent
from digital_twin.schemas import ChatRequest

app = FastAPI(title="DT4LC")

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
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

    async def gen() -> AsyncGenerator:
        try:
            async for chunk in agent.stream([m.model_dump() for m in req.messages]):
                yield sse_frame({"delta": chunk})
                await asyncio.sleep(0)  # cooperative
            yield sse_frame({"done": True})
        except Exception as e:
            yield sse_frame({"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")
