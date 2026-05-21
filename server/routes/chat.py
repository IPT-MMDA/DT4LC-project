"""Chat, plan, and execute endpoints."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ChatRequest as COEChatRequest
from dta.dti.schemas import ExecutionPlan

from ..schemas import ChatRequest, ExecuteResponse, PlanResponse
from ..utils import sse_frame

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/plan",
    response_model=PlanResponse,
    summary="Generate execution plan",
    response_description="COE plan without running the pipeline",
)
async def create_plan(req: ChatRequest) -> PlanResponse:
    """Generate an execution plan from the latest user message (no execution)."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = req.messages[-1].content
    coe_req = COEChatRequest(prompt=prompt, attachments=[])

    result = orchestrate(coe_req)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Plan generation failed"))

    return PlanResponse(plan=result["plan"])


@router.post(
    "/execute",
    response_model=ExecuteResponse,
    summary="Plan and execute pipeline",
)
async def execute_plan(req: ChatRequest) -> ExecuteResponse:
    """Plan via COE and execute the pipeline synchronously."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    prompt = req.messages[-1].content
    coe_req = COEChatRequest(prompt=prompt, attachments=[])

    orch_result = orchestrate(coe_req)
    if not orch_result.get("ok"):
        raise HTTPException(status_code=400, detail=orch_result.get("error", "Plan generation failed"))

    plan_dict = orch_result["plan"]
    plan = ExecutionPlan(**plan_dict)
    executor = PipelineExecutor()

    progress_events: list[dict[str, Any]] = []

    def on_progress(event: dict[str, Any]) -> None:
        progress_events.append(event)

    exec_result = executor.execute(plan, on_progress=on_progress)

    return ExecuteResponse(
        plan=plan_dict,
        result=exec_result,
        progress=progress_events,
    )


@router.post(
    "/chat",
    summary="Stream plan and execution (SSE)",
    response_description="Server-Sent Events stream of planning and execution progress",
)
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream planning and execution as ``text/event-stream`` (legacy chat API)."""

    async def gen() -> AsyncIterator[bytes]:
        try:
            if not req.messages:
                yield sse_frame({"error": "No messages provided"})
                yield sse_frame({"done": True})
                return

            prompt = req.messages[-1].content
            coe_req = COEChatRequest(prompt=prompt, attachments=[])

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

            plan = ExecutionPlan(**orch_result["plan"])
            executor = PipelineExecutor()

            yield sse_frame({"event": "executing", "message": "Running pipeline..."})

            exec_result = executor.execute(plan, on_progress=lambda _e: None)

            yield sse_frame({"event": "complete", "result": exec_result})
            yield sse_frame({"done": True})

        except Exception as e:
            yield sse_frame({"error": str(e)})
            yield sse_frame({"done": True})

    return StreamingResponse(gen(), media_type="text/event-stream")
