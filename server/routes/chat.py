"""Chat, plan, and execute endpoints."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ChatRequest as COEChatRequest

from ..schemas import ChatRequest
from ..utils import sse_frame

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/plan")  # type: ignore[misc]
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


@router.post("/execute")  # type: ignore[misc]
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


@router.post("/chat")  # type: ignore[misc]
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
