"""Chat, plan, and execute endpoints."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
import logging
import re
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from dta.dti.coe.conversation import generate_reply, input_needs, is_resume_message, offline_reply
from dta.dti.coe.intent_classifier import IntentType, classify_intent
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.coe.triggers import get_trigger_index
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ChatRequest as COEChatRequest
from dta.dti.schemas import ExecutionPlan

from ..job_events import sse_event_frame
from ..schemas import (
    ChatRequest,
    ChatTurn,
    ConversationAnalysis,
    ConversationReply,
    ConversationRequest,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    ExecuteResponse,
    JobSubmitRequest,
    PendingIntent,
    PlanResponse,
)
from ..sessions import CurrentSession, Session
from .jobs import submit_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


def _planning_failed(result: dict[str, Any]) -> JSONResponse:
    """400 ErrorResponse that keeps the rejected plan candidate in ``details``."""
    error = ErrorDetail(
        code=ErrorCode.PLANNING_FAILED,
        message=result.get("error", "Plan generation failed"),
        details={"candidate": result.get("candidate")},
    )
    return JSONResponse(ErrorResponse(error=error).model_dump(mode="json"), status_code=400)


@router.post(  # type: ignore[misc]
    "/plan",
    response_model=PlanResponse,
    summary="Generate execution plan",
    response_description="COE plan without running the pipeline",
)
async def create_plan(req: ChatRequest) -> PlanResponse | JSONResponse:
    """Generate an execution plan from the latest user message (no execution)."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        prompt = req.messages[-1].content
        coe_req = COEChatRequest(prompt=prompt, attachments=[])

        result = orchestrate(coe_req)
        if not result.get("ok"):
            return _planning_failed(result)

        return PlanResponse(plan=result["plan"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {e}") from e


@router.post(  # type: ignore[misc]
    "/execute",
    response_model=ExecuteResponse,
    summary="Plan and execute pipeline",
)
async def execute_plan(req: ChatRequest) -> ExecuteResponse | JSONResponse:
    """Plan via COE and execute the pipeline synchronously."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        prompt = req.messages[-1].content
        coe_req = COEChatRequest(prompt=prompt, attachments=[])

        orch_result = orchestrate(coe_req)
        if not orch_result.get("ok"):
            return _planning_failed(orch_result)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}") from e


# --- Conversation: POST /v1/chat -----------------------------------------------

# The history the LLM sees: the last HISTORY_TURNS turns, each cut to TURN_CHARS, oldest dropped
# until the total is at most HISTORY_CHARS.
HISTORY_TURNS = 10
TURN_CHARS = 2000
HISTORY_CHARS = 8000
# A pending analysis older than this is not resumed.
PENDING_TTL = timedelta(minutes=30)
# Words per reply_delta frame (providers return whole replies, so the reply is streamed in chunks).
CHUNK_WORDS = 6

TWO_FILES_ASK = (
    "That's one file, and this analysis compares two dates. Attach the **before** and the **after** "
    "GeoTIFF together, and I'll run it."
)


def cap_history(history: list[ChatTurn]) -> list[tuple[str, str]]:
    """``(role, text)`` of the turns the LLM sees: see :data:`HISTORY_TURNS`."""
    turns: list[tuple[str, str]] = [(t.role, t.text[:TURN_CHARS]) for t in history[-HISTORY_TURNS:] if t.text.strip()]
    while sum(len(text) for _, text in turns) > HISTORY_CHARS:
        turns.pop(0)
    return turns


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expired(pending: PendingIntent) -> bool:
    asked = pending.asked_at if pending.asked_at.tzinfo else pending.asked_at.replace(tzinfo=timezone.utc)
    return _now() - asked > PENDING_TTL


def _reply(text: str, pending: PendingIntent | None = None) -> ConversationReply:
    return ConversationReply(text=text, pending=pending)


def _ask_for_input(prompt: str, text: str) -> ConversationReply:
    """The registry's ask for the missing file, with the request kept as ``pending``."""
    item_id, needs = input_needs(prompt)
    return _reply(text, PendingIntent(prompt=prompt, item_id=item_id, needs=needs, asked_at=_now()))


def _answer(message: str, history: list[tuple[str, str]]) -> str:
    """The LLM's markdown answer, or the catalog when no provider answers."""
    try:
        return generate_reply(message, history)
    except Exception as e:
        logger.info(f"Conversation reply from the catalog (no provider answered: {type(e).__name__})")
        return offline_reply()


async def _analysis(prompt: str, req: ConversationRequest, session: Session) -> ConversationAnalysis:
    """Submit ``prompt`` exactly as POST /v1/jobs does (quota, owner, region, proposal)."""
    job = await submit_job(JobSubmitRequest(prompt=prompt, attachments=req.attachments, region=req.region), session)
    return ConversationAnalysis(job_id=job.id, job=job)


async def converse(req: ConversationRequest, session: Session) -> ConversationReply | ConversationAnalysis:
    """One Workspace message: a reply (no job), or an analysis submitted through the job queue."""
    files = sum(1 for a in req.attachments if a.path)
    pending = req.pending if req.pending is not None and not _expired(req.pending) else None

    # "here" + files after the assistant asked for them: run the request that waited.
    if pending is not None and is_resume_message(req.message):
        two = "RasterPathBefore" in pending.needs
        if req.region is not None or files >= (2 if two else 1):
            return await _analysis(pending.prompt, req, session)
        ask = get_trigger_index().render_missing_file_response(pending.prompt)
        text = TWO_FILES_ASK if two and files == 1 else ask
        return _reply(text, pending.model_copy(update={"asked_at": _now()}))

    region = {"region": req.region.echo()} if req.region is not None else {}
    coe_req = COEChatRequest(prompt=req.message, attachments=req.attachments, metadata=region)
    intent = await asyncio.to_thread(classify_intent, coe_req)
    if intent.get("intent") == IntentType.CONVERSATION:
        if intent.get("rule") == "missing_input":
            return _ask_for_input(req.message, str(intent.get("response", "")))
        text = str(intent.get("response") or "").strip()
        if not text:  # a question about methods or results, or an LLM verdict without an answer
            text = await asyncio.to_thread(_answer, req.message, cap_history(req.history))
        return _reply(text)
    if not files and req.region is None:
        return _ask_for_input(req.message, get_trigger_index().render_missing_file_response(req.message))
    return await _analysis(req.message, req, session)


def _chunks(text: str) -> list[str]:
    """``text`` in pieces of :data:`CHUNK_WORDS` words; joined they give ``text`` back exactly."""
    tokens = re.split(r"(?<=\s)(?=\S)", text)
    return ["".join(tokens[i : i + CHUNK_WORDS]) for i in range(0, len(tokens), CHUNK_WORDS)]


def _events(outcome: ConversationReply | ConversationAnalysis) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(outcome, ConversationAnalysis):
        return [("analysis", {"job_id": outcome.job_id, "job": outcome.job.model_dump(mode="json")})]
    done: dict[str, Any] = {"text": outcome.text}
    if outcome.pending is not None:
        done["pending"] = outcome.pending.model_dump(mode="json")
    return [("reply_delta", {"text": chunk}) for chunk in _chunks(outcome.text)] + [("reply_done", done)]


async def _frames(outcome: ConversationReply | ConversationAnalysis) -> AsyncIterator[bytes]:
    for seq, (kind, data) in enumerate(_events(outcome), start=1):
        yield sse_event_frame({"seq": seq, "type": kind, "data": data})
        await asyncio.sleep(0)


@router.post(  # type: ignore[misc]
    "/chat",
    summary="Send a Workspace message",
    response_model=ConversationReply | ConversationAnalysis,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE with Accept: text/event-stream"}},
)
async def chat(req: ConversationRequest, session: CurrentSession, accept: str | None = Header(None)) -> Response:
    """Answer one Workspace message: a markdown reply that creates no job, or an analysis.

    Conversation (greetings, "what can you do?", questions about methods or a result) gets a
    reply; it never creates a job and does not count against the guest quota. An analysis
    request is submitted as by ``POST /v1/jobs`` (quota, owner, region and proposal apply: a
    guest over the limit gets the same 429 ``GUEST_QUOTA``); follow its
    ``GET /v1/jobs/{job_id}/events``. A request without a file or region gets the registry's ask
    for the file and a ``pending``; send that ``pending`` back with the next message, and
    "here" (or any short message, or none) with the files runs it. Replace the stored pending
    with each reply's (none means clear it).

    JSON: ``{kind: "reply", text, pending?}`` or ``{kind: "analysis", job_id, job}``. With
    ``Accept: text/event-stream``: ``reply_delta {text}`` frames then ``reply_done {text,
    pending?}``, or one ``analysis {job_id, job}`` frame, in the job event frame format.
    """
    outcome = await converse(req, session)
    if "text/event-stream" in (accept or ""):
        return StreamingResponse(
            _frames(outcome),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    if isinstance(outcome, ConversationAnalysis):
        return JSONResponse(outcome.model_dump(mode="json"))
    return JSONResponse(outcome.model_dump(mode="json", exclude_none=True))
