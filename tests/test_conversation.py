"""Conversation path (v0.6.0): ``POST /v1/chat`` answers conversation without a job.

Greetings, "what can you do?" and questions get markdown replies and create no job, with or
without a provider; an analysis request goes through the job queue as ``POST /v1/jobs`` does;
a request without a file gets a ``pending`` that "here" + the file resumes.

The real router points at a closed Ollama port (no test reaches a live provider); the
"mocked LLM" tests replace it with a stub that records what it was sent.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import rasterio

from dta.config import ROOT_DIR
from dta.dti.coe.conversation import OFFLINE_PREFIX, conversational_rule, is_resume_message
from dta.dti.coe.intent_classifier import IntentType, classify_intent
from dta.dti.coe.llm import LLMResponse, get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.raster import describe_raster
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import ChatRequest
from server import jobs as jobs_mod
from server.jobs import JobQueue, SQLiteJobStore
from server.routes import chat as chat_routes
from server.schemas import ChatTurn

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
NDMI_PROMPT = "Map moisture change with NDMI between the two dates"
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
SSE = {"Accept": "text/event-stream"}


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Real router with Ollama on a closed port as its only provider: every generate() raises."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    reset_llm_router()
    assert all(isinstance(p, OllamaProvider) for p in get_llm_router().providers)
    yield
    reset_llm_router()


@pytest.fixture
async def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    await q.start()
    yield q
    await q.stop()


class _StubRouter:
    """An available provider: every generate() returns ``text`` and records the messages."""

    providers: list[Any] = []

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(text=self.text, model="stub", provider="stub")


@pytest.fixture
def llm() -> Iterator[_StubRouter]:
    """A mocked LLM for the classifier and the conversation reply."""
    stub = _StubRouter("**NDWI** uses the green band:\n\n- water reflects green\n- and absorbs NIR")
    with (
        patch("dta.dti.coe.conversation.get_llm_router", return_value=stub),
        patch("dta.dti.coe.intent_classifier.get_llm_router", return_value=stub),
    ):
        yield stub


def _client() -> httpx.AsyncClient:
    """A browser on the app (no lifespan): its own cookie jar, so its own session."""
    from server.app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _attachment(path: str) -> dict[str, Any]:
    """What the UI sends for an upload: the ``POST /v1/upload`` body, metadata included."""
    name = Path(path).name
    with rasterio.open(path) as src:
        meta = describe_raster(src, name, os.path.getsize(path)).model_dump()
    return {"id": name, "filename": name, "mime_type": "image/tiff", "path": path, "metadata": meta}


def _frames(body: str) -> list[tuple[int, str, dict[str, Any]]]:
    """``(id, event, data)`` of each SSE frame."""
    frames = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        frames.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return frames


async def _chat(client: httpx.AsyncClient, sse: bool = False, **body: Any) -> dict[str, Any]:
    """POST /v1/chat; an SSE response is folded into the JSON shape (and its frames checked)."""
    r = await client.post("/v1/chat", json=body, headers=SSE if sse else {})
    assert r.status_code == 200, r.text
    if not sse:
        assert r.headers["content-type"].startswith("application/json")
        return r.json()  # type: ignore[no-any-return]
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    assert [f[0] for f in frames] == list(range(1, len(frames) + 1))
    if frames[0][1] == "analysis":
        assert len(frames) == 1
        return {"kind": "analysis", **frames[0][2]}
    *deltas, (_, last, done) = frames
    assert last == "reply_done"
    assert deltas and {f[1] for f in deltas} == {"reply_delta"}
    assert "".join(f[2]["text"] for f in deltas) == done["text"]
    return {"kind": "reply", **done}


async def _job_count(client: httpx.AsyncClient, q: JobQueue) -> tuple[int, int]:
    listed = (await client.get("/v1/jobs")).json()
    return q._store.count(), listed["total"]


async def _wait(q: JobQueue, job_id: str, statuses: set[str], timeout: float = 120) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        job = await q.get_job(job_id)
        assert job is not None
        if job.status.value in statuses:
            return job.to_dict()
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def _later_description_line(item_id: str) -> str:
    """The last line of a registry description, on one line as the catalog writes it."""
    lines = (get_item(load_registry(), item_id).description or "").strip().splitlines()
    assert len(lines) > 1
    return lines[-1].strip()


# --- hello and "what can you do?" ------------------------------------------------


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("mocked_llm", [False, True], ids=["no-llm", "mocked-llm"])
async def test_hello_and_capabilities_reply_in_markdown_and_create_no_job(
    queue: JobQueue, sse: bool, mocked_llm: bool, request: pytest.FixtureRequest
) -> None:
    stub = request.getfixturevalue("llm") if mocked_llm else None
    async with _client() as client:
        hello = await _chat(client, sse, message="hello")
        assert hello["kind"] == "reply"
        assert hello["text"].startswith("Hello!")
        assert "\n- " in hello["text"]  # a markdown list
        assert "pending" not in hello

        caps = await _chat(client, sse, message="What can you do?")
        assert caps["kind"] == "reply"
        assert "\n- **NDVI Calculation** — " in caps["text"]  # a markdown list, bold names
        # Whole descriptions: a later line of the NDVI and Delineate-Anything descriptions is there.
        assert _later_description_line("algorithms/ndvi") in caps["text"]
        assert _later_description_line("models/delineate-anything") in caps["text"]
        assert "pending" not in caps

        assert await _job_count(client, queue) == (0, 0)
    if stub is not None:
        assert stub.calls == []  # both are deterministic: no LLM call


async def test_capability_reply_escapes_formula_asterisks(queue: JobQueue) -> None:
    async with _client() as client:
        caps = await _chat(client, message="what can you do")
    assert r"6\*RED - 7.5\*BLUE" in caps["text"]  # EVI's formula, not emphasis


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
async def test_method_question_offline_answers_from_the_catalog(queue: JobQueue, sse: bool) -> None:
    async with _client() as client:
        reply = await _chat(client, sse, message="Why is NDWI better than NDVI for water?")
        assert reply["kind"] == "reply"
        assert reply["text"].startswith(OFFLINE_PREFIX)
        assert "\n- **NDWI Calculation** — " in reply["text"]
        assert await _job_count(client, queue) == (0, 0)


async def test_method_question_with_llm_gets_its_markdown_with_history(queue: JobQueue, llm: _StubRouter) -> None:
    history = [
        {"role": "user", "text": "Map NDWI change between the two dates"},
        {"role": "assistant", "text": "Done: +428 km² of water gained, 43 km² lost."},
    ]
    async with _client() as client:
        reply = await _chat(client, True, message="Explain the last result", history=history)
        assert reply == {"kind": "reply", "text": llm.text}
        assert await _job_count(client, queue) == (0, 0)
    (messages,) = llm.calls
    system, user = messages
    assert system.role == "system"
    assert "concise markdown" in system.content
    assert _later_description_line("algorithms/ndwi") in system.content
    assert system.content.endswith(
        "The conversation so far:\n\nUser: Map NDWI change between the two dates\n\n"
        "Assistant: Done: +428 km² of water gained, 43 km² lost."
    )
    assert (user.role, user.content) == ("user", "Explain the last result")


# --- history cap -------------------------------------------------------------------


def test_cap_history_keeps_the_last_turns_within_the_char_cap() -> None:
    turns = [ChatTurn(role="user" if i % 2 == 0 else "assistant", text=f"turn {i}") for i in range(30)]
    capped = chat_routes.cap_history(turns)
    assert len(capped) == chat_routes.HISTORY_TURNS == 10
    assert capped[0] == ("user", "turn 20")
    assert capped[-1] == ("assistant", "turn 29")

    long = [ChatTurn(role="user", text=f"{i}" + "x" * 4999) for i in range(10)]
    capped = chat_routes.cap_history(long)
    assert all(len(text) == chat_routes.TURN_CHARS for _, text in capped)
    assert sum(len(text) for _, text in capped) <= chat_routes.HISTORY_CHARS
    assert [text[0] for _, text in capped] == ["6", "7", "8", "9"]  # oldest dropped first


async def test_history_cap_applies_to_what_the_llm_sees(queue: JobQueue, llm: _StubRouter) -> None:
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}"} for i in range(30)]
    async with _client() as client:
        await _chat(client, message="What is NDVI?", history=history)
    (messages,) = llm.calls
    transcript = messages[0].content.split("The conversation so far:\n\n", 1)[1]
    assert [line.split(": ", 1)[1] for line in transcript.split("\n\n")] == [f"turn {i}" for i in range(20, 30)]
    assert messages[1].content == "What is NDVI?"


# --- pending intent ----------------------------------------------------------------


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
async def test_calculate_ndvi_asks_for_the_file_then_here_resumes_it(
    queue: JobQueue, sse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _client() as client:
        ask = await _chat(client, sse, message="Calculate NDVI")
        assert ask["kind"] == "reply"
        assert ask["text"] == get_item(load_registry(), "algorithms/ndvi").user_guide.missing_file_response.strip()
        pending = ask["pending"]
        assert pending["prompt"] == "Calculate NDVI"
        assert pending["item_id"] == "algorithms/ndvi"
        assert pending["needs"] == ["RasterPath"]
        asked_at = datetime.fromisoformat(pending["asked_at"].replace("Z", "+00:00"))
        assert abs(datetime.now(timezone.utc) - asked_at) < timedelta(minutes=1)
        assert await _job_count(client, queue) == (0, 0)

        started = await _chat(client, sse, message="here", attachments=[_attachment(SAMPLE_0601)], pending=pending)
        assert started["kind"] == "analysis"
        job_id = started["job_id"]
        assert started["job"]["id"] == job_id
        assert started["job"]["prompt"] == "Calculate NDVI"
        assert [e["type"] for e in started["job"]["events"]] == ["intent"]
        assert await _job_count(client, queue) == (1, 1)

        job = await _wait(queue, job_id, {"completed", "failed", "cancelled"})
        assert job["status"] == "completed", job["error"]
        assert [s["uses"] for s in job["plan"]["steps"]][:2] == ["input/file", "algorithms/ndvi"]
        assert job["events"][-1]["type"] == "done"
        assert job["events"][-1]["data"]["summary"].endswith("NDVI")


async def test_here_with_the_sample_pair_resumes_calculate_ndvi_on_9_june(queue: JobQueue) -> None:
    """D1: "here" + both sample files after "Calculate NDVI" ran NDVI change detection."""
    async with _client() as client:
        ask = await _chat(client, message="Calculate NDVI")
        both = [_attachment(SAMPLE_0601), _attachment(SAMPLE_0609)]
        started = await _chat(client, message="here", attachments=both, pending=ask["pending"])
        assert started["kind"] == "analysis"
        assert started["job"]["events"][0]["data"]["text"] == "I'll compute NDVI for 9 June from the sample."

        job = await _wait(queue, started["job_id"], {"completed", "failed", "cancelled"})
        assert job["status"] == "completed", job["error"]
        steps = job["plan"]["steps"]
        assert [s["uses"] for s in steps] == ["input/file", "algorithms/ndvi", "post-processing/agent-analysis"]
        assert steps[0]["binds"] == {"RasterPath": SAMPLE_0609}


async def test_here_without_pending_is_a_conversational_reply(queue: JobQueue) -> None:
    async with _client() as client:
        bare = await _chat(client, message="here")
        assert bare["kind"] == "reply"
        assert "pending" not in bare
        with_file = await _chat(client, message="here", attachments=[_attachment(SAMPLE_0601)])
        assert with_file["kind"] == "reply"
        assert with_file["text"].startswith("Got it. What should I compute from this file?")
        assert await _job_count(client, queue) == (0, 0)


async def test_expired_pending_is_not_resumed(queue: JobQueue) -> None:
    stale = datetime.now(timezone.utc) - chat_routes.PENDING_TTL - timedelta(minutes=1)
    pending = {"prompt": "Calculate NDVI", "item_id": "algorithms/ndvi", "needs": ["RasterPath"]}
    async with _client() as client:
        reply = await _chat(
            client,
            message="here",
            attachments=[_attachment(SAMPLE_0601)],
            pending={**pending, "asked_at": stale.isoformat()},
        )
        assert reply["kind"] == "reply"
        assert "pending" not in reply
        assert await _job_count(client, queue) == (0, 0)


async def test_change_request_with_one_file_asks_again_then_two_files_run_it(queue: JobQueue) -> None:
    async with _client() as client:
        ask = await _chat(client, message=NDMI_PROMPT)
        pending = ask["pending"]
        assert pending["needs"] == ["RasterPathBefore", "RasterPathAfter"]
        assert pending["item_id"] == "algorithms/band-math-change"

        again = await _chat(client, message="here you go", attachments=[_attachment(SAMPLE_0601)], pending=pending)
        assert again["kind"] == "reply"
        assert again["text"] == chat_routes.TWO_FILES_ASK
        assert again["pending"]["prompt"] == NDMI_PROMPT
        assert again["pending"]["asked_at"] >= pending["asked_at"]
        assert await _job_count(client, queue) == (0, 0)

        both = [_attachment(SAMPLE_0601), _attachment(SAMPLE_0609)]
        started = await _chat(client, message="", attachments=both, pending=again["pending"])
        assert started["kind"] == "analysis"
        job = await _wait(queue, started["job_id"], {"proposal", "completed", "failed"})
        assert job["status"] == "proposal", job["error"]


async def test_a_new_request_replaces_the_pending(queue: JobQueue) -> None:
    pending = {
        "prompt": "Calculate NDVI",
        "item_id": "algorithms/ndvi",
        "needs": ["RasterPath"],
        "asked_at": datetime.now(timezone.utc).isoformat(),
    }
    async with _client() as client:
        started = await _chat(
            client, message="Calculate NDWI for this image", attachments=[_attachment(SAMPLE_0601)], pending=pending
        )
        assert started["kind"] == "analysis"
        assert started["job"]["prompt"] == "Calculate NDWI for this image"


# --- analyses through /v1/chat ------------------------------------------------------


@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
async def test_ndmi_with_two_files_via_chat_reaches_proposal(queue: JobQueue, sse: bool) -> None:
    async with _client() as client:
        started = await _chat(
            client, sse, message=NDMI_PROMPT, attachments=[_attachment(SAMPLE_0601), _attachment(SAMPLE_0609)]
        )
        assert started["kind"] == "analysis"
        job = await _wait(queue, started["job_id"], {"proposal", "completed", "failed"})
        assert job["status"] == "proposal", job["error"]
        record = (await client.get(f"/v1/jobs/{started['job_id']}")).json()
        assert record["proposal"]["plan"]["steps"][2]["uses"] == "algorithms/band-math-change"


async def test_guest_quota_counts_analyses_not_conversation(queue: JobQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setenv("GUEST_ANALYSES_PER_HOUR", "1")

    async def used(client: httpx.AsyncClient) -> int:
        return int((await client.get("/v1/session")).json()["analyses"]["used"])

    async with _client() as client:
        for message in ("hello", "What can you do?", "What is NDVI?", "Calculate NDVI", "thanks"):
            assert (await _chat(client, message=message))["kind"] == "reply"
        assert await used(client) == 0

        started = await _chat(client, message="Calculate NDVI", attachments=[_attachment(SAMPLE_0601)])
        assert started["kind"] == "analysis"
        assert await used(client) == 1

        for headers in ({}, SSE):
            r = await client.post(
                "/v1/chat",
                json={"message": "Calculate NDWI", "attachments": [_attachment(SAMPLE_0601)]},
                headers=headers,
            )
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "GUEST_QUOTA"
            assert "Retry-After" in r.headers

        assert (await _chat(client, message="hello"))["kind"] == "reply"  # conversation still answers
        await _wait(queue, started["job_id"], {"completed", "failed"})
        assert await _job_count(client, queue) == (1, 1)


# --- the conversational rule set --------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "rule"),
    [
        ("hello", "greeting"),
        ("Hi there!", "greeting"),
        ("Good morning, how are you?", "greeting"),
        ("thanks", "thanks"),
        ("Ok, thank you very much!", "thanks"),
        ("here", "deictic"),
        ("Here you go", "deictic"),
        ("this one", "deictic"),
        ("", "deictic"),
        ("What is NDVI?", "question"),
        ("why is NDWI better than NDVI for water?", "question"),
        ("Explain the last result", "question"),
        ("Is the water gain significant?", "question"),
        ("How do I compute NDVI?", None),  # an analysis verb
        ("Calculate NDVI", None),
        (NDMI_PROMPT, None),
        ("The quick brown fox jumps over the lazy dog.", None),
    ],
)
def test_conversational_rule(prompt: str, rule: str | None) -> None:
    assert conversational_rule(prompt, has_input=False) == rule


def test_question_rule_needs_no_input() -> None:
    assert conversational_rule("What is in this image?", has_input=True) is None
    assert conversational_rule("hello", has_input=True) == "greeting"


@pytest.mark.parametrize(
    ("message", "resumes"),
    [
        ("here", True),
        ("", True),
        ("use this", True),
        ("thanks", False),
        ("hello", False),
        ("Explain the last result", False),
        ("NDWI please", False),
        ("Calculate NDWI for this image", False),
    ],
)
def test_resume_message(message: str, resumes: bool) -> None:
    assert is_resume_message(message) is resumes


def test_classifier_with_every_provider_down_keeps_conversation_out_of_the_pipeline() -> None:
    """The HD fallback reads a prompt no rule matches as a pipeline request; these never reach it."""
    for prompt in ("hello", "thanks", "here", "What is NDVI?", "What can you do?"):
        assert classify_intent(ChatRequest(prompt=prompt))["intent"] == IntentType.CONVERSATION, prompt
    assert classify_intent(ChatRequest(prompt="The quick brown fox."))["intent"] == IntentType.PIPELINE
