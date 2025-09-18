from __future__ import annotations

import asyncio  # <-- add
from collections.abc import AsyncIterator, Sequence
import json

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from digital_twin.config import get_settings

from .base import Agent, ChatTurn

SYSTEM_PRIMER = (
    "You are the Context Understanding Agent. "
    "Clarify user intent and data; propose task tags and a high-level pipeline. "
    "Be concise. If input is ambiguous, ask specific follow-ups."
)


def _to_contents(messages: Sequence[ChatTurn]) -> list[types.Content]:
    out: list[types.Content] = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        out.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
    return out


class GeminiContextUnderstandingAgent(Agent):
    def __init__(self) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            raise RuntimeError("Missing API key. Set GEMINI_API_KEY in your environment.")
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_model

    def stream(self, messages: Sequence[ChatTurn]) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            contents = _to_contents(messages)
            try:
                resp_stream = self._client.models.generate_content_stream(
                    model=self._model,
                    contents=_to_contents(messages),
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        system_instruction=types.Part.from_text(text=SYSTEM_PRIMER),  # <-- keyword 'text='
                    ),
                )

                # --- NEW: handle both async and sync streams ---
                if hasattr(resp_stream, "__aiter__"):
                    async for event in resp_stream:  # async path
                        text = event.text()
                        if text:
                            yield text
                else:
                    for event in resp_stream:  # sync path
                        text = event.text()
                        if text:
                            yield text
                        # yield control to the loop so outer SSE isn’t starved
                        await asyncio.sleep(0)
                # ---------------------------------------------

            except ClientError as e:
                retry_after = None
                try:
                    details = e.response_json.get("error", {}).get("details", [])
                    for d in details:
                        if d.get("@type", "").endswith("RetryInfo"):
                            retry_after = int(d.get("retryDelay", "0s").rstrip("s") or "0")
                            break
                except Exception:
                    pass
                payload = {"type": "rate_limit", "message": str(e), "retry_after": retry_after}
                yield "__ERROR__::" + json.dumps(payload)

        return _gen()
