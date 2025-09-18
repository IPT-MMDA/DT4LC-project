from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import types

from digital_twin.config import get_settings

from .base import Agent, ChatTurn

SYSTEM_PRIMER = (
    "You are the Context Understanding Agent. "
    "Clarify user intent and data; propose task tags and a high-level pipeline. "
    "Be concise. If input is ambiguous, ask specific follow-ups."
)


def _to_contents(messages: Sequence[ChatTurn]) -> list[types.Content]:
    # Prepend primer as a user instruction (Gemini Dev style)
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=f"[SYSTEM]\n{SYSTEM_PRIMER}")])]
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return contents


class GeminiContextUnderstandingAgent(Agent):
    def __init__(self) -> None:
        s = get_settings()
        # Fail fast if key missing (no mock path)
        if not s.gemini_api_key:
            raise RuntimeError("Missing API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment.")
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._model = s.gemini_model  # e.g., "gemini-2.5-flash" per docs

    def stream(self, messages: Sequence[ChatTurn]) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            contents = _to_contents(messages)
            # Async streaming API (SDK v1.35.0): await to get the stream, then iterate
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            async for event in stream:
                # For text-only chunks the SDK exposes .text
                txt = getattr(event, "text", None)
                if txt:
                    yield txt

        return _gen()
