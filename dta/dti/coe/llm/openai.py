"""OpenAI LLM Provider.

OpenAI is a commercial provider; users supply their own API key via the
``OPENAI_API_KEY`` environment variable (``GPT_SECRET`` is accepted as a
fallback name). Requests use the Chat Completions API.

Setup:
    1. Get API key from https://platform.openai.com/api-keys
    2. Set OPENAI_API_KEY environment variable
    3. Optional: set OPENAI_MODELS=gpt-5.4-mini to override the default
"""

from __future__ import annotations

import os
import re
from typing import Any

import openai

from .base import BaseLLMProvider, LLMMessage, LLMResponse

# Default model: the cheapest chat model that passes the `-m llm` planner tests (17/17).
# Published Standard-tier pricing (USD per 1M input/output tokens, checked 2026-10-21):
# gpt-4o-mini 0.15/0.60. Cheaper ones fail here: gpt-5-nano (0.05/0.40) spends its token
# budget on reasoning, gpt-4.1-nano (0.10/0.40) failed 4 of 17.
# Change it here; OPENAI_MODELS overrides it per deployment.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Models that reject any `temperature` other than the default 1 (400 "Unsupported value:
# 'temperature' ..."). `chat-latest` is an alias of the newest chat model.
_RESTRICTED_SAMPLING_MODELS: frozenset[str] = frozenset({"chat-latest"})

# Reasoning models: the o-series, the original gpt-5 family (gpt-5, gpt-5-mini, gpt-5-nano),
# gpt-5.5 and later, and generation 6+. gpt-4.x and gpt-5.1 to gpt-5.4 accept temperature.
_RESTRICTED_SAMPLING_PATTERN = re.compile(r"^(?:o\d|gpt-5(?:-|$)|gpt-5\.(?:[5-9]|\d{2,})|gpt-(?:[6-9]|\d{2,}))")

# TODO: no per-model pricing table yet, so estimate_cost() returns the base 0.0 and
# LLM_STRATEGY=cost ranks OpenAI as free.


def _get_api_key(config: dict[str, Any]) -> str | None:
    """API key from config, then OPENAI_API_KEY, then the legacy GPT_SECRET name."""
    key: str | None = config.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_SECRET")
    return key


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions provider.

    Example:
        >>> provider = OpenAIProvider("gpt-4o-mini")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        **config: Any,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model ID
            **config: Additional configuration:
                - api_key: Override (otherwise reads OPENAI_API_KEY, then GPT_SECRET)
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: SDK retry count (default: 2)
        """
        super().__init__(model, **config)
        self._client: openai.OpenAI | None = None
        self.timeout = config.get("timeout", 60.0)
        self.max_retries = config.get("max_retries", 2)

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    def is_available(self) -> bool:
        """True when an API key is configured."""
        return bool(_get_api_key(self.config))

    def _get_client(self) -> openai.OpenAI:
        """Lazy-init the SDK client."""
        if self._client is None:
            api_key = _get_api_key(self.config)
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY (or GPT_SECRET) environment variable or api_key config required. "
                    "Get your key at: https://platform.openai.com/api-keys"
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _supports_sampling(self) -> bool:
        if self.model in _RESTRICTED_SAMPLING_MODELS:
            return False
        return _RESTRICTED_SAMPLING_PATTERN.match(self.model) is None

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with Chat Completions.

        Args:
            messages: Conversation messages (system messages stay in the array)
            temperature: Sampling temperature (omitted for reasoning models)
            max_tokens: Max output tokens, sent as ``max_completion_tokens``
            **kwargs: Unused

        Returns:
            LLM response

        Raises:
            ValueError: If API key is not configured
            Exception: If generation fails
        """
        client = self._get_client()

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        # `max_tokens` is deprecated and rejected by reasoning models; `max_completion_tokens` works for all.
        if max_tokens:
            req_kwargs["max_completion_tokens"] = max_tokens

        # Sampling params: omitted on reasoning models (rejected with 400 otherwise).
        if self._supports_sampling():
            req_kwargs["temperature"] = temperature

        try:
            response = client.chat.completions.create(**req_kwargs)
        except openai.APIError as e:
            raise Exception(f"OpenAI generation failed: {e}") from e

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            usage=usage,
            provider=self.name,
            metadata={"finish_reason": choice.finish_reason},
        )
