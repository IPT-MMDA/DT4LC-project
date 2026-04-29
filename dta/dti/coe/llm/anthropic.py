"""Anthropic Claude LLM Provider with prompt caching.

Anthropic Claude is a commercial provider; users supply their own API key via
the ``ANTHROPIC_API_KEY`` environment variable. Prompt caching is enabled for
the system prompt because the orchestrator sends the same registry description
on every request, which yields ~90% input-token cost reduction on cache hits.

Setup:
    1. Get API key from https://console.anthropic.com/
    2. Set ANTHROPIC_API_KEY environment variable
    3. Optional: set ANTHROPIC_MODELS=claude-sonnet-4-6 to override the default
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from .base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

# Default model. Opus 4.7 is the most capable Claude model. Users can override
# via ANTHROPIC_MODELS=claude-sonnet-4-6 for a cheaper/faster default.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"

# Models that reject sampling parameters (temperature/top_p/top_k) and the
# legacy `thinking: {type: "enabled", budget_tokens: N}` shape. Sending them
# returns a 400. Adaptive thinking is the only supported thinking mode here.
_RESTRICTED_SAMPLING_MODELS: frozenset[str] = frozenset({"claude-opus-4-7"})

# Per-million-token pricing in USD as (input, output). Used by estimate_cost()
# so LLM_STRATEGY=cost can route between providers meaningfully.
# Source: https://platform.claude.com/docs/en/about-claude/models/overview
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider with system-prompt caching.

    Supports text generation, multimodal inputs (vision), and ephemeral
    prompt caching on the system prompt. Adaptive thinking can be enabled
    per-call via ``kwargs={"thinking": {"type": "adaptive"}}`` or via the
    ``thinking`` config key on construction.

    Example:
        >>> provider = AnthropicProvider("claude-sonnet-4-6")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        **config: Any,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            model: Claude model ID (see _PRICING_USD_PER_MTOK for known IDs)
            **config: Additional configuration:
                - api_key: Override (otherwise reads ANTHROPIC_API_KEY)
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: SDK retry count (default: 2)
                - thinking: Override thinking config, e.g. {"type": "adaptive"}
        """
        super().__init__(model, **config)
        self._client: anthropic.Anthropic | None = None
        self.timeout: float = float(config.get("timeout", 60.0))
        self.max_retries: int = int(config.get("max_retries", 2))

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    @property
    def supports_images(self) -> bool:
        """Claude is multimodal."""
        return True

    def is_available(self) -> bool:
        """True when an API key is configured."""
        api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        return bool(api_key)

    def _get_client(self) -> anthropic.Anthropic:
        """Lazy-init the SDK client."""
        if self._client is None:
            api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable or api_key config required. "
                    "Get your key at: https://console.anthropic.com/"
                )
            self._client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    @staticmethod
    def _split_messages(
        messages: list[LLMMessage],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Split system messages out of the conversation array.

        Anthropic takes ``system`` as a top-level parameter, not as a role in
        ``messages``. Multiple system messages are concatenated with a blank-
        line separator so the cache key stays byte-stable across requests
        with the same system blocks.
        """
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                api_messages.append({"role": msg.role, "content": msg.content})
        return "\n\n".join(system_parts), api_messages

    def _supports_sampling(self) -> bool:
        return self.model not in _RESTRICTED_SAMPLING_MODELS

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response, caching the system prompt prefix.

        Args:
            messages: Conversation messages (system messages are split out)
            temperature: Sampling temperature (ignored on Opus 4.7)
            max_tokens: Max output tokens (defaults to 4096)
            **kwargs: Optional ``thinking`` override

        Returns:
            LLM response with cache hit/miss stats in ``metadata``

        Raises:
            ValueError: If API key is not configured
            Exception: If generation fails
        """
        client = self._get_client()
        system_text, api_messages = self._split_messages(messages)

        # Build the request. The system prompt is wrapped in a list with
        # cache_control: ephemeral so repeat requests serve it from cache.
        # Falls back to no-system when there's nothing to cache.
        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 4096,
            "messages": api_messages,
        }
        if system_text:
            req_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Sampling params: omitted on Opus 4.7 (rejected with 400 otherwise).
        if self._supports_sampling():
            req_kwargs["temperature"] = temperature

        # Adaptive thinking: opt-in per-call (kwargs) or per-provider (config).
        thinking = kwargs.get("thinking", self.config.get("thinking"))
        if thinking:
            req_kwargs["thinking"] = thinking

        try:
            response = client.messages.create(**req_kwargs)
        except anthropic.AuthenticationError as e:
            raise Exception(f"Anthropic authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise Exception(f"Anthropic rate limit hit: {e}") from e
        except anthropic.APIError as e:
            raise Exception(f"Anthropic generation failed: {e}") from e

        # Concatenate text blocks (the response can interleave text with
        # thinking/tool_use blocks; we only surface text here).
        text = "".join(b.text for b in response.content if b.type == "text")

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Surface cache stats so callers can verify caching is working — if
        # cache_read_input_tokens is consistently 0 across repeats, a silent
        # invalidator is at work (see shared/prompt-caching.md).
        metadata: dict[str, Any] = {"stop_reason": response.stop_reason}
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
        cache_read = getattr(response.usage, "cache_read_input_tokens", None)
        if cache_creation is not None:
            metadata["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            metadata["cache_read_input_tokens"] = cache_read

        return LLMResponse(
            text=text,
            model=response.model,
            usage=usage,
            provider=self.name,
            metadata=metadata,
        )

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Estimate request cost in USD.

        Uses the per-Mtok pricing table for the configured model. Input
        tokens are approximated at 4 chars/token; output is estimated at
        25% of input.
        """
        prices = _PRICING_USD_PER_MTOK.get(self.model)
        if not prices:
            return 0.0
        input_per_mtok, output_per_mtok = prices
        input_chars = sum(len(m.content) for m in messages)
        input_tokens = input_chars / 4
        output_tokens = input_tokens * 0.25
        input_cost = input_tokens / 1_000_000 * input_per_mtok
        output_cost = output_tokens / 1_000_000 * output_per_mtok
        return input_cost + output_cost
