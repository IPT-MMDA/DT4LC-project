"""OpenRouter LLM Provider.

OpenRouter (https://openrouter.ai) serves many vendors' models behind one OpenAI-compatible Chat Completions
API, so this provider is :class:`~.openai.OpenAIProvider` with OpenRouter's base URL and key. Model IDs carry
the vendor: ``openai/gpt-4o-mini``, ``anthropic/claude-haiku-4.5``, ``google/gemini-2.5-flash``.

Setup:
    1. Get an API key from https://openrouter.ai/keys
    2. Set OPENROUTER_API_KEY and add ``openrouter`` to LLM_PROVIDER_ORDER
    3. Optional: OPENROUTER_MODELS (comma-separated rotation), OPENROUTER_BASE_URL
"""

from __future__ import annotations

import os
from typing import Any

import openai

from .openai import _RESTRICTED_SAMPLING_MODELS, _RESTRICTED_SAMPLING_PATTERN, OpenAIProvider

# The OpenAI default (the cheapest model that passes the planner tests), through OpenRouter.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_api_key(config: dict[str, Any]) -> str | None:
    """API key from config, then OPENROUTER_API_KEY."""
    key: str | None = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    return key


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter Chat Completions provider (OpenAI SDK, OpenRouter endpoint).

    Example:
        >>> provider = OpenRouterProvider("openai/gpt-4o-mini")
        >>> response = provider.generate([LLMMessage(role="user", content="Hello")])
    """

    def __init__(self, model: str = DEFAULT_OPENROUTER_MODEL, **config: Any) -> None:
        """Initialize OpenRouter provider.

        Args:
            model: OpenRouter model ID (``vendor/model``)
            **config: As :class:`~.openai.OpenAIProvider`, plus ``base_url`` (otherwise OPENROUTER_BASE_URL,
                then :data:`DEFAULT_OPENROUTER_BASE_URL`)
        """
        super().__init__(model, **config)
        self.base_url = config.get("base_url") or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL

    @property
    def name(self) -> str:
        """Provider name."""
        return "openrouter"

    def is_available(self) -> bool:
        """True when an API key is configured."""
        return bool(_get_api_key(self.config))

    def _get_client(self) -> openai.OpenAI:
        """Lazy-init the SDK client on OpenRouter's endpoint."""
        if self._client is None:
            api_key = _get_api_key(self.config)
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY environment variable or api_key config required. "
                    "Get your key at: https://openrouter.ai/keys"
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def _supports_sampling(self) -> bool:
        """OpenAI's rule on the bare name of an ``openai/`` model; other vendors take ``temperature``."""
        vendor, _, bare = self.model.rpartition("/")
        if vendor and vendor != "openai":
            return True
        return bare not in _RESTRICTED_SAMPLING_MODELS and _RESTRICTED_SAMPLING_PATTERN.match(bare) is None
