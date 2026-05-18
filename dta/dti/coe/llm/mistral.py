"""Mistral AI LLM Provider.

Mistral AI offers commercial models (mistral-large, mistral-medium, mistral-small,
ministral, codestral) via an OpenAI-compatible API. Users supply their own
``MISTRAL_API_KEY``; not to be confused with Groq's ``mixtral-*`` model
family, which is a different company's hosted version of Mistral's open weights.

Setup:
    1. Get API key from https://console.mistral.ai/
    2. Set MISTRAL_API_KEY environment variable
    3. Optional: set MISTRAL_MODELS=mistral-large-latest to override the default
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMMessage, LLMResponse, estimate_token_cost

# Default model. mistral-medium-latest balances cost and quality for orchestration.
DEFAULT_MISTRAL_MODEL = "mistral-medium-latest"

# Mistral API endpoint (OpenAI-compatible).
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Per-million-token pricing in USD as (input, output). Used by estimate_cost()
# so LLM_STRATEGY=cost can route between providers meaningfully.
# Source: https://mistral.ai/pricing
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "mistral-large-latest": (2.00, 6.00),
    "mistral-medium-latest": (0.40, 2.00),
    "mistral-small-latest": (0.20, 0.60),
    "ministral-8b-latest": (0.10, 0.10),
    "ministral-3b-latest": (0.04, 0.04),
    "codestral-latest": (0.30, 0.90),
}


class MistralProvider(BaseLLMProvider):
    """Mistral AI LLM provider.

    Mistral's API is OpenAI-compatible at ``api.mistral.ai/v1/chat/completions``,
    so this provider uses raw httpx for consistency with the existing Groq
    provider rather than pulling in the ``mistralai`` SDK.

    Example:
        >>> provider = MistralProvider("mistral-medium-latest")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str = DEFAULT_MISTRAL_MODEL,
        api_key: str | None = None,
        timeout: float = 60.0,
        **config: Any,
    ) -> None:
        """Initialize Mistral provider.

        Args:
            model: Mistral model ID (see _PRICING_USD_PER_MTOK for known IDs)
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        super().__init__(model, **config)
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Provider name."""
        return "mistral"

    @property
    def supports_images(self) -> bool:
        """Text-only at this provider — vision models like pixtral are out of scope."""
        return False

    def is_available(self) -> bool:
        """Check if Mistral API key is configured."""
        return bool(self.api_key)

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Mistral API.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLM response

        Raises:
            Exception: If API call fails
        """
        if not self.api_key:
            raise Exception("Mistral API key not configured. Set MISTRAL_API_KEY environment variable.")

        # Mistral uses OpenAI-compatible message format directly.
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(MISTRAL_API_URL, json=payload, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Mistral API error {response.status_code}: {response.text}")

                data = response.json()

                choice = data["choices"][0]
                text = choice["message"]["content"]
                usage = data.get("usage", {})

                return LLMResponse(
                    text=text,
                    model=self.model,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    provider=self.name,
                    metadata={"finish_reason": choice.get("finish_reason")},
                )

        except httpx.TimeoutException as e:
            raise Exception(f"Mistral request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            raise Exception(f"Mistral request failed: {e}") from e

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Estimate request cost in USD using the per-model pricing table."""
        prices = _PRICING_USD_PER_MTOK.get(self.model)
        if not prices:
            return 0.0
        return estimate_token_cost(messages, *prices)
