"""Groq LLM Provider - Ultra-fast inference with LPU.

Groq offers extremely fast inference (300+ tokens/sec) with their
specialized Language Processing Unit (LPU). Free tier available.

Setup:
    1. Get API key from https://console.groq.com/
    2. Set GROQ_API_KEY environment variable
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMMessage, LLMResponse

# Default Groq model - Llama 3.3 70B is fast and capable
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider with ultra-fast inference.

    Groq's LPU provides 18x faster inference than traditional GPUs,
    making it excellent for real-time applications.

    Free tier includes:
    - Rate limits that are generous for research use
    - Access to Llama, Mixtral, and other open models

    Example:
        >>> provider = GroqProvider("llama-3.3-70b-versatile")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str = DEFAULT_GROQ_MODEL,
        api_key: str | None = None,
        timeout: float = 60.0,
        **config: Any,
    ) -> None:
        """Initialize Groq provider.

        Args:
            model: Groq model ID. Options include:
                - llama-3.3-70b-versatile (recommended)
                - llama-3.1-8b-instant (faster, smaller)
                - mixtral-8x7b-32768 (good for code)
                - gemma2-9b-it (Google's Gemma)
            api_key: Groq API key (or set GROQ_API_KEY env var)
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        super().__init__(model, **config)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.timeout = timeout

    @property
    def name(self) -> str:
        """Provider name."""
        return "groq"

    def is_available(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.api_key)

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Groq API.

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
            raise Exception("Groq API key not configured. Set GROQ_API_KEY environment variable.")

        # Convert messages to OpenAI format (Groq uses OpenAI-compatible API)
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
                response = client.post(GROQ_API_URL, json=payload, headers=headers)

                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(f"Groq API error {response.status_code}: {error_detail}")

                data = response.json()

                # Extract response
                choice = data["choices"][0]
                text = choice["message"]["content"]

                # Extract usage
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
                    metadata={
                        "finish_reason": choice.get("finish_reason"),
                        "response_time_ms": data.get("x_groq", {}).get("usage", {}).get("total_time"),
                    },
                )

        except httpx.TimeoutException as e:
            raise Exception(f"Groq request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            raise Exception(f"Groq request failed: {e}") from e

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Estimate cost - Groq free tier has no per-token cost.

        Args:
            messages: Messages to estimate

        Returns:
            Estimated cost (0.0 for free tier)
        """
        # Groq free tier - no per-token charges
        # Paid tier is very cheap (~$0.05-0.10 per 1M tokens)
        return 0.0

    @property
    def supports_images(self) -> bool:
        """Groq currently doesn't support image inputs."""
        return False
