"""Google Gemini LLM Provider."""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from .base import BaseLLMProvider, LLMMessage, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider.

    Supports:
    - Text generation
    - Multimodal (images)
    - Streaming (future)

    Example:
        >>> provider = GeminiProvider("gemini-2.0-flash-exp")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(self, model: str = "gemini-2.0-flash-exp", **config: Any) -> None:
        """Initialize Gemini provider.

        Args:
            model: Gemini model ID
            **config: Additional configuration
                - api_key: Override API key (otherwise uses GEMINI_API_KEY env)
                - timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(model, **config)
        self._client: genai.Client | None = None
        self.timeout = config.get("timeout", 60)

    def _get_client(self) -> genai.Client:
        """Lazy initialization of Gemini client."""
        if self._client is None:
            api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable or api_key config required. "
                    "Get your key at: https://aistudio.google.com/apikey"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Gemini.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max output tokens
            **kwargs: Additional Gemini parameters

        Returns:
            LLM response

        Raises:
            ValueError: If API key not configured
            Exception: If generation fails
        """
        client = self._get_client()

        # Convert messages to Gemini format
        contents: list[types.Part | str] = []

        for msg in messages:
            if msg.role == "system":
                # System messages prepended as text
                contents.insert(0, msg.content)
            elif msg.role == "user":
                # Note: Image inputs not currently used in pipeline context
                contents.append(msg.content)
            elif msg.role == "assistant":
                # Gemini doesn't use assistant messages in the same way
                # Skip for now or convert to user context
                pass

        # Generation config
        gen_config: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            gen_config["max_output_tokens"] = max_tokens

        # Merge with kwargs
        gen_config.update(kwargs.get("generation_config", {}))

        try:
            # Pass generation_config as config dict, not as types.GenerateContentConfig
            gen_kwargs: dict[str, Any] = {
                "model": self.model,
                "contents": contents,
            }
            if gen_config:
                gen_kwargs["config"] = types.GenerateContentConfig(**gen_config)

            # Use signal-based timeout on Unix, or try with concurrent.futures on all platforms
            def _generate_with_timeout() -> Any:
                """Generate with timeout handling."""
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(client.models.generate_content, **gen_kwargs)
                    try:
                        return future.result(timeout=self.timeout)
                    except concurrent.futures.TimeoutError as e:
                        raise TimeoutError(f"Gemini request timed out after {self.timeout}s") from e

            response = _generate_with_timeout()

            # Extract usage if available
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            return LLMResponse(
                text=response.text,
                model=self.model,
                usage=usage,
                provider="gemini",
                metadata={
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                },
            )

        except TimeoutError:
            raise  # Re-raise timeout as-is
        except Exception as e:
            raise Exception(f"Gemini generation failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Gemini is available based on API key presence."""
        api_key = self.config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        return bool(api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "gemini"

    @property
    def supports_images(self) -> bool:
        """Gemini supports multimodal inputs."""
        return True

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Estimate cost for Gemini request.

        Gemini pricing (as of 2025):
        - Flash models: ~$0.075 per 1M input tokens, $0.30 per 1M output tokens
        - Pro models: ~$1.25 per 1M input tokens, $5.00 per 1M output tokens

        Args:
            messages: Messages to estimate

        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (4 chars per token)
        total_chars = sum(len(m.content) for m in messages)
        estimated_tokens = total_chars // 4

        if "flash" in self.model.lower():
            # Flash pricing
            input_cost = (estimated_tokens / 1_000_000) * 0.075
            # Assume output is ~25% of input
            output_cost = (estimated_tokens * 0.25 / 1_000_000) * 0.30
            return input_cost + output_cost
        else:
            # Pro pricing
            input_cost = (estimated_tokens / 1_000_000) * 1.25
            output_cost = (estimated_tokens * 0.25 / 1_000_000) * 5.00
            return input_cost + output_cost
