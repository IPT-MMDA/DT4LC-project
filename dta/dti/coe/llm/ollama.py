"""Ollama Local LLM Provider.

Supports running local models like Llama, Mistral, Phi, etc. via Ollama.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import requests

from .base import BaseLLMProvider, LLMMessage, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider.

    Connects to local Ollama instance to run models like:
    - llama3.2, llama3.1, llama2
    - mistral, mixtral
    - phi3
    - qwen, gemma, etc.

    Prerequisites:
        - Ollama installed (https://ollama.com)
        - Model pulled (e.g., `ollama pull llama3.2`)
        - Ollama running (`ollama serve`)

    Example:
        >>> provider = OllamaProvider("llama3.2")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        **config: Any,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral")
            base_url: Ollama API base URL
            **config: Additional configuration
                - timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(model, **config)
        self.base_url = base_url.rstrip("/")
        self.timeout = config.get("timeout", 120)

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Ollama.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max output tokens
            **kwargs: Additional Ollama parameters

        Returns:
            LLM response

        Raises:
            Exception: If Ollama is not available or generation fails
        """
        url = urljoin(self.base_url, "/api/chat")

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            msg_dict: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.images:
                msg_dict["images"] = msg.images
            ollama_messages.append(msg_dict)

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **kwargs.get("options", {}),
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            text = data.get("message", {}).get("content", "")

            # Extract usage if available
            usage = None
            if "prompt_eval_count" in data or "eval_count" in data:
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                }

            return LLMResponse(
                text=text,
                model=self.model,
                usage=usage,
                provider="ollama",
                metadata={
                    "done_reason": data.get("done_reason"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_duration": data.get("prompt_eval_duration"),
                    "eval_duration": data.get("eval_duration"),
                },
            )

        except requests.exceptions.ConnectionError as e:
            raise Exception(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`). "
                f"Error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise Exception(f"Ollama request timed out after {self.timeout}s: {e}") from e
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is available."""
        try:
            # Check if Ollama is running
            version_url = urljoin(self.base_url, "/api/version")
            response = requests.get(version_url, timeout=5)
            response.raise_for_status()

            # Check if model is available
            tags_url = urljoin(self.base_url, "/api/tags")
            tags_response = requests.get(tags_url, timeout=5)
            tags_response.raise_for_status()

            models = tags_response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            return self.model in model_names or any(self.model in name for name in model_names)

        except requests.exceptions.RequestException:
            # Network error, connection refused, timeout, etc.
            return False
        except (KeyError, ValueError):
            # JSON parsing issues
            return False

    @property
    def name(self) -> str:
        """Provider name."""
        return "ollama"

    @property
    def supports_images(self) -> bool:
        """Some Ollama models support images (llava, bakllava)."""
        vision_models = ["llava", "bakllava"]
        return any(vm in self.model.lower() for vm in vision_models)

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Local models are free!

        Args:
            messages: Messages (ignored for local models)

        Returns:
            0.0 (local models have no API costs)
        """
        return 0.0

    def list_available_models(self) -> list[str]:
        """List all models available in Ollama.

        Returns:
            List of model names

        Raises:
            Exception: If Ollama is not available
        """
        try:
            tags_url = urljoin(self.base_url, "/api/tags")
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]

        except Exception as e:
            raise Exception(f"Failed to list Ollama models: {e}") from e
