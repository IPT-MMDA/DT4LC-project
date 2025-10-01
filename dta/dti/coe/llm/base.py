"""Base LLM Provider Protocol.

Defines the interface that all LLM providers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class LLMMessage:
    """Represents a message in an LLM conversation.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message text content
        images: Optional list of image data (base64 or URLs)
    """

    role: Literal["user", "assistant", "system"]
    content: str
    images: list[str] | None = None


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        text: Generated text response
        model: Model that generated the response
        usage: Token usage information
        provider: Provider name
        metadata: Additional provider-specific data
    """

    text: str
    model: str
    usage: dict[str, int] | None = None
    provider: str = "unknown"
    metadata: dict[str, Any] | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (Gemini, Ollama, OpenAI, etc.) must implement this interface.
    """

    def __init__(self, model: str, **config: Any) -> None:
        """Initialize the provider.

        Args:
            model: Model identifier (e.g., "gemini-2.0-flash-exp", "llama3.2")
            **config: Provider-specific configuration
        """
        self.model = model
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLM response with text and metadata

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available.

        Returns:
            True if provider can be used, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'gemini', 'ollama')."""
        pass

    @property
    def supports_images(self) -> bool:
        """Whether this provider supports image inputs.

        Returns:
            True if images are supported
        """
        return False

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Estimate cost for this request in USD.

        Args:
            messages: Messages to estimate cost for

        Returns:
            Estimated cost in USD (0.0 for local models)
        """
        return 0.0
