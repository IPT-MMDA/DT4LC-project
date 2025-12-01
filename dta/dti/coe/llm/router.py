"""LLM Router - Smart routing with fallback and load balancing."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM requests to appropriate providers with fallback.

    Features:
    - Primary/fallback provider selection
    - Automatic failover on errors
    - Cost-aware routing (prefer cheap models for simple tasks)
    - Availability checking

    Example:
        >>> from .gemini import GeminiProvider
        >>> from .ollama import OllamaProvider
        >>>
        >>> router = LLMRouter(
        ...     providers=[
        ...         GeminiProvider("gemini-2.0-flash-exp"),
        ...         OllamaProvider("llama3.2"),
        ...     ]
        ... )
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = router.generate(messages)
    """

    def __init__(
        self,
        providers: list[BaseLLMProvider],
        strategy: str = "fallback",
        **config: Any,
    ) -> None:
        """Initialize router.

        Args:
            providers: List of providers in priority order
            strategy: Routing strategy:
                - "fallback": Try primary, fallback on failure
                - "cost": Route based on estimated cost
                - "availability": Route to first available
            **config: Additional configuration
        """
        self.providers = providers
        self.strategy = strategy
        self.config = config

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using best available provider.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max output tokens
            **kwargs: Provider-specific parameters

        Returns:
            LLM response

        Raises:
            Exception: If all providers fail
        """
        if self.strategy == "cost":
            # Sort by estimated cost
            providers = sorted(self.providers, key=lambda p: p.estimate_cost(messages))
        elif self.strategy == "availability":
            # Filter to available providers
            providers = [p for p in self.providers if p.is_available()]
        else:
            # Default fallback strategy (use order as-is)
            providers = self.providers

        if not providers:
            raise Exception("No available LLM providers")

        # Try each provider in order
        last_error = None
        for provider in providers:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} not available, skipping")
                continue

            try:
                logger.info(f"Attempting generation with {provider.name} ({provider.model})")
                response = provider.generate(messages, temperature, max_tokens, **kwargs)
                logger.info(f"Success with {provider.name}")
                return response

            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def add_provider(self, provider: BaseLLMProvider, priority: int | None = None) -> None:
        """Add a provider to the router.

        Args:
            provider: Provider to add
            priority: Insert position (None = append to end)
        """
        if priority is not None:
            self.providers.insert(priority, provider)
        else:
            self.providers.append(provider)

    def remove_provider(self, provider_name: str) -> None:
        """Remove a provider by name.

        Args:
            provider_name: Name of provider to remove
        """
        self.providers = [p for p in self.providers if p.name != provider_name]

    def get_available_providers(self) -> list[BaseLLMProvider]:
        """Get list of currently available providers.

        Returns:
            List of available providers
        """
        return [p for p in self.providers if p.is_available()]

    def estimate_cost(self, messages: list[LLMMessage]) -> dict[str, float]:
        """Estimate cost for each provider.

        Args:
            messages: Messages to estimate

        Returns:
            Dict mapping provider name to estimated cost
        """
        return {p.name: p.estimate_cost(messages) for p in self.providers}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMRouter:
        """Create router from configuration dict.

        Args:
            config: Configuration with provider specs

        Returns:
            Configured LLM router

        Example:
            >>> config = {
            ...     "providers": [
            ...         {"type": "gemini", "model": "gemini-2.0-flash-exp"},
            ...         {"type": "ollama", "model": "llama3.2"},
            ...     ],
            ...     "strategy": "fallback",
            ... }
            >>> router = LLMRouter.from_config(config)
        """
        from .apertus import ApertusProvider
        from .gemini import GeminiProvider
        from .groq import GroqProvider
        from .ollama import OllamaProvider

        provider_map = {
            "gemini": GeminiProvider,
            "groq": GroqProvider,
            "ollama": OllamaProvider,
            "apertus": ApertusProvider,
        }

        providers = []
        for prov_config in config.get("providers", []):
            prov_type = prov_config.get("type")
            if prov_type not in provider_map:
                logger.warning(f"Unknown provider type: {prov_type}")
                continue

            prov_class = provider_map[prov_type]
            model = prov_config.get("model")
            prov_kwargs = {k: v for k, v in prov_config.items() if k not in ("type", "model")}

            provider = prov_class(model=model, **prov_kwargs)
            providers.append(provider)

        strategy = config.get("strategy", "fallback")
        return cls(providers, strategy)
