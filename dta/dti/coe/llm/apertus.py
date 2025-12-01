"""Apertus LLM Provider (Swiss AI - Local HuggingFace).

Apertus is an open-source multilingual LLM from Swiss AI (EPFL, ETH Zurich, CSCS).
Available in 8B and 70B parameter versions.

This provider runs locally using HuggingFace Transformers.
Requires significant GPU memory:
- 8B model: ~16GB VRAM
- 70B model: ~140GB VRAM

See: https://huggingface.co/collections/swiss-ai/apertus-llm
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BaseLLMProvider, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy dependencies unless needed
_pipeline = None
_torch = None


def _get_pipeline():
    """Lazy load the transformers pipeline."""
    global _pipeline, _torch
    if _pipeline is None:
        try:
            import torch
            from transformers import pipeline

            _torch = torch
            _pipeline = pipeline
        except ImportError as e:
            raise ImportError(
                "Apertus provider requires 'transformers' and 'torch'. Install with: pip install transformers torch"
            ) from e
    return _pipeline, _torch


class ApertusProvider(BaseLLMProvider):
    """Swiss AI Apertus LLM provider (local HuggingFace).

    Runs Apertus models locally using HuggingFace Transformers.
    Requires GPU with sufficient VRAM.

    Available models:
    - swiss-ai/Apertus-8B (~16GB VRAM)
    - swiss-ai/Apertus-70B (~140GB VRAM)

    Example:
        >>> provider = ApertusProvider("swiss-ai/Apertus-8B")
        >>> messages = [LLMMessage(role="user", content="Hello")]
        >>> response = provider.generate(messages)
        >>> print(response.text)
    """

    def __init__(self, model: str = "swiss-ai/Apertus-8B", **config: Any) -> None:
        """Initialize Apertus provider.

        Args:
            model: HuggingFace model ID (default: swiss-ai/Apertus-8B)
            **config: Additional configuration
                - device: Device to use ("cuda", "cpu", or device index)
                - dtype: Torch dtype ("bfloat16", "float16", "float32")
                - max_memory: Max memory per device for model sharding
        """
        super().__init__(model, **config)
        self._text_pipeline = None
        self._loaded = False

    def _load_model(self) -> None:
        """Load the model into memory (lazy loading)."""
        if self._loaded:
            return

        pipeline_fn, torch = _get_pipeline()

        # Determine device
        device = self.config.get("device")
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        # Determine dtype
        dtype_str = self.config.get("dtype", "bfloat16")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        logger.info(f"Loading Apertus model {self.model} on device={device}, dtype={dtype_str}")

        try:
            self._text_pipeline = pipeline_fn(
                task="text-generation",
                model=self.model,
                torch_dtype=dtype,
                device=device,
            )
            self._loaded = True
            logger.info(f"Apertus model {self.model} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Apertus model: {e}")
            raise

    def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Apertus.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max output tokens (default: 512)
            **kwargs: Additional generation parameters

        Returns:
            LLM response

        Raises:
            Exception: If generation fails
        """
        # Load model on first use
        self._load_model()

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.insert(0, f"System: {msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}\n")

        prompt = "".join(prompt_parts) + "Assistant:"

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens or 512,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self._text_pipeline.tokenizer.eos_token_id,
        }

        # Add any extra kwargs
        gen_kwargs.update(kwargs.get("generation_config", {}))

        try:
            outputs = self._text_pipeline(prompt, **gen_kwargs)
            generated_text = outputs[0]["generated_text"]

            # Extract only the new content (after the prompt)
            response_text = generated_text[len(prompt) :].strip()

            # Rough token estimation
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response_text) // 4

            return LLMResponse(
                text=response_text,
                model=self.model,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                provider="apertus",
                metadata={"device": str(self.config.get("device", "auto"))},
            )

        except Exception as e:
            raise Exception(f"Apertus generation failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Apertus is available.

        Returns True if:
        1. APERTUS_ENABLED env var is set to true
        2. transformers and torch are installed
        3. (Optionally) CUDA is available for GPU models
        """
        # Check if explicitly enabled
        enabled = os.environ.get("APERTUS_ENABLED", "false").lower()
        if enabled not in ("true", "1", "yes", "on"):
            return False

        # Check if dependencies are available
        try:
            _get_pipeline()
        except ImportError:
            return False

        return True

    @property
    def name(self) -> str:
        """Provider name."""
        return "apertus"

    @property
    def supports_images(self) -> bool:
        """Apertus is text-only."""
        return False

    def estimate_cost(self, messages: list[LLMMessage]) -> float:
        """Local model has no API cost.

        Returns:
            0.0 (local inference)
        """
        return 0.0
