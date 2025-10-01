"""LLM Backend Abstraction Layer.

This package provides a unified interface for multiple LLM providers with
automatic fallback, load balancing, and cost optimization.
"""

from .base import BaseLLMProvider, LLMMessage, LLMResponse
from .router import LLMRouter

__all__ = ["BaseLLMProvider", "LLMMessage", "LLMResponse", "LLMRouter"]
