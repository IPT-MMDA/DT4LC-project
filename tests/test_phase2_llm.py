"""Phase 2 Tests - LLM Backend Infrastructure."""

import os
from unittest.mock import MagicMock

import pytest

from dta.dti.coe.llm import LLMMessage, LLMResponse, LLMRouter
from dta.dti.coe.llm.base import BaseLLMProvider
from dta.dti.coe.llm.gemini import GeminiProvider
from dta.dti.coe.llm.ollama import OllamaProvider


def test_llm_message_creation() -> None:
    """Test LLMMessage dataclass."""
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.images is None


def test_llm_response_creation() -> None:
    """Test LLMResponse dataclass."""
    resp = LLMResponse(
        text="Hello back!",
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    assert resp.text == "Hello back!"
    assert resp.model == "test-model"
    assert resp.provider == "test"
    assert resp.usage["total_tokens"] == 15


def test_gemini_provider_initialization() -> None:
    """Test Gemini provider can be initialized."""
    provider = GeminiProvider("gemini-2.0-flash-exp")
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.0-flash-exp"
    assert provider.supports_images is True


def test_gemini_is_available_without_key() -> None:
    """Test Gemini availability check without API key."""
    # Temporarily clear API key
    old_key = os.environ.get("GEMINI_API_KEY")
    if old_key:
        del os.environ["GEMINI_API_KEY"]

    try:
        provider = GeminiProvider("gemini-2.0-flash-exp")
        assert provider.is_available() is False
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key


def test_gemini_is_available_with_key() -> None:
    """Test Gemini availability check with API key."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    provider = GeminiProvider("gemini-2.0-flash-exp")
    assert provider.is_available() is True


def test_gemini_estimate_cost() -> None:
    """Test Gemini cost estimation."""
    provider = GeminiProvider("gemini-2.0-flash-exp")
    messages = [LLMMessage(role="user", content="Hello " * 100)]  # ~500 chars

    cost = provider.estimate_cost(messages)
    assert cost > 0  # Should have some cost
    assert cost < 0.01  # Should be very small for this short request


def test_ollama_provider_initialization() -> None:
    """Test Ollama provider can be initialized."""
    provider = OllamaProvider("llama3.2")
    assert provider.name == "ollama"
    assert provider.model == "llama3.2"
    assert provider.base_url == "http://localhost:11434"


def test_ollama_estimate_cost() -> None:
    """Test Ollama cost is always zero."""
    provider = OllamaProvider("llama3.2")
    messages = [LLMMessage(role="user", content="Hello")]

    cost = provider.estimate_cost(messages)
    assert cost == 0.0  # Local models are free


def test_router_initialization() -> None:
    """Test LLM router can be initialized."""
    providers = [
        GeminiProvider("gemini-2.0-flash-exp"),
        OllamaProvider("llama3.2"),
    ]
    router = LLMRouter(providers)

    assert len(router.providers) == 2
    assert router.strategy == "fallback"


def test_router_from_config() -> None:
    """Test router creation from config dict."""
    config = {
        "providers": [
            {"type": "gemini", "model": "gemini-2.0-flash-exp"},
            {"type": "ollama", "model": "llama3.2"},
        ],
        "strategy": "fallback",
    }

    router = LLMRouter.from_config(config)

    assert len(router.providers) == 2
    assert router.providers[0].name == "gemini"
    assert router.providers[1].name == "ollama"


def test_router_get_available_providers() -> None:
    """Test getting available providers."""
    # Create mock providers
    available_provider = MagicMock(spec=BaseLLMProvider)
    available_provider.is_available.return_value = True
    available_provider.name = "available"

    unavailable_provider = MagicMock(spec=BaseLLMProvider)
    unavailable_provider.is_available.return_value = False
    unavailable_provider.name = "unavailable"

    router = LLMRouter([available_provider, unavailable_provider])
    available = router.get_available_providers()

    assert len(available) == 1
    assert available[0].name == "available"


def test_router_estimate_cost() -> None:
    """Test router cost estimation for all providers."""
    providers = [
        GeminiProvider("gemini-2.0-flash-exp"),
        OllamaProvider("llama3.2"),
    ]
    router = LLMRouter(providers)

    messages = [LLMMessage(role="user", content="Hello")]
    costs = router.estimate_cost(messages)

    assert "gemini" in costs
    assert "ollama" in costs
    assert costs["gemini"] > 0
    assert costs["ollama"] == 0


def test_router_fallback_generation_mock() -> None:
    """Test router fallback with mocked providers."""
    # Create mock providers
    failing_provider = MagicMock(spec=BaseLLMProvider)
    failing_provider.is_available.return_value = True
    failing_provider.name = "failing"
    failing_provider.model = "failing-model"
    failing_provider.generate.side_effect = Exception("Provider failed")

    success_provider = MagicMock(spec=BaseLLMProvider)
    success_provider.is_available.return_value = True
    success_provider.name = "success"
    success_provider.model = "success-model"
    success_provider.generate.return_value = LLMResponse(
        text="Success!",
        model="test-model",
        provider="success",
    )

    router = LLMRouter([failing_provider, success_provider], strategy="fallback")
    messages = [LLMMessage(role="user", content="Test")]

    response = router.generate(messages)

    assert response.text == "Success!"
    assert response.provider == "success"
    assert failing_provider.generate.called
    assert success_provider.generate.called


def test_config_get_default() -> None:
    """Test default LLM configuration generation."""
    from dta.dti.coe.llm.config import get_default_config

    config = get_default_config()

    assert "providers" in config
    assert "strategy" in config
    assert len(config["providers"]) >= 1  # At least Ollama
    assert config["strategy"] == "fallback"


def test_config_create_router_from_env() -> None:
    """Test router creation from environment."""
    from dta.dti.coe.llm.config import create_router_from_env

    router = create_router_from_env()

    assert router is not None
    assert len(router.providers) > 0


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_generation_real() -> None:
    """Test real Gemini generation (requires API key)."""
    provider = GeminiProvider("gemini-2.0-flash-exp")
    messages = [LLMMessage(role="user", content="Say 'test successful' and nothing else")]

    response = provider.generate(messages, temperature=0.0)

    assert response.text is not None
    assert len(response.text) > 0
    assert "test" in response.text.lower()
    assert response.provider == "gemini"
