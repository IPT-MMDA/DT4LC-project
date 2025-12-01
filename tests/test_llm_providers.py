"""Tests for LLM provider infrastructure.

Tests LLM providers (Gemini, Ollama), router, and configuration.
"""

import os
from unittest.mock import MagicMock

import pytest

from dta.dti.coe.llm import LLMMessage, LLMResponse, LLMRouter
from dta.dti.coe.llm.base import BaseLLMProvider
from dta.dti.coe.llm.gemini import GeminiProvider
from dta.dti.coe.llm.ollama import OllamaProvider


class TestLLMDataClasses:
    """Tests for LLM data classes."""

    def test_llm_message_creation(self) -> None:
        """Test LLMMessage dataclass."""
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_llm_response_creation(self) -> None:
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


class TestGeminiProvider:
    """Tests for Gemini LLM provider."""

    def test_initialization(self) -> None:
        """Test Gemini provider can be initialized."""
        provider = GeminiProvider("gemini-2.0-flash-exp")
        assert provider.name == "gemini"
        assert provider.model == "gemini-2.0-flash-exp"
        assert provider.supports_images is True

    def test_is_available_without_key(self) -> None:
        """Test Gemini availability check without API key."""
        old_key = os.environ.get("GEMINI_API_KEY")
        if old_key:
            del os.environ["GEMINI_API_KEY"]

        try:
            provider = GeminiProvider("gemini-2.0-flash-exp")
            assert provider.is_available() is False
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_is_available_with_key(self) -> None:
        """Test Gemini availability check with API key."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        provider = GeminiProvider("gemini-2.0-flash-exp")
        assert provider.is_available() is True

    def test_estimate_cost(self) -> None:
        """Test Gemini cost estimation."""
        provider = GeminiProvider("gemini-2.0-flash-exp")
        messages = [LLMMessage(role="user", content="Hello " * 100)]

        cost = provider.estimate_cost(messages)
        assert cost > 0
        assert cost < 0.01

    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_generation_real(self) -> None:
        """Test real Gemini generation (requires API key)."""
        provider = GeminiProvider("gemini-2.0-flash-exp")
        messages = [LLMMessage(role="user", content="Say 'test successful' and nothing else")]

        try:
            response = provider.generate(messages, temperature=0.0)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                pytest.skip("Gemini quota exceeded - router will use Ollama fallback in production")
            raise

        assert response.text is not None
        assert len(response.text) > 0
        assert "test" in response.text.lower()
        assert response.provider == "gemini"


class TestOllamaProvider:
    """Tests for Ollama LLM provider."""

    def test_initialization(self) -> None:
        """Test Ollama provider can be initialized."""
        provider = OllamaProvider("llama3.2")
        assert provider.name == "ollama"
        assert provider.model == "llama3.2"
        assert provider.base_url == "http://localhost:11434"

    def test_estimate_cost(self) -> None:
        """Test Ollama cost is always zero."""
        provider = OllamaProvider("llama3.2")
        messages = [LLMMessage(role="user", content="Hello")]

        cost = provider.estimate_cost(messages)
        assert cost == 0.0


class TestLLMRouter:
    """Tests for LLM router."""

    def test_initialization(self) -> None:
        """Test LLM router can be initialized."""
        providers = [
            GeminiProvider("gemini-2.0-flash-exp"),
            OllamaProvider("llama3.2"),
        ]
        router = LLMRouter(providers)

        assert len(router.providers) == 2
        assert router.strategy == "fallback"

    def test_from_config(self) -> None:
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

    def test_get_available_providers(self) -> None:
        """Test getting available providers."""
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

    def test_estimate_cost(self) -> None:
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

    def test_fallback_generation_mock(self) -> None:
        """Test router fallback with mocked providers."""
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


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_get_default_config(self) -> None:
        """Test default LLM configuration generation."""
        from dta.dti.coe.llm.config import get_default_config

        config = get_default_config()

        assert "providers" in config
        assert "strategy" in config
        assert len(config["providers"]) >= 1
        assert config["strategy"] == "fallback"

    def test_create_router_from_env(self) -> None:
        """Test router creation from environment."""
        from dta.dti.coe.llm.config import create_router_from_env

        router = create_router_from_env()

        assert router is not None
        assert len(router.providers) > 0

    def test_router_fallback_works(self) -> None:
        """Test that router successfully falls back when primary provider fails."""
        from dta.dti.coe.llm.config import create_router_from_env

        router = create_router_from_env()
        messages = [LLMMessage(role="user", content="Say 'hello' and nothing else")]

        response = router.generate(messages, temperature=0.0)

        assert response.text is not None
        assert len(response.text) > 0
        assert response.provider in ["gemini", "ollama"]


class TestOllamaContextAnalysis:
    """Tests for Ollama context analysis integration."""

    @pytest.mark.asyncio
    async def test_context_analysis_format(self) -> None:
        """Test that Ollama returns correctly formatted arrays."""
        from dta.dti.coe.context_agent import analyze
        from dta.dti.schemas import ChatRequest

        req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
        registry_types = ["Raster", "Features", "NDVIMap", "ChangeMap", "Statistics", "Insights"]

        result = analyze(req, registry_types)

        assert hasattr(result, "goal")
        assert hasattr(result, "desired_outputs")
        assert hasattr(result, "required_inputs")
        assert hasattr(result, "hints")

        assert isinstance(result.desired_outputs, list)
        assert isinstance(result.required_inputs, list)

        for output in result.desired_outputs:
            assert isinstance(output, str), f"desired_outputs contains non-string: {output}"

        for inp in result.required_inputs:
            assert isinstance(inp, str), f"required_inputs contains non-string: {inp}"

        assert isinstance(result.goal, str)
        assert len(result.goal) > 0

    @pytest.mark.asyncio
    async def test_various_prompts_format(self) -> None:
        """Test Ollama format consistency across different prompts."""
        from dta.dti.coe.context_agent import analyze
        from dta.dti.schemas import ChatRequest

        registry_types = ["Raster", "Features", "NDVIMap", "ChangeMap", "Statistics"]

        test_cases = [
            "calculate ndvi",
            "analyze vegetation changes",
            "load raster and compute statistics",
            "detect land cover changes",
        ]

        for prompt in test_cases:
            req = ChatRequest(prompt=prompt, attachments=[])
            result = analyze(req, registry_types)

            assert all(isinstance(x, str) for x in result.desired_outputs), f"Failed for prompt: {prompt}"
            assert all(isinstance(x, str) for x in result.required_inputs), f"Failed for prompt: {prompt}"

    def test_ollama_available(self) -> None:
        """Test that Ollama provider is available and configured."""
        from dta.dti.coe.llm.config import create_router_from_env

        router = create_router_from_env()

        assert len(router.providers) > 0

        available = [p for p in router.providers if p.is_available()]
        assert len(available) > 0, "No LLM providers available"

        ollama_providers = [p for p in router.providers if p.name == "ollama"]
        if ollama_providers:
            assert ollama_providers[0].is_available(), "Ollama configured but not available"
