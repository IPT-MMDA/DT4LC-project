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

    @pytest.mark.llm
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


class TestAnthropicProvider:
    """Tests for Anthropic Claude LLM provider."""

    def test_initialization(self) -> None:
        """Test Anthropic provider can be initialized with default model."""
        from dta.dti.coe.llm.anthropic import DEFAULT_ANTHROPIC_MODEL, AnthropicProvider

        provider = AnthropicProvider()
        assert provider.name == "anthropic"
        assert provider.model == DEFAULT_ANTHROPIC_MODEL
        assert provider.supports_images is True

    def test_is_available_without_key(self) -> None:
        """Test Anthropic availability check without API key."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        old_key = os.environ.get("ANTHROPIC_API_KEY")
        if old_key:
            del os.environ["ANTHROPIC_API_KEY"]
        try:
            provider = AnthropicProvider("claude-sonnet-4-6")
            assert provider.is_available() is False
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_is_available_with_explicit_key(self) -> None:
        """Test Anthropic availability when api_key is passed via config."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider("claude-sonnet-4-6", api_key="sk-ant-test-key")
        assert provider.is_available() is True

    def test_estimate_cost_known_model(self) -> None:
        """Test Anthropic cost estimation uses pricing table."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider("claude-sonnet-4-6")
        messages = [LLMMessage(role="user", content="Hello " * 1000)]
        cost = provider.estimate_cost(messages)
        # Sonnet 4.6 input is $3/Mtok; ~6000 chars / 4 = ~1500 tokens → ~$0.0045
        assert cost > 0
        assert cost < 0.05

    def test_estimate_cost_unknown_model_returns_zero(self) -> None:
        """Unknown model IDs fall back to 0.0 cost (don't break routing)."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider("claude-some-future-model")
        cost = provider.estimate_cost([LLMMessage(role="user", content="Hi")])
        assert cost == 0.0

    def test_split_messages_extracts_system(self) -> None:
        """System messages are extracted into Anthropic's top-level system param."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="system", content="Be concise."),
            LLMMessage(role="user", content="Hi"),
            LLMMessage(role="assistant", content="Hello."),
        ]
        system_text, api_messages = AnthropicProvider._split_messages(messages)
        assert "You are helpful." in system_text
        assert "Be concise." in system_text
        assert len(api_messages) == 2
        assert api_messages[0]["role"] == "user"
        assert api_messages[1]["role"] == "assistant"

    def test_opus_4_7_omits_sampling_params(self) -> None:
        """Opus 4.7 rejects temperature/top_p — provider must skip them."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        opus_47 = AnthropicProvider("claude-opus-4-7", api_key="sk-ant-test")
        sonnet_46 = AnthropicProvider("claude-sonnet-4-6", api_key="sk-ant-test")
        assert opus_47._supports_sampling() is False
        assert sonnet_46._supports_sampling() is True

    def test_generate_passes_cache_control_on_system(self) -> None:
        """Verify cache_control: ephemeral is attached to the system block."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        provider = AnthropicProvider("claude-sonnet-4-6", api_key="sk-ant-test")
        # Stub the Anthropic SDK client to capture the request payload.
        captured: dict[str, object] = {}
        fake_response = MagicMock()
        fake_response.content = [MagicMock(type="text", text="ok")]
        fake_response.model = "claude-sonnet-4-6"
        fake_response.stop_reason = "end_turn"
        fake_response.usage = MagicMock(
            input_tokens=10,
            output_tokens=2,
            cache_creation_input_tokens=8,
            cache_read_input_tokens=0,
        )
        fake_client = MagicMock()
        fake_client.messages.create.side_effect = lambda **kw: (captured.update(kw) or fake_response)
        provider._client = fake_client

        messages = [
            LLMMessage(role="system", content="You are a planner."),
            LLMMessage(role="user", content="Plan."),
        ]
        response = provider.generate(messages)

        # System param is a list with cache_control: ephemeral on the first block.
        system_param = captured.get("system")
        assert isinstance(system_param, list)
        assert system_param[0]["cache_control"] == {"type": "ephemeral"}
        assert system_param[0]["text"] == "You are a planner."
        # Cache stats are surfaced in metadata for verification.
        assert response.metadata is not None
        assert response.metadata["cache_creation_input_tokens"] == 8


class TestMistralProvider:
    """Tests for Mistral AI LLM provider."""

    def test_initialization(self) -> None:
        """Test Mistral provider can be initialized."""
        from dta.dti.coe.llm.mistral import DEFAULT_MISTRAL_MODEL, MistralProvider

        provider = MistralProvider()
        assert provider.name == "mistral"
        assert provider.model == DEFAULT_MISTRAL_MODEL
        assert provider.supports_images is False

    def test_is_available_without_key(self) -> None:
        """Test Mistral availability check without API key."""
        from dta.dti.coe.llm.mistral import MistralProvider

        old_key = os.environ.get("MISTRAL_API_KEY")
        if old_key:
            del os.environ["MISTRAL_API_KEY"]
        try:
            provider = MistralProvider("mistral-medium-latest")
            assert provider.is_available() is False
        finally:
            if old_key:
                os.environ["MISTRAL_API_KEY"] = old_key

    def test_is_available_with_explicit_key(self) -> None:
        """Test Mistral availability when api_key is passed."""
        from dta.dti.coe.llm.mistral import MistralProvider

        provider = MistralProvider("mistral-medium-latest", api_key="test-key")
        assert provider.is_available() is True

    def test_estimate_cost_known_model(self) -> None:
        """Test Mistral cost estimation uses pricing table."""
        from dta.dti.coe.llm.mistral import MistralProvider

        provider = MistralProvider("mistral-medium-latest")
        messages = [LLMMessage(role="user", content="Hello " * 1000)]
        cost = provider.estimate_cost(messages)
        # mistral-medium input is $0.40/Mtok; ~6000 chars / 4 = ~1500 tokens
        assert cost > 0
        assert cost < 0.01

    def test_estimate_cost_unknown_model_returns_zero(self) -> None:
        """Unknown model IDs fall back to 0.0 cost."""
        from dta.dti.coe.llm.mistral import MistralProvider

        provider = MistralProvider("mistral-some-future-model")
        cost = provider.estimate_cost([LLMMessage(role="user", content="Hi")])
        assert cost == 0.0

    def test_generate_without_key_raises(self) -> None:
        """Without an API key, generate() raises a clear error."""
        from dta.dti.coe.llm.mistral import MistralProvider

        old_key = os.environ.get("MISTRAL_API_KEY")
        if old_key:
            del os.environ["MISTRAL_API_KEY"]
        try:
            provider = MistralProvider("mistral-medium-latest")
            with pytest.raises(Exception, match="MISTRAL_API_KEY"):
                provider.generate([LLMMessage(role="user", content="Hi")])
        finally:
            if old_key:
                os.environ["MISTRAL_API_KEY"] = old_key


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

    def test_anthropic_in_config_when_key_and_order_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting ANTHROPIC_API_KEY + adding 'anthropic' to LLM_PROVIDER_ORDER
        produces an anthropic entry in the default config."""
        from dta.dti.coe.llm.config import get_default_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,gemini,ollama")
        monkeypatch.setenv("ANTHROPIC_MODELS", "claude-sonnet-4-6")

        config = get_default_config()
        anthropic_entries = [p for p in config["providers"] if p["type"] == "anthropic"]
        assert len(anthropic_entries) == 1
        assert anthropic_entries[0]["model"] == "claude-sonnet-4-6"

    def test_mistral_in_config_when_key_and_order_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting MISTRAL_API_KEY + adding 'mistral' to LLM_PROVIDER_ORDER
        produces a mistral entry in the default config."""
        from dta.dti.coe.llm.config import get_default_config

        monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,gemini,ollama")
        monkeypatch.setenv("MISTRAL_MODELS", "mistral-large-latest")

        config = get_default_config()
        mistral_entries = [p for p in config["providers"] if p["type"] == "mistral"]
        assert len(mistral_entries) == 1
        assert mistral_entries[0]["model"] == "mistral-large-latest"

    def test_anthropic_skipped_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anthropic in LLM_PROVIDER_ORDER without ANTHROPIC_API_KEY is dropped."""
        from dta.dti.coe.llm.config import get_default_config

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,ollama")

        config = get_default_config()
        types = [p["type"] for p in config["providers"]]
        assert "anthropic" not in types

    @pytest.mark.llm
    def test_router_fallback_works(self) -> None:
        """Test that router successfully falls back when primary provider fails."""
        from dta.dti.coe.llm.config import create_router_from_env

        router = create_router_from_env()
        messages = [LLMMessage(role="user", content="Say 'hello' and nothing else")]

        response = router.generate(messages, temperature=0.0)

        assert response.text is not None
        assert len(response.text) > 0
        assert response.provider in ["gemini", "ollama"]


@pytest.mark.llm
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
