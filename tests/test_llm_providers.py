"""Tests for LLM providers using mocks."""

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.coe.llm import LLMMessage, LLMResponse, LLMRouter
from dta.dti.coe.llm.apertus import ApertusProvider
from dta.dti.coe.llm.base import BaseLLMProvider
from dta.dti.coe.llm.gemini import GeminiProvider
from dta.dti.coe.llm.groq import GroqProvider
from dta.dti.coe.llm.ollama import OllamaProvider


def _configured_gemini_models() -> list[str]:
    """Models the app would use: GEMINI_MODELS (loaded from .env by conftest) or the code default."""
    from dta.dti.coe.llm.config import _get_models_list
    from dta.dti.coe.llm.gemini import DEFAULT_GEMINI_MODEL

    return _get_models_list("GEMINI_MODELS", DEFAULT_GEMINI_MODEL)


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

    def test_default_model_is_current(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Code default and env-config default are the same current model, not retired 2.0."""
        from dta.dti.coe.llm.config import get_default_config
        from dta.dti.coe.llm.gemini import DEFAULT_GEMINI_MODEL

        assert GeminiProvider().model == DEFAULT_GEMINI_MODEL
        assert not DEFAULT_GEMINI_MODEL.startswith("gemini-2.0")

        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.delenv("GEMINI_MODELS", raising=False)
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini")
        monkeypatch.setenv("LLM_ENABLE_GEMINI", "true")
        assert get_default_config()["providers"] == [{"type": "gemini", "model": DEFAULT_GEMINI_MODEL}]

    def test_generate_request_shape(self) -> None:
        """System text leads the contents; temperature/max_tokens go into GenerateContentConfig."""
        provider = GeminiProvider("gemini-3.8-flash", api_key="test-gemini-key")
        captured: dict[str, object] = {}
        fake_response = MagicMock(text="ok", usage_metadata=None, candidates=[])
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = lambda **kw: captured.update(kw) or fake_response
        provider._client = fake_client

        messages = [
            LLMMessage(role="system", content="You are a planner."),
            LLMMessage(role="user", content="Plan."),
        ]
        response = provider.generate(messages, temperature=0.2, max_tokens=50)

        assert captured["model"] == "gemini-3.8-flash"
        assert captured["contents"] == ["You are a planner.", "Plan."]
        config = captured["config"]
        assert config.temperature == 0.2
        assert config.max_output_tokens == 50
        assert response.text == "ok"
        assert response.provider == "gemini"

    def test_sdk_error_reaches_router_message(self) -> None:
        """A retired model's 404 surfaces with its SDK type and text, not 'Last error: None'."""
        from google.genai import errors

        provider = GeminiProvider("gemini-2.0-flash-exp", api_key="test-gemini-key")
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = errors.ClientError(
            404, {"error": {"code": 404, "message": "models/gemini-2.0-flash-exp is not found", "status": "NOT_FOUND"}}
        )
        provider._client = fake_client

        with pytest.raises(Exception, match="All LLM providers failed") as exc_info:
            LLMRouter([provider]).generate([LLMMessage(role="user", content="Hi")])

        message = str(exc_info.value)
        assert "gemini (gemini-2.0-flash-exp): ClientError: Gemini generation failed: 404 NOT_FOUND" in message
        assert "None" not in message

    @pytest.mark.llm
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    @pytest.mark.parametrize("model", _configured_gemini_models())
    def test_generation_real(self, model: str) -> None:
        """Live smoke for every model configured in GEMINI_MODELS (requires API key)."""
        provider = GeminiProvider(model)
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

    pytest.importorskip("anthropic")

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

    def test_claude_5_models_omit_sampling_params(self) -> None:
        """Claude 5.x models reject temperature ("deprecated for this model")."""
        from dta.dti.coe.llm.anthropic import AnthropicProvider

        for model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-5-1"):
            assert AnthropicProvider(model, api_key="sk-ant-test")._supports_sampling() is False, model
        for model in ("claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6"):
            assert AnthropicProvider(model, api_key="sk-ant-test")._supports_sampling() is True, model

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
        fake_client.messages.create.side_effect = lambda **kw: captured.update(kw) or fake_response
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


def _configured_openai_models() -> list[str]:
    """Models the app would use: OPENAI_MODELS (loaded from .env by conftest) or the code default."""
    from dta.dti.coe.llm.config import _get_models_list

    try:
        from dta.dti.coe.llm.openai import DEFAULT_OPENAI_MODEL
    except ImportError:  # openai SDK is part of the `server` extra
        return []
    return _get_models_list("OPENAI_MODELS", DEFAULT_OPENAI_MODEL)


@pytest.mark.skipif(importlib.util.find_spec("openai") is None, reason="openai not installed")
class TestOpenAIProvider:
    """Tests for the OpenAI Chat Completions provider."""

    @staticmethod
    def _fake_client(captured: dict[str, object]) -> MagicMock:
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(finish_reason="stop")]
        fake_response.choices[0].message.content = "ok"
        fake_response.model = "gpt-4.1-nano-2025-04-14"
        fake_response.usage = MagicMock(prompt_tokens=10, completion_tokens=2, total_tokens=12)
        client = MagicMock()
        client.chat.completions.create.side_effect = lambda **kw: captured.update(kw) or fake_response
        return client

    def test_initialization(self) -> None:
        """Default model is the module constant; images are not forwarded (like Anthropic)."""
        from dta.dti.coe.llm.openai import DEFAULT_OPENAI_MODEL, OpenAIProvider

        provider = OpenAIProvider()
        assert provider.name == "openai"
        assert provider.model == DEFAULT_OPENAI_MODEL
        assert provider.supports_images is False

    def test_unavailable_without_any_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither OPENAI_API_KEY nor GPT_SECRET → unavailable, generate() names both."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GPT_SECRET", raising=False)
        provider = OpenAIProvider()
        assert provider.is_available() is False
        with pytest.raises(ValueError, match="OPENAI_API_KEY \\(or GPT_SECRET\\)"):
            provider.generate([LLMMessage(role="user", content="Hi")])

    def test_gpt_secret_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GPT_SECRET is used when OPENAI_API_KEY is unset or empty; OPENAI_API_KEY wins otherwise."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("GPT_SECRET", "sk-gpt-secret-test")
        with patch("dta.dti.coe.llm.openai.openai.OpenAI") as client_cls:
            provider = OpenAIProvider()
            assert provider.is_available() is True
            provider._get_client()
        assert client_cls.call_args.kwargs["api_key"] == "sk-gpt-secret-test"

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        with patch("dta.dti.coe.llm.openai.openai.OpenAI") as client_cls:
            OpenAIProvider()._get_client()
        assert client_cls.call_args.kwargs["api_key"] == "sk-openai-test"

    def test_reasoning_models_omit_sampling_params(self) -> None:
        """Models observed to 400 on temperature are recognised; the rest keep it."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        for model in (
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-nano-2025-08-07",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-6-sol",
            "gpt-6.1-sol",
            "chat-latest",
        ):
            assert OpenAIProvider(model, api_key="sk-test")._supports_sampling() is False, model
        for model in ("gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-nano"):
            assert OpenAIProvider(model, api_key="sk-test")._supports_sampling() is True, model

    def test_generate_request_shape(self) -> None:
        """System stays in messages; max_tokens → max_completion_tokens; temperature for chat models."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        provider = OpenAIProvider("gpt-4.1-nano", api_key="sk-test")
        captured: dict[str, object] = {}
        provider._client = self._fake_client(captured)

        messages = [
            LLMMessage(role="system", content="You are a planner."),
            LLMMessage(role="user", content="Plan."),
        ]
        response = provider.generate(messages, temperature=0.2, max_tokens=50)

        assert captured == {
            "model": "gpt-4.1-nano",
            "messages": [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "Plan."},
            ],
            "max_completion_tokens": 50,
            "temperature": 0.2,
        }
        assert response.text == "ok"
        assert response.provider == "openai"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        assert response.metadata == {"finish_reason": "stop"}

    def test_reasoning_model_request_has_no_temperature(self) -> None:
        """gpt-5-nano rejects temperature=0.2 with a 400, so it must not be sent."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        provider = OpenAIProvider("gpt-5-nano", api_key="sk-test")
        captured: dict[str, object] = {}
        provider._client = self._fake_client(captured)

        provider.generate([LLMMessage(role="user", content="Hi")], temperature=0.2)

        assert "temperature" not in captured
        assert "max_tokens" not in captured

    def test_api_error_is_wrapped_with_cause(self) -> None:
        """SDK errors are re-raised as 'OpenAI generation failed' with the SDK error chained."""
        import httpx
        import openai

        from dta.dti.coe.llm.openai import OpenAIProvider

        provider = OpenAIProvider("gpt-4.1-nano", api_key="sk-test")
        sdk_error = openai.APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )
        provider._client = MagicMock()
        provider._client.chat.completions.create.side_effect = sdk_error

        with pytest.raises(Exception, match="OpenAI generation failed") as exc_info:
            provider.generate([LLMMessage(role="user", content="Hi")])
        assert exc_info.value.__cause__ is sdk_error

    @pytest.mark.llm
    @pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_SECRET")),
        reason="OPENAI_API_KEY / GPT_SECRET not set",
    )
    @pytest.mark.parametrize("model", _configured_openai_models())
    def test_generation_real(self, model: str) -> None:
        """Live smoke for every model configured in OPENAI_MODELS (requires API key)."""
        from dta.dti.coe.llm.openai import OpenAIProvider

        provider = OpenAIProvider(model)
        response = provider.generate(
            [LLMMessage(role="user", content="Say 'test successful' and nothing else")],
            temperature=0.0,
            max_tokens=200,
        )

        assert "test" in response.text.lower()
        assert response.provider == "openai"


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
        pytest.importorskip("anthropic")
        pytest.importorskip("openai")  # from_config imports every provider SDK
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

    def test_all_failed_reports_every_underlying_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """The final error names each provider's exception type and message and chains the last one."""
        first = MagicMock(spec=BaseLLMProvider, model="m1")
        first.name = "first"
        first.is_available.return_value = True
        first.generate.side_effect = ValueError("bad request shape")

        cause = TimeoutError("read timed out")
        second = MagicMock(spec=BaseLLMProvider, model="m2")
        second.name = "second"
        second.is_available.return_value = True
        wrapped = Exception("Second generation failed: read timed out")
        wrapped.__cause__ = cause
        second.generate.side_effect = wrapped

        with caplog.at_level("WARNING", logger="dta.dti.coe.llm.router"):
            with pytest.raises(Exception, match="All LLM providers failed") as exc_info:
                LLMRouter([first, second]).generate([LLMMessage(role="user", content="Hi")])

        message = str(exc_info.value)
        assert "first (m1): ValueError: bad request shape" in message
        assert "second (m2): TimeoutError: Second generation failed: read timed out" in message
        assert exc_info.value.__cause__ is wrapped
        assert "ValueError: bad request shape" in caplog.text
        assert "TimeoutError: Second generation failed" in caplog.text

    def test_all_unavailable_says_so(self) -> None:
        """When every provider is skipped the message says why, instead of 'Last error: None'."""
        offline = MagicMock(spec=BaseLLMProvider, model="llama3.2")
        offline.name = "ollama"
        offline.is_available.return_value = False

        with pytest.raises(Exception, match="All LLM providers failed") as exc_info:
            LLMRouter([offline]).generate([LLMMessage(role="user", content="Hi")])

        assert str(exc_info.value) == "All LLM providers failed: ollama (llama3.2): not available"

    def test_key_fragments_redacted_from_log_and_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """OpenAI's 401 echoes the key prefix and last 4 chars; neither may reach users or logs."""
        provider = MagicMock(spec=BaseLLMProvider, model="gpt-4o-mini")
        provider.name = "openai"
        provider.is_available.return_value = True
        provider.generate.side_effect = Exception(
            "OpenAI generation failed: Error code: 401 - {'error': {'message': "
            "'Incorrect API key provided: sk-proj-***********************WXYZ. You can find your API key at "
            "https://platform.openai.com/account/api-keys.'}} "
            "(also sk-ant-FAKEKEYabcd1234, gsk_FAKEKEYabcd1234, AIzaFAKEKEYabcd1234)"
        )

        with caplog.at_level("WARNING", logger="dta.dti.coe.llm.router"):
            with pytest.raises(Exception, match="All LLM providers failed") as exc_info:
                LLMRouter([provider]).generate([LLMMessage(role="user", content="Hi")])

        for text in (str(exc_info.value), caplog.text):
            assert "Error code: 401" in text
            assert "Incorrect API key provided: ***" in text
            for fragment in ("WXYZ", "sk-proj", "FAKEKEY", "gsk_", "AIza"):
                assert fragment not in text, fragment


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
        pytest.importorskip("anthropic")
        pytest.importorskip("openai")  # from_config imports every provider SDK
        from dta.dti.coe.llm.config import create_router_from_env

        router = create_router_from_env()

        assert router is not None
        assert len(router.providers) > 0

    def test_anthropic_in_config_when_key_and_order_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting ANTHROPIC_API_KEY + adding 'anthropic' to LLM_PROVIDER_ORDER
        produces an anthropic entry in the default config."""
        pytest.importorskip("anthropic")
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

    def test_openai_selected_through_provider_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GPT_SECRET alone enables openai; LLM_PROVIDER_ORDER sets its position; router builds it."""
        pytest.importorskip("openai")
        from dta.dti.coe.llm.config import create_router_from_env, get_default_config
        from dta.dti.coe.llm.openai import DEFAULT_OPENAI_MODEL, OpenAIProvider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODELS", raising=False)
        monkeypatch.setenv("GPT_SECRET", "sk-gpt-secret-test")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setenv("GEMINI_MODELS", "gemini-3.8-flash")
        monkeypatch.setenv("LLM_ENABLE_OPENAI", "true")
        monkeypatch.setenv("LLM_ENABLE_GEMINI", "true")

        monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,openai")
        assert [p["type"] for p in get_default_config()["providers"]] == ["gemini", "openai"]

        monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,gemini")
        providers = get_default_config()["providers"]
        assert providers[0] == {"type": "openai", "model": DEFAULT_OPENAI_MODEL}
        router = create_router_from_env()
        assert isinstance(router.providers[0], OpenAIProvider)
        assert [p.name for p in router.providers] == ["openai", "gemini"]

        monkeypatch.setenv("OPENAI_MODELS", "gpt-5.4-nano")
        assert get_default_config()["providers"][0]["model"] == "gpt-5.4-nano"

        monkeypatch.setenv("LLM_ENABLE_OPENAI", "false")
        assert [p["type"] for p in get_default_config()["providers"]] == ["gemini"]

    def test_openai_skipped_without_any_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """openai in LLM_PROVIDER_ORDER without OPENAI_API_KEY or GPT_SECRET is dropped."""
        from dta.dti.coe.llm.config import get_default_config

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GPT_SECRET", raising=False)
        monkeypatch.setenv("LLM_PROVIDER_ORDER", "openai,ollama")

        assert "openai" not in [p["type"] for p in get_default_config()["providers"]]

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


def test_apertus_generate() -> None:
    pytest.importorskip("transformers")
    provider = ApertusProvider(api_key="fake", api_url="http://fake")
    messages = [LLMMessage(role="user", content="hello")]

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"generated_text": "User: hello\nAssistant: world"}]
    fake_pipeline.tokenizer = MagicMock()
    fake_pipeline.tokenizer.eos_token_id = 0

    with patch("dta.dti.coe.llm.apertus._get_pipeline", return_value=(lambda *a, **kw: fake_pipeline, MagicMock())):
        provider._loaded = True
        provider._text_pipeline = fake_pipeline
        res = provider.generate(messages)

    assert res.provider == "apertus"


def test_apertus_generate_error() -> None:
    pytest.importorskip("transformers")
    provider = ApertusProvider(api_key="fake")
    messages = [LLMMessage(role="user", content="hello")]

    fake_pipeline = MagicMock()
    fake_pipeline.side_effect = Exception("Network error")
    fake_pipeline.tokenizer = MagicMock()
    fake_pipeline.tokenizer.eos_token_id = 0

    provider._loaded = True
    provider._text_pipeline = fake_pipeline

    with pytest.raises(Exception, match="Network error"):
        provider.generate(messages)


def test_groq_generate() -> None:
    provider = GroqProvider(api_key="fake")
    messages = [LLMMessage(role="user", content="hello")]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("dta.dti.coe.llm.groq.httpx.Client", return_value=mock_client):
        res = provider.generate(messages)

    assert res.text == "world"
    assert res.provider == "groq"
