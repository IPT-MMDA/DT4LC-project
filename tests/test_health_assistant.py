"""``GET /v1/health`` reports the assistant (LLM provider) status."""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from server.app import app
import server.routes.health as health


def _provider(name: str, model: str, available: bool) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.model = model
    provider.is_available.return_value = available
    return provider


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(health, "_assistant_cache", None)
    yield


def _with_providers(*providers: MagicMock) -> AbstractContextManager[MagicMock]:
    router = MagicMock()
    router.providers = list(providers)
    return patch("server.routes.health.get_llm_router", return_value=router)


def test_no_provider_is_offline(client: TestClient) -> None:
    with _with_providers():
        body = client.get("/v1/health").json()
    assert body["ok"] is True
    assert body["assistant"] == {"connected": False, "offline": True, "provider": None, "model": None}


def test_unavailable_providers_are_offline(client: TestClient) -> None:
    with _with_providers(_provider("ollama", "llama3.2", False)):
        body = client.get("/v1/health").json()
    assert body["assistant"] == {"connected": False, "offline": True, "provider": None, "model": None}


def test_first_available_provider_is_reported(client: TestClient) -> None:
    down = _provider("groq", "llama-3.3-70b-versatile", False)
    up = _provider("gemini", "gemini-2.5-flash", True)
    later = _provider("ollama", "llama3.2", True)
    with _with_providers(down, up, later):
        body = client.get("/v1/health").json()
    expected = {"connected": True, "offline": False, "provider": "gemini", "model": "gemini-2.5-flash"}
    assert body["assistant"] == expected
    later.is_available.assert_not_called()


def test_router_error_is_offline(client: TestClient) -> None:
    with patch("server.routes.health.get_llm_router", side_effect=ValueError("bad config")):
        body = client.get("/v1/health").json()
    assert body["assistant"]["connected"] is False


def test_detailed_health_keeps_assistant(client: TestClient) -> None:
    with (
        _with_providers(_provider("gemini", "gemini-2.5-flash", True)),
        patch("server.routes.health._LOCALHOST_HOSTS", {"testclient"}),
        patch("server.routes.health._collect_llm_providers", return_value=[]),
        patch("server.routes.health._collect_gee_status", return_value=None),
        patch("server.routes.health._collect_models_info", return_value=None),
        patch("server.routes.health._collect_disk_usage", return_value=None),
    ):
        body = client.get("/v1/health?detailed=true").json()
    assert body["assistant"]["provider"] == "gemini"
    assert body["llm_providers"] == []


def test_status_is_cached_for_the_ttl(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider("ollama", "llama3.2", True)
    clock = [1000.0]
    monkeypatch.setattr(health.time, "monotonic", lambda: clock[0])
    with _with_providers(provider):
        client.get("/v1/health")
        clock[0] += health.ASSISTANT_STATUS_TTL_S - 1
        client.get("/v1/health")
        assert provider.is_available.call_count == 1
        clock[0] += 1
        client.get("/v1/health")
        assert provider.is_available.call_count == 2


def test_openapi_documents_assistant() -> None:
    schemas = app.openapi()["components"]["schemas"]
    assert "assistant" in schemas["HealthResponse"]["properties"]
    assert set(schemas["AssistantStatus"]["required"]) == {"connected", "offline"}
    assert set(schemas["AssistantStatus"]["properties"]) == {"connected", "offline", "provider", "model"}
