"""Tests for the detailed health check endpoint (?detailed=true).

Covers localhost-only restriction, response structure, and individual
diagnostics collectors.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from server.app import app
from server.routes.health import (
    _collect_disk_usage,
    _collect_gee_status,
    _collect_llm_providers,
    _collect_models_info,
    _dir_size_bytes,
    _fmt_bytes,
)
from server.schemas import DiskEntry, DiskUsage, GEEStatus, LLMProviderStatus, ModelsInfo


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# Endpoint-level tests


class TestHealthEndpointDetailed:
    def test_simple_health_no_detailed(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data.get("llm_providers") is None

    def test_detailed_forbidden_from_non_localhost(self, client: TestClient) -> None:
        # TestClient sends requests with host "testclient", not 127.0.0.1
        resp = client.get("/v1/health", params={"detailed": "true"})
        assert resp.status_code == 403
        assert "localhost" in resp.json()["detail"]

    def test_detailed_returns_full_diagnostics_from_localhost(self, client: TestClient) -> None:
        gee_stub = GEEStatus(initialized=False, service_account_configured=False)
        models_stub = ModelsInfo()
        disk_stub = DiskUsage()
        with (
            patch("server.routes.health._LOCALHOST_HOSTS", {"testclient"}),
            patch("server.routes.health._collect_llm_providers", return_value=[]),
            patch("server.routes.health._collect_gee_status", return_value=gee_stub),
            patch("server.routes.health._collect_models_info", return_value=models_stub),
            patch("server.routes.health._collect_disk_usage", return_value=disk_stub),
        ):
            resp = client.get("/v1/health", params={"detailed": "true"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "llm_providers" in data
        assert "gee" in data
        assert "models" in data
        assert "disk" in data


# Unit tests for helper functions


class TestFmtBytes:
    def test_bytes(self) -> None:
        assert _fmt_bytes(512) == "512.0 B"

    def test_kilobytes(self) -> None:
        assert _fmt_bytes(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        assert _fmt_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self) -> None:
        assert _fmt_bytes(3 * 1024**3) == "3.0 GB"


class TestDirSizeBytes:
    def test_nonexistent_dir_returns_zero(self, tmp_path: Path) -> None:
        assert _dir_size_bytes(tmp_path / "nonexistent") == 0

    def test_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        assert _dir_size_bytes(tmp_path) == 0

    def test_counts_file_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "b.txt").write_bytes(b"world!")
        assert _dir_size_bytes(tmp_path) == 11

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.bin").write_bytes(b"x" * 100)
        assert _dir_size_bytes(tmp_path) == 100


class TestCollectGeeStatus:
    def test_returns_initialized_false_when_not_initialized(self) -> None:
        with patch("server.routes.health.is_initialized", return_value=False):
            status = _collect_gee_status()
        assert status.initialized is False
        assert status.error is None

    def test_returns_initialized_true_when_gee_connection_live(self) -> None:
        with (
            patch("server.routes.health.is_initialized", return_value=True),
            patch("ee.Number") as mock_num,
        ):
            mock_num.return_value.getInfo.return_value = 1
            status = _collect_gee_status()
        assert status.initialized is True
        assert status.error is None

    def test_returns_error_when_gee_connection_fails(self) -> None:
        with (
            patch("server.routes.health.is_initialized", return_value=True),
            patch("ee.Number") as mock_num,
        ):
            mock_num.return_value.getInfo.side_effect = Exception("timeout")
            status = _collect_gee_status()
        assert status.initialized is False
        assert status.error is not None

    def test_service_account_flag_reflects_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEE_SERVICE_ACCOUNT_KEY", "/path/to/key.json")
        with patch("server.routes.health.is_initialized", return_value=False):
            status = _collect_gee_status()
        assert status.service_account_configured is True

    def test_service_account_false_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEE_SERVICE_ACCOUNT_KEY", raising=False)
        with patch("server.routes.health.is_initialized", return_value=False):
            status = _collect_gee_status()
        assert status.service_account_configured is False


class TestCollectLLMProviders:
    def test_returns_list_on_success(self) -> None:
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.model = "gpt-4"
        mock_provider.is_available.return_value = True

        mock_router = MagicMock()
        mock_router.providers = [mock_provider]

        with (
            patch("server.routes.health.get_default_config", return_value=MagicMock()),
            patch("server.routes.health.LLMRouter.from_config", return_value=mock_router),
        ):
            result = _collect_llm_providers()

        assert len(result) == 1
        assert isinstance(result[0], LLMProviderStatus)
        assert result[0].name == "test"
        assert result[0].available is True

    def test_returns_error_entry_on_exception(self) -> None:
        with patch("server.routes.health.get_default_config", side_effect=ValueError("bad cfg")):
            result = _collect_llm_providers()
        assert len(result) == 1
        assert result[0].error is not None


class TestCollectModelsInfo:
    def test_returns_model_with_required_fields(self) -> None:
        mock_registry = MagicMock()
        mock_registry.list_all.return_value = []
        mock_registry.list_available.return_value = []

        with (
            patch("server.routes.health.get_model_registry", return_value=mock_registry),
            patch("server.routes.health.load_registry", side_effect=Exception("no yaml")),
        ):
            info = _collect_models_info()

        assert isinstance(info, ModelsInfo)
        assert info.total == 0
        assert info.available == 0
        assert info.models == []

    def test_counts_available_models(self) -> None:
        mock_model = MagicMock()
        mock_model.is_available.return_value = True

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = ["model-a"]
        mock_registry.list_available.return_value = ["model-a"]
        mock_registry.get.return_value = mock_model

        with (
            patch("server.routes.health.get_model_registry", return_value=mock_registry),
            patch("server.routes.health.load_registry", side_effect=Exception("no yaml")),
        ):
            info = _collect_models_info()

        assert info.total == 1
        assert info.available == 1
        assert info.models[0].id == "model-a"


class TestCollectDiskUsage:
    def test_returns_all_expected_partitions(self, tmp_path: Path) -> None:
        with (
            patch("server.routes.health.UPLOADS_PATH", tmp_path),
            patch("server.routes.health.CACHE_PATH", tmp_path),
            patch("server.routes.health.MODELS_PATH", tmp_path),
        ):
            usage = _collect_disk_usage()

        assert isinstance(usage, DiskUsage)
        assert usage.uploads is not None
        assert usage.cache is not None
        assert usage.models is not None
        assert usage.exports is not None

    def test_each_entry_has_bytes_and_human(self, tmp_path: Path) -> None:
        with (
            patch("server.routes.health.UPLOADS_PATH", tmp_path),
            patch("server.routes.health.CACHE_PATH", tmp_path),
            patch("server.routes.health.MODELS_PATH", tmp_path),
        ):
            usage = _collect_disk_usage()

        assert isinstance(usage.uploads, DiskEntry)
        assert isinstance(usage.uploads.bytes, int)
        assert isinstance(usage.uploads.human, str)
