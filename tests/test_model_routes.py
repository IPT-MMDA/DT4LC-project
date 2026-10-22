"""Tests for the ML models API routes."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from dta.dti.models import ModelStatus
from server.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestModelRoutes:
    """Tests for the /v1/ml-models endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        """Test listing models."""
        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [
            {"id": "test-model", "name": "Test", "size_mb": 100, "status": ModelStatus.AVAILABLE.value}
        ]
        mock_manager.cache_dir = "/tmp/cache"

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models")

        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 1
        assert data["total_installed_mb"] == 100

    def test_list_models_error(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.list_models.side_effect = Exception("DB error")

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models")

        assert resp.status_code == 500

    def test_get_model_found(self, client: TestClient) -> None:
        """Test getting a specific model."""
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model", "name": "Test"}

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models/test-model")

        assert resp.status_code == 200
        assert resp.json()["id"] == "test-model"

    def test_get_model_not_found(self, client: TestClient) -> None:
        """Test getting a nonexistent model."""
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = None

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models/nonexistent")

        assert resp.status_code == 404

    def test_start_download_success(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model", "name": "Test", "size_mb": 100}
        mock_manager.get_model_status.return_value = ModelStatus.NOT_INSTALLED
        mock_progress = MagicMock()
        mock_progress.status = ModelStatus.DOWNLOADING
        mock_manager.start_download.return_value = mock_progress

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.post("/v1/ml-models/test-model/download")

        assert resp.status_code == 202
        assert resp.json()["status"] == ModelStatus.DOWNLOADING.value

    def test_start_download_already_downloading(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model", "name": "Test"}
        mock_manager.get_model_status.return_value = ModelStatus.DOWNLOADING

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.post("/v1/ml-models/test-model/download")

        assert resp.status_code == 409

    def test_start_download_already_installed(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model", "name": "Test"}
        mock_manager.get_model_status.return_value = ModelStatus.AVAILABLE

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.post("/v1/ml-models/test-model/download")

        assert resp.status_code == 409

    def test_cancel_download_success(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model"}
        mock_manager.get_model_status.return_value = ModelStatus.DOWNLOADING
        mock_manager.cancel_download.return_value = True

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.post("/v1/ml-models/test-model/cancel")

        assert resp.status_code == 200

    def test_cancel_download_not_downloading(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model"}
        mock_manager.get_model_status.return_value = ModelStatus.AVAILABLE

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.post("/v1/ml-models/test-model/cancel")

        assert resp.status_code == 400

    def test_delete_model_success(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model", "name": "Test", "size_mb": 100}
        mock_manager.get_model_status.return_value = ModelStatus.AVAILABLE
        mock_manager.delete_model.return_value = True

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.delete("/v1/ml-models/test-model")

        assert resp.status_code == 200

    def test_delete_model_not_installed(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {"id": "test-model"}
        mock_manager.get_model_status.return_value = ModelStatus.NOT_INSTALLED

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.delete("/v1/ml-models/test-model")

        assert resp.status_code == 400

    def test_delete_model_not_found(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = None

        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.delete("/v1/ml-models/nonexistent")

        assert resp.status_code == 404
