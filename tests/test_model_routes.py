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


# --- Who may download, cancel and delete (MODEL_DOWNLOAD_ON_DEMAND) -------------------------

PRITHVI = {"id": "prithvi-eo-v1-100m", "name": "Prithvi EO v1 (100M)", "size_mb": 450}


def _manager(status: ModelStatus) -> MagicMock:
    manager = MagicMock()
    manager.get_model_info.return_value = PRITHVI
    manager.get_model_status.return_value = status
    progress = MagicMock()
    progress.status = ModelStatus.DOWNLOADING
    manager.start_download.return_value = progress
    manager.cancel_download.return_value = True
    manager.delete_model.return_value = True
    return manager


def _gated(message: str) -> dict:
    return {"ok": False, "error": {"code": "MODEL_GATED", "message": message, "details": {"model_id": PRITHVI["id"]}}}


GUEST_REFUSAL = "Prithvi EO v1 (100M) isn't installed on this server and isn't available for guests — ask an operator."
DOWNLOADS_OFF_REFUSAL = "Prithvi EO v1 (100M) isn't installed on this server (450 MB) — ask an operator to install it."
DELETE_REFUSAL = "Only an operator can delete Prithvi EO v1 (100M) from this server."


def _call(client: TestClient, manager: MagicMock, action: str) -> tuple[int, dict]:
    with patch("server.model_routes.get_model_manager", return_value=manager):
        if action == "delete":
            resp = client.delete(f"/v1/ml-models/{PRITHVI['id']}")
        else:
            resp = client.post(f"/v1/ml-models/{PRITHVI['id']}/{action}")
    return resp.status_code, resp.json()


@pytest.mark.parametrize(
    ("action", "status", "message"),
    [
        ("download", ModelStatus.NOT_INSTALLED, GUEST_REFUSAL),
        ("cancel", ModelStatus.DOWNLOADING, GUEST_REFUSAL),
        ("delete", ModelStatus.AVAILABLE, DELETE_REFUSAL),
    ],
)
def test_guest_is_refused_and_nothing_changes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, action: str, status: ModelStatus, message: str
) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    manager = _manager(status)

    assert _call(client, manager, action) == (403, _gated(message))
    manager.start_download.assert_not_called()
    manager.cancel_download.assert_not_called()
    manager.delete_model.assert_not_called()


def test_member_can_download_and_cancel_but_not_delete(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "member")

    manager = _manager(ModelStatus.NOT_INSTALLED)
    assert _call(client, manager, "download")[0] == 202
    manager.start_download.assert_called_once_with(PRITHVI["id"])

    manager = _manager(ModelStatus.DOWNLOADING)
    assert _call(client, manager, "cancel")[0] == 200
    manager.cancel_download.assert_called_once_with(PRITHVI["id"])

    manager = _manager(ModelStatus.AVAILABLE)
    assert _call(client, manager, "delete") == (403, _gated(DELETE_REFUSAL))
    manager.delete_model.assert_not_called()


def test_operator_can_delete(client: TestClient) -> None:  # DEV_ROLE=operator (conftest)
    manager = _manager(ModelStatus.AVAILABLE)
    assert _call(client, manager, "delete")[0] == 200
    manager.delete_model.assert_called_once_with(PRITHVI["id"])


@pytest.mark.parametrize("role", ["member", "operator"])
def test_downloads_off_refuses_every_role(client: TestClient, monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    monkeypatch.setenv("DEV_ROLE", role)
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "none")
    manager = _manager(ModelStatus.NOT_INSTALLED)

    assert _call(client, manager, "download") == (403, _gated(DOWNLOADS_OFF_REFUSAL))
    manager.start_download.assert_not_called()


def test_operators_only_policy_names_the_member(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "member")
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "operators")
    manager = _manager(ModelStatus.NOT_INSTALLED)

    status, body = _call(client, manager, "download")
    assert status == 403
    assert body["error"]["message"] == GUEST_REFUSAL.replace("guests", "members")


def test_all_policy_lets_a_guest_download_and_cancel(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "all")

    manager = _manager(ModelStatus.NOT_INSTALLED)
    assert _call(client, manager, "download")[0] == 202
    manager.start_download.assert_called_once_with(PRITHVI["id"])

    manager = _manager(ModelStatus.DOWNLOADING)
    assert _call(client, manager, "cancel")[0] == 200

    manager = _manager(ModelStatus.AVAILABLE)
    assert _call(client, manager, "delete")[0] == 403  # deleting stays operator-only


def test_guest_can_still_read_models(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    manager = _manager(ModelStatus.NOT_INSTALLED)
    manager.list_models.return_value = [{**PRITHVI, "status": ModelStatus.NOT_INSTALLED.value}]
    manager.cache_dir = "/tmp/cache"

    with patch("server.model_routes.get_model_manager", return_value=manager):
        assert client.get("/v1/ml-models").status_code == 200
        assert client.get(f"/v1/ml-models/{PRITHVI['id']}").status_code == 200
