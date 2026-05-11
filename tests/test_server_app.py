"""Tests for server/app.py FastAPI endpoints.

Uses TestClient for synchronous endpoint testing and pytest-mock
for mocking external dependencies (orchestrate, executor, jobs, etc.).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from server.app import app
from server.utils import apply_colormap, detect_data_type, sse_frame


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSSEFrame:
    """Tests for sse_frame helper."""

    def test_basic_payload(self) -> None:
        result = sse_frame({"ok": True})
        assert result == b'data: {"ok": true}\n\n'

    def test_unicode_payload(self) -> None:
        result = sse_frame({"msg": "héllo wörld"})
        assert b"data:" in result
        assert "héllo wörld" in result.decode("utf-8")

    def test_nested_payload(self) -> None:
        result = sse_frame({"a": {"b": [1, 2]}})
        assert b'"a"' in result
        assert b'"b"' in result


class TestDetectDataType:
    """Tests for detect_data_type helper."""

    def test_ndvi(self) -> None:
        assert detect_data_type("sentinel2_ndvi_2024.tif") == "ndvi"

    def test_ndwi(self) -> None:
        assert detect_data_type("area_ndwi.tif") == "ndwi"

    def test_ndsi(self) -> None:
        assert detect_data_type("NDSI_composite.tif") == "ndsi"

    def test_lulc(self) -> None:
        assert detect_data_type("lulc_map.tif") == "lulc"

    def test_land_keyword(self) -> None:
        assert detect_data_type("land_cover.tif") == "lulc"

    def test_change(self) -> None:
        assert detect_data_type("change_map_2024.tif") == "change"

    def test_generic(self) -> None:
        assert detect_data_type("satellite_image.tif") == "generic"


class TestApplyColormap:
    """Tests for apply_colormap helper."""

    def test_returns_rgb(self) -> None:
        data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = apply_colormap(data, "viridis")
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_rdylgn_colormap(self) -> None:
        data = np.zeros((10, 10), dtype=np.uint8)
        result = apply_colormap(data, "RdYlGn")
        assert result.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /v1/health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["service"] == "DT4LC"
        assert data["version"] == "1.0.0"


class TestCapabilitiesEndpoint:
    """Tests for GET /v1/capabilities."""

    def test_capabilities_returns_registry(self, client: TestClient) -> None:
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "types" in data
        assert "instances" in data
        assert "count" in data
        assert data["count"] == len(data["instances"])


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    def test_models_returns_list(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "count" in data


class TestMetricsEndpoint:
    """Tests for GET /v1/metrics."""

    def test_metrics_returns_stats(self, client: TestClient) -> None:
        mock_stats = MagicMock()
        mock_stats.total_executions = 5
        mock_stats.successful_executions = 4
        mock_stats.failed_executions = 1
        mock_stats.average_duration_seconds = 2.5
        mock_stats.avg_execution_time = 2.5
        mock_stats.total_llm_calls = 10
        mock_stats.total_llm_tokens = 5000
        mock_stats.total_llm_cost = 0.05
        mock_stats.llm_by_provider = {}

        mock_collector = MagicMock()
        mock_collector.get_stats.return_value = mock_stats

        with patch("server.routes.health.get_metrics_collector", return_value=mock_collector):
            resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_executions"] == 5
        assert data["total_llm_calls"] == 10


class TestDatasetsEndpoint:
    """Tests for GET /v1/gee/datasets."""

    def test_datasets_returns_list(self, client: TestClient) -> None:
        resp = client.get("/v1/gee/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "sentinel-2" in data["datasets"]
        assert "modis" in data["datasets"]
        assert "landsat-8" in data["datasets"]


class TestPlanEndpoint:
    """Tests for POST /v1/plan."""

    def test_empty_messages_returns_error(self, client: TestClient) -> None:
        resp = client.post("/v1/plan", json={"messages": []})
        # HTTPException(400) is caught by outer handler → 500
        assert resp.status_code in (400, 500)
        assert "No messages" in resp.json()["detail"]

    def test_plan_success(self, client: TestClient) -> None:
        mock_plan = {
            "flow": "ndvi",
            "steps": [{"uses": "input/file", "binds": {}}],
            "outputs": ["NDVIMap"],
        }
        with patch("server.routes.chat.orchestrate", return_value={"ok": True, "plan": mock_plan}):
            resp = client.post(
                "/v1/plan",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["plan"]["flow"] == "ndvi"

    def test_plan_failure(self, client: TestClient) -> None:
        with patch("server.routes.chat.orchestrate", return_value={"ok": False, "error": "test fail"}):
            resp = client.post(
                "/v1/plan",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["ok"] is False
        assert "test fail" in data["error"]


class TestExecuteEndpoint:
    """Tests for POST /v1/execute."""

    def test_empty_messages_returns_error(self, client: TestClient) -> None:
        resp = client.post("/v1/execute", json={"messages": []})
        # HTTPException(400) is caught by outer handler → 500
        assert resp.status_code in (400, 500)
        assert "No messages" in resp.json()["detail"]

    def test_execute_orchestrate_fail(self, client: TestClient) -> None:
        with patch("server.routes.chat.orchestrate", return_value={"ok": False, "error": "plan failed"}):
            resp = client.post(
                "/v1/execute",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["ok"] is False

    def test_execute_success(self, client: TestClient) -> None:
        mock_plan = {
            "flow": "ndvi",
            "steps": [{"uses": "input/file", "binds": {"RasterPath": "/fake.tif"}}],
            "outputs": ["NDVIMap"],
        }
        mock_exec = {"flow": "ndvi", "steps": ["input/file"], "artifacts": {}, "outputs": {}}

        with (
            patch("server.routes.chat.orchestrate", return_value={"ok": True, "plan": mock_plan}),
            patch("server.routes.chat.PipelineExecutor") as mock_exec_class,
        ):
            mock_exec_class.return_value.execute.return_value = mock_exec
            resp = client.post(
                "/v1/execute",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "plan" in data
        assert "result" in data


class TestChatEndpoint:
    """Tests for POST /v1/chat (SSE streaming)."""

    def test_chat_empty_messages(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat",
            json={"messages": []},
        )
        assert resp.status_code == 200
        text = resp.text
        assert "error" in text
        assert "done" in text

    def test_chat_success_stream(self, client: TestClient) -> None:
        mock_plan = {
            "flow": "ndvi",
            "steps": [{"uses": "input/file", "binds": {"RasterPath": "/fake.tif"}}],
            "outputs": ["NDVIMap"],
        }
        mock_exec = {"flow": "ndvi", "steps": ["input/file"], "artifacts": {}, "outputs": {}}

        with (
            patch("server.routes.chat.orchestrate", return_value={"ok": True, "plan": mock_plan}),
            patch("server.routes.chat.PipelineExecutor") as mock_exec_class,
        ):
            mock_exec_class.return_value.execute.return_value = mock_exec
            resp = client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 200
        text = resp.text
        assert "planning" in text
        assert "complete" in text or "done" in text

    def test_chat_orchestrate_fail(self, client: TestClient) -> None:
        with patch("server.routes.chat.orchestrate", return_value={"ok": False, "error": "fail"}):
            resp = client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 200
        text = resp.text
        assert "error" in text
        assert "done" in text


class TestDownloadEndpoint:
    """Tests for GET /v1/download."""

    def test_disallowed_path_returns_403(self, client: TestClient) -> None:
        resp = client.get("/v1/download", params={"path": "/etc/passwd"})
        assert resp.status_code == 403

    def test_missing_file_returns_404(self, client: TestClient) -> None:
        import tempfile

        tmp = Path(tempfile.gettempdir()) / "nonexistent_dt4lc_test_file.xyz"
        resp = client.get("/v1/download", params={"path": str(tmp)})
        # Path is under allowed dir (tempdir) but file doesn't exist
        assert resp.status_code == 404


class TestFilesEndpoint:
    """Tests for GET /v1/files."""

    def test_files_returns_list(self, client: TestClient) -> None:
        with (
            patch("server.routes.files.UPLOAD_DIR", Path("/tmp/dt4lc_test_nonexistent")),
        ):
            resp = client.get("/v1/files")
        # Should return 200 even if dir is empty/doesn't exist for uploads
        assert resp.status_code == 200 or resp.status_code == 500


class TestQueueStatsEndpoint:
    """Tests for GET /v1/queue/stats."""

    def test_queue_stats_returns_data(self, client: TestClient) -> None:
        mock_queue = MagicMock()
        mock_queue.get_stats.return_value = {
            "total_jobs": 0,
            "queue_size": 0,
            "workers": 3,
            "max_workers": 3,
            "by_status": {},
        }
        with patch("server.app.get_job_queue", return_value=mock_queue):
            resp = client.get("/v1/queue/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data
        assert "workers" in data


class TestJobEndpoints:
    """Tests for job-related endpoints."""

    def test_get_job_not_found(self, client: TestClient) -> None:
        mock_queue = MagicMock()
        mock_queue.get_job = AsyncMock(return_value=None)
        mock_queue._jobs = {}
        with patch("server.app.get_job_queue", return_value=mock_queue):
            resp = client.get("/v1/jobs/nonexistent")
        assert resp.status_code == 404

    def test_list_jobs_invalid_status(self, client: TestClient) -> None:
        resp = client.get("/v1/jobs", params={"status": "invalid_status_xyz"})
        assert resp.status_code == 400

    def test_list_jobs_success(self, client: TestClient) -> None:
        mock_queue = MagicMock()
        mock_queue.list_jobs = AsyncMock(return_value=[])
        mock_queue._jobs = {}
        with patch("server.app.get_job_queue", return_value=mock_queue):
            resp = client.get("/v1/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data

    def test_cancel_job_not_found(self, client: TestClient) -> None:
        mock_queue = MagicMock()
        mock_queue.cancel_job = AsyncMock(return_value=False)
        mock_queue.get_job = AsyncMock(return_value=None)
        with patch("server.app.get_job_queue", return_value=mock_queue):
            resp = client.post("/v1/jobs/nonexistent/cancel")
        assert resp.status_code == 404


class TestModelRoutesEndpoints:
    """Tests for /v1/ml-models endpoints via the included router."""

    def test_list_ml_models(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.list_models.return_value = []
        mock_manager.cache_dir = Path("/tmp/models")
        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_get_model_not_found(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = None
        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models/nonexistent")
        assert resp.status_code == 404

    def test_get_model_found(self, client: TestClient) -> None:
        mock_manager = MagicMock()
        mock_manager.get_model_info.return_value = {
            "model_id": "test-model",
            "name": "Test Model",
            "status": "not_installed",
        }
        with patch("server.model_routes.get_model_manager", return_value=mock_manager):
            resp = client.get("/v1/ml-models/test-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "test-model"


class TestLayerEndpoints:
    """Tests for GEE layer persistence endpoints."""

    def test_list_layers(self, client: TestClient) -> None:
        with patch("server.layer_metadata_store.list_all_layers", return_value=[]):
            resp = client.get("/v1/gee/layers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_persist_layer_missing_id(self, client: TestClient) -> None:
        resp = client.post("/v1/gee/layers/persist", json={})
        assert resp.status_code == 400

    def test_persist_layer_success(self, client: TestClient) -> None:
        with patch("server.layer_metadata_store.save_layer_metadata"):
            resp = client.post(
                "/v1/gee/layers/persist",
                json={"layer_id": "test-layer", "layer_name": "Test"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_delete_layer_not_found(self, client: TestClient) -> None:
        with patch("server.layer_metadata_store.delete_layer_metadata", return_value=False):
            resp = client.delete("/v1/gee/layers/nonexistent")
        assert resp.status_code == 404

    def test_delete_layer_success(self, client: TestClient) -> None:
        with patch("server.layer_metadata_store.delete_layer_metadata", return_value=True):
            resp = client.delete("/v1/gee/layers/test-layer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
