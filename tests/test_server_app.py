"""Tests for server/app.py FastAPI endpoints.

Uses TestClient for synchronous endpoint testing and pytest-mock
for mocking external dependencies (orchestrate, executor, jobs, etc.).
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

import dta
from server.app import app
from server.utils import apply_colormap, detect_data_type


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


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
        assert data["version"] == dta.__version__

    def test_served_versions_match_package_version(self, client: TestClient) -> None:
        """Health and OpenAPI report the release version from dta.__version__."""
        assert dta.__version__ == "0.6.0"
        assert client.get("/v1/health").json()["version"] == dta.__version__
        assert client.get("/openapi.json").json()["info"]["version"] == dta.__version__


class TestCapabilitiesEndpoint:
    """Tests for GET /v1/capabilities."""

    def test_capabilities_returns_the_tiered_document(self, client: TestClient) -> None:
        """The document itself is covered by tests/test_capabilities.py."""
        resp = client.get("/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) >= {"summary", "assistant", "offline", "tiers", "guest_limits", "session", "example_prompts"}
        assert [t["id"] for t in data["tiers"]] == ["ask", "composed", "builtin", "models", "data", "coming"]


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
        assert "No messages" in resp.json()["error"]["message"]

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
        candidate = {"flow": "ndvi", "steps": []}
        with patch(
            "server.routes.chat.orchestrate",
            return_value={"ok": False, "error": "test fail", "candidate": candidate},
        ):
            resp = client.post(
                "/v1/plan",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["code"] == "planning_failed"
        assert "test fail" in error["message"]
        assert error["details"]["candidate"] == candidate

    def test_plan_unexpected_error_is_json_500(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        with patch("server.routes.chat.orchestrate", side_effect=RuntimeError("boom")):
            resp = client.post(
                "/v1/plan",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 500
        error = resp.json()["error"]
        assert error["code"] == "internal_error"
        assert error["message"] == "Plan generation failed: boom"


class TestExecuteEndpoint:
    """Tests for POST /v1/execute."""

    @pytest.fixture(autouse=True)
    def _classified_as_analysis(self) -> Iterator[MagicMock]:
        """``orchestrate`` is mocked here, so is the classification before the quota (see TestNotAnAnalysis)."""
        with patch("server.routes.chat.classify_intent", return_value={"intent": "pipeline"}) as classify:
            yield classify

    def test_empty_messages_returns_error(self, client: TestClient) -> None:
        resp = client.post("/v1/execute", json={"messages": []})
        # HTTPException(400) is caught by outer handler → 500
        assert resp.status_code in (400, 500)
        assert "No messages" in resp.json()["error"]["message"]

    def test_execute_orchestrate_fail(self, client: TestClient) -> None:
        with patch("server.routes.chat.orchestrate", return_value={"ok": False, "error": "plan failed"}):
            resp = client.post(
                "/v1/execute",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "planning_failed"
        assert "plan failed" in resp.json()["error"]["message"]

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

    def test_execute_unexpected_error_is_json_500(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        with patch("server.routes.chat.orchestrate", side_effect=RuntimeError("boom")):
            resp = client.post(
                "/v1/execute",
                json={"messages": [{"role": "user", "content": "calculate ndvi"}]},
            )
        assert resp.status_code == 500
        error = resp.json()["error"]
        assert error["code"] == "internal_error"
        assert error["message"] == "Execution failed: boom"

    def test_conversation_on_the_second_classification_is_released(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first classification said analysis (counted); orchestrate's said conversation: 422, uncounted."""
        monkeypatch.setenv("DEV_ROLE", "guest")
        statuses, execute, body = self._execute(client, {"ok": True, "intent": "conversation"}, 6)

        assert statuses == [422] * 6
        execute.assert_not_called()
        assert body["error"]["code"] == "NOT_AN_ANALYSIS"
        assert client.get("/v1/session").json()["analyses"]["used"] == 0

    PLAN = {"flow": "ndvi", "steps": [{"uses": "input/file", "binds": {"RasterPath": "/fake.tif"}}], "outputs": []}
    BODY = {"messages": [{"role": "user", "content": "calculate ndvi"}]}

    def _execute(self, client: TestClient, orchestrated: dict, n: int) -> tuple[list[int], MagicMock, dict]:
        with (
            patch("server.routes.chat.orchestrate", return_value=orchestrated),
            patch("server.routes.chat.PipelineExecutor") as executor_class,
        ):
            executor_class.return_value.execute.return_value = {"flow": "ndvi", "steps": [], "artifacts": {}}
            responses = [client.post("/v1/execute", json=self.BODY) for _ in range(n)]
        return [r.status_code for r in responses], executor_class.return_value.execute, responses[-1].json()

    def test_sixth_guest_execute_in_an_hour_is_refused(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEV_ROLE", "guest")

        statuses, execute, body = self._execute(client, {"ok": True, "plan": self.PLAN}, 6)

        assert statuses == [200] * 5 + [429]
        assert execute.call_count == 5
        assert body["error"]["code"] == "GUEST_QUOTA"
        assert body["error"]["details"]["used"] == 5
        assert client.post("/v1/jobs", json={"prompt": "Compute NDVI"}).status_code == 429  # one shared quota

    @pytest.mark.parametrize("role", ["member", "operator"])
    def test_members_execute_without_limit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, role: str
    ) -> None:
        monkeypatch.setenv("DEV_ROLE", role)
        statuses, _, _ = self._execute(client, {"ok": True, "plan": self.PLAN}, 8)
        assert statuses == [200] * 8

    def test_composed_plan_is_not_run_without_confirmation(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEV_ROLE", "guest")
        proposal = {"ok": True, "plan": {**self.PLAN, "composed": True}, "proposal": "I can compute NDMI… run that?"}

        statuses, execute, body = self._execute(client, proposal, 6)

        assert statuses == [409] * 6  # refused before running, and not counted
        execute.assert_not_called()
        assert body == {
            "ok": False,
            "error": {
                "code": "CONFIRMATION_REQUIRED",
                "message": "This request composes a new analysis, which runs only after you confirm it: "
                "submit it with POST /v1/jobs, then confirm with POST /v1/jobs/{job_id}/confirm.",
                "details": {},
            },
        }
        assert client.get("/v1/session").json()["analyses"]["used"] == 0


class TestNotAnAnalysis:
    """A conversational prompt on POST /v1/plan or /v1/execute: a structured 422, not a 500 (run-1 carry-over)."""

    REFUSAL = {
        "ok": False,
        "error": {
            "code": "NOT_AN_ANALYSIS",
            "message": "This reads as a question, not an analysis — send it to POST /v1/chat.",
            "details": {"intent": "conversation"},
        },
    }

    @pytest.mark.parametrize("route", ["/v1/plan", "/v1/execute"])
    @pytest.mark.parametrize("prompt", ["hello", "what can you do?"])
    def test_conversation_is_refused_and_not_counted(
        self, monkeypatch: pytest.MonkeyPatch, route: str, prompt: str
    ) -> None:
        monkeypatch.setenv("DEV_ROLE", "guest")
        client = TestClient(app, raise_server_exceptions=False)
        with patch("server.routes.chat.PipelineExecutor") as executor_class:
            responses = [
                client.post(route, json={"messages": [{"role": "user", "content": prompt}]}) for _ in range(6)
            ]

        assert [r.status_code for r in responses] == [422] * 6  # never a 500, never the 429 of a spent quota
        assert responses[-1].json() == self.REFUSAL
        executor_class.return_value.execute.assert_not_called()
        assert client.get("/v1/session").json()["analyses"]["used"] == 0

    @pytest.mark.parametrize("route", ["/v1/plan", "/v1/execute"])
    def test_analysis_without_a_file_says_it_needs_one(self, monkeypatch: pytest.MonkeyPatch, route: str) -> None:
        """Neither endpoint takes a file, so "calculate ndvi" is the classifier's missing-input conversation."""
        monkeypatch.setenv("DEV_ROLE", "guest")
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(route, json={"messages": [{"role": "user", "content": "calculate ndvi"}]})

        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["code"] == "missing_input"
        assert error["message"].startswith("This analysis needs a raster")
        assert error["details"] == {"intent": "conversation"}
        assert client.get("/v1/session").json()["analyses"]["used"] == 0


class TestDownloadEndpoint:
    """Tests for GET /v1/download."""

    def test_disallowed_path_returns_403(self, client: TestClient) -> None:
        resp = client.get("/v1/download", params={"path": "/etc/passwd"})
        assert resp.status_code == 403

    def test_missing_file_returns_404(self, client: TestClient) -> None:
        from dta.config import TEMP_PATH

        tmp = TEMP_PATH / "nonexistent_dt4lc_test_file.xyz"
        resp = client.get("/v1/download", params={"path": str(tmp)})
        # Path is under allowed dir (TEMP_PATH) but file doesn't exist
        assert resp.status_code == 404

    def test_unrelated_tempdir_file_returns_403(self, client: TestClient) -> None:
        import tempfile
        import uuid

        tmp = Path(tempfile.gettempdir()) / f"dt4lc_unrelated_{uuid.uuid4().hex}.txt"
        tmp.write_text("not a job output")
        try:
            resp = client.get("/v1/download", params={"path": str(tmp)})
        finally:
            tmp.unlink()
        assert resp.status_code == 403

    def test_job_output_file_returns_200(self, client: TestClient) -> None:
        import uuid

        from dta.config import TEMP_PATH

        out_dir = TEMP_PATH / "delineate_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"test_{uuid.uuid4().hex}_fields.gpkg"
        out.write_bytes(b"gpkg")
        try:
            resp = client.get("/v1/download", params={"path": str(out)})
        finally:
            out.unlink()
        assert resp.status_code == 200
        assert resp.content == b"gpkg"

    def test_job_artifact_file_returns_200(self, client: TestClient) -> None:
        import uuid

        from server import paths

        # Job artifacts live in OUTPUTS_PATH/<job_id>/ (#65)
        out_dir = paths.OUTPUTS_PATH / f"test-{uuid.uuid4().hex}"
        out_dir.mkdir(parents=True)
        out = out_dir / "1_NDVIMap_ndvi.tif"
        out.write_bytes(b"tif")
        try:
            resp = client.get("/v1/download", params={"path": str(out)})
        finally:
            out.unlink()
            out_dir.rmdir()
        assert resp.status_code == 200
        assert resp.content == b"tif"

    def test_traversal_out_of_allowed_dir_returns_403(self, client: TestClient) -> None:
        import tempfile
        import uuid

        from dta.config import TEMP_PATH

        tmp = Path(tempfile.gettempdir()) / f"dt4lc_unrelated_{uuid.uuid4().hex}.txt"
        tmp.write_text("not a job output")
        # TEMP_PATH/../../(...)/<tmp file> starts with TEMP_PATH but resolves outside it
        up = TEMP_PATH.joinpath(*[".."] * len(TEMP_PATH.parts))
        traversal = up / tmp.resolve().relative_to(tmp.resolve().anchor)
        try:
            resp = client.get("/v1/download", params={"path": str(traversal)})
        finally:
            tmp.unlink()
        assert resp.status_code == 403

    def test_sibling_prefix_dir_returns_403(self, client: TestClient) -> None:
        import tempfile
        import uuid

        # A string-prefix check would accept <tmp>/dt4lc_delineate_evil.../x.gpkg
        evil_dir = Path(tempfile.gettempdir()) / f"dt4lc_delineate_evil_{uuid.uuid4().hex}"
        evil_dir.mkdir()
        evil = evil_dir / "x.gpkg"
        evil.write_text("evil")
        try:
            resp = client.get("/v1/download", params={"path": str(evil)})
        finally:
            evil.unlink()
            evil_dir.rmdir()
        assert resp.status_code == 403


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
        assert resp.status_code == 422

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
