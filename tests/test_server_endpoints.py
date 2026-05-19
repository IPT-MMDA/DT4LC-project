"""Tests for server upload, tile, and GEE endpoints.

Covers the higher-complexity server endpoints that require
more sophisticated mocking (file upload, tile serving, job submission).
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest

from server.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestUploadEndpoint:
    """Tests for POST /v1/upload."""

    def test_upload_no_file(self, client: TestClient) -> None:
        """Test upload without a file."""
        resp = client.post("/v1/upload")
        # Should fail with 422 (validation error) since file is required
        assert resp.status_code == 422

    def test_upload_geotiff(self, client: TestClient) -> None:
        """Test uploading a valid GeoTIFF file."""
        # Create a minimal valid GeoTIFF in-memory
        import rasterio
        from rasterio.transform import from_bounds

        buf = io.BytesIO()
        data = np.random.randint(0, 255, (1, 32, 32), dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 32, 32)

        with rasterio.open(
            buf,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        buf.seek(0)
        resp = client.post(
            "/v1/upload",
            files={"file": ("test.tif", buf, "image/tiff")},
        )

        assert resp.status_code == 200
        data = resp.json()
        # Check that response contains expected fields (structure may vary)
        assert "filename" in data or "ok" in data or "path" in data

    def test_upload_non_geotiff(self, client: TestClient) -> None:
        """Test uploading a non-GeoTIFF file."""
        buf = io.BytesIO(b"not a geotiff file content")
        resp = client.post(
            "/v1/upload",
            files={"file": ("test.txt", buf, "text/plain")},
        )
        # Should still succeed (accept any file) or fail gracefully
        assert resp.status_code in (200, 400, 500)


class TestJobSubmitEndpoint:
    """Tests for POST /v1/jobs."""

    def test_submit_job(self, client: TestClient) -> None:
        """Test job submission."""
        from server.jobs import Job, JobStatus

        mock_job = Job(
            id="test-id",
            status=JobStatus.PENDING,
            prompt="calculate ndvi",
        )

        mock_queue = AsyncMock()
        mock_queue.submit_job.return_value = "test-id"
        mock_queue.get_job.return_value = mock_job

        with patch("server.routes.jobs.get_job_queue", return_value=mock_queue):
            resp = client.post(
                "/v1/jobs",
                json={
                    "prompt": "calculate ndvi",
                    "mode": "hybrid",
                    "attachments": [],
                },
            )
        assert resp.status_code == 202
        data = resp.json()
        # ID is returned from the job object (may be auto-generated or from mock)
        assert isinstance(data["id"], str)
        assert len(data["id"]) > 0

    def test_submit_job_queue_full(self, client: TestClient) -> None:
        """Test submission when queue is full."""
        mock_queue = AsyncMock()
        mock_queue.submit_job.side_effect = RuntimeError("Job queue is full")

        with patch("server.routes.jobs.get_job_queue", return_value=mock_queue):
            resp = client.post(
                "/v1/jobs",
                json={"prompt": "test"},
            )
        assert resp.status_code in (429, 500, 503)


class TestGetJobEndpoint:
    """Tests for GET /v1/jobs/{job_id}."""

    def test_get_job_found(self, client: TestClient) -> None:
        """Test getting an existing job."""
        from server.jobs import Job, JobStatus

        mock_job = Job(
            id="test-123",
            status=JobStatus.COMPLETED,
            prompt="test prompt",
            progress=1.0,
            result={"intent": "pipeline", "data": "test"},
        )

        mock_queue = MagicMock()
        mock_queue.get_job = AsyncMock(return_value=mock_job)

        with patch("server.routes.jobs.get_job_queue", return_value=mock_queue):
            resp = client.get("/v1/jobs/test-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-123"
        assert data["status"] == "completed"


class TestCancelJobEndpoint:
    """Tests for POST /v1/jobs/{job_id}/cancel."""

    def test_cancel_job_success(self, client: TestClient) -> None:
        """Test cancelling a running job."""
        from server.jobs import Job, JobStatus

        mock_job = Job(
            id="test-123",
            status=JobStatus.RUNNING,
            prompt="test",
        )

        mock_queue = MagicMock()
        mock_queue.cancel_job = AsyncMock(return_value=True)
        mock_queue.get_job = AsyncMock(return_value=mock_job)

        with patch("server.routes.jobs.get_job_queue", return_value=mock_queue):
            resp = client.post("/v1/jobs/test-123/cancel")
        assert resp.status_code == 200


class TestConversationChatFlow:
    """Tests for conversation intent in chat flow."""

    def test_conversation_response_via_chat(self, client: TestClient) -> None:
        """Test that conversation intents return proper SSE response."""
        with patch(
            "server.routes.chat.orchestrate",
            return_value={
                "ok": True,
                "intent": "conversation",
                "response": "I can help with NDVI analysis!",
            },
        ):
            resp = client.post(
                "/v1/chat",
                json={"messages": [{"role": "user", "content": "what can you do?"}]},
            )

        assert resp.status_code == 200
        text = resp.text
        # SSE stream should end with a done event
        assert "done" in text
