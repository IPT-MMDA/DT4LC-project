"""API contract tests — pin response shapes the frontend (cognitive_ui) reads.

The frontend in ``cognitive_ui/src/types/index.ts`` declares strict TypeScript
interfaces for the JSON it consumes. Any backend rename or removed key breaks
the UI silently. These tests pin the load-bearing keys at the API boundary so
backend refactors cannot accidentally drop or rename a field the frontend
depends on.

Scope: only the endpoints the frontend actually calls. Pipeline execution is
not exercised here — that's covered in ``test_demos.py``. The "what can you
do?" prompt below short-circuits inside ``intent_classifier`` via a regex
match (capability question → CONVERSATION intent), so no LLM key is needed.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient with the server lifespan started.

    Lifespan boots the JobQueue worker pool, which is required for the
    /v1/jobs endpoints to actually run a submitted job.
    """
    from server.app import app

    with TestClient(app) as c:
        yield c


class TestHealthContract:
    """``GET /v1/health`` — frontend's HealthResponse interface."""

    def test_health_keys(self, client: TestClient) -> None:
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        # cognitive_ui/src/types/index.ts: HealthResponse { ok, service, version }
        for key in ("ok", "service", "version"):
            assert key in body, f"frontend reads .{key} from /v1/health"
        assert isinstance(body["ok"], bool)


class TestUploadContract:
    """``POST /v1/upload`` — frontend's Attachment-shaped response."""

    def test_upload_returns_attachment_shape(self, client: TestClient, synthetic_raster_path: str) -> None:
        with open(synthetic_raster_path, "rb") as f:
            r = client.post(
                "/v1/upload",
                files={"file": ("synthetic.tif", f, "image/tiff")},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        # Keys the frontend Attachment interface reads (some optional in TS,
        # but required at the API boundary for the chat-with-attachment flow).
        for key in ("id", "filename", "path", "preview_png_base64"):
            assert key in body, f"frontend reads .{key} from /v1/upload"
        # The path must point at a real file the executor can later read.
        assert Path(body["path"]).exists()


class TestJobSubmitContract:
    """``POST /v1/jobs`` and ``GET /v1/jobs/{id}`` — Job interface keys.

    Uses a capability-question prompt that the intent classifier handles
    deterministically (no LLM call), so the test runs in CI without keys.
    """

    REQUIRED_JOB_KEYS = (
        "id",
        "status",
        "progress",
        "plan",
        "result",
        "error",
        "created_at",
    )

    def test_submit_returns_job_shape(self, client: TestClient) -> None:
        r = client.post(
            "/v1/jobs",
            json={"prompt": "what can you do?", "attachments": []},
        )
        assert r.status_code == 202, r.text
        job = r.json()
        for key in self.REQUIRED_JOB_KEYS:
            assert key in job, f"frontend Job interface reads .{key}"
        assert isinstance(job["id"], str)
        # status enum values the frontend handles: pending|running|completed|failed|cancelled
        assert job["status"] in {"pending", "running", "completed", "failed", "cancelled"}
        assert isinstance(job["progress"], int | float)

    def test_get_job_returns_same_shape(self, client: TestClient) -> None:
        # First submit a job to have something to fetch.
        submitted = client.post(
            "/v1/jobs",
            json={"prompt": "what can you do?", "attachments": []},
        ).json()
        job_id = submitted["id"]

        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        for key in self.REQUIRED_JOB_KEYS:
            assert key in job, f"frontend Job interface reads .{key} from GET"
        assert job["id"] == job_id


class TestJobListContract:
    """``GET /v1/jobs`` — frontend's JobsListResponse pagination keys."""

    def test_list_keys(self, client: TestClient) -> None:
        r = client.get("/v1/jobs?limit=5&offset=0")
        assert r.status_code == 200
        body = r.json()
        # cognitive_ui/src/types/index.ts: JobsListResponse { jobs, total, limit, offset }
        for key in ("jobs", "total", "limit", "offset"):
            assert key in body, f"frontend reads .{key} from /v1/jobs (list)"
        assert isinstance(body["jobs"], list)
        assert isinstance(body["total"], int)
