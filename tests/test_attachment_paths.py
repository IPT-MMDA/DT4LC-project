"""Attachment paths in ``POST /v1/jobs`` and ``POST /v1/chat`` are checked at the door (server/paths.py).

Every attachment ``path`` must be a file the session may download: 403 outside the served roots, 404
missing or another owner's file, both before the quota counts the request and before ``/v1/chat``
classifies it; the message never holds the path.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from dta.config import ROOT_DIR
from server.app import app
import server.jobs as jobs_mod
from server.jobs import JobQueue, MemoryJobStore
import server.paths as paths_mod
from server.paths import ATTACHMENT_NOT_FOUND, ATTACHMENT_REFUSED
import server.routes.files as files_mod

# The probe: the bundled sample as it sits in resources/, outside every served root.
PROBE = Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif"
REFUSED = {"ok": False, "error": {"code": "unauthorized", "message": ATTACHMENT_REFUSED, "details": {}}}
NOT_FOUND = {"ok": False, "error": {"code": "not_found", "message": ATTACHMENT_NOT_FOUND, "details": {}}}


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[JobQueue]:
    """A queue without workers: submitted jobs stay pending."""
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    yield q
    q._thread_pool.shutdown(wait=False)


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(files_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(paths_mod, "UPLOAD_DIR", upload_dir)
    return upload_dir


@pytest.fixture
def guest(monkeypatch: pytest.MonkeyPatch, queue: JobQueue, uploads: Path) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")


def _client() -> TestClient:
    """A browser: its own cookie jar, so its own session (same client IP as every other TestClient)."""
    return TestClient(app, raise_server_exceptions=False)


def _geotiff_bytes() -> bytes:
    with (
        MemoryFile() as mem,
        mem.open(
            driver="GTiff",
            width=32,
            height=32,
            count=3,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(32.0, 47.0, 0.01, 0.01),
        ) as dst,
    ):
        dst.write(np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32))
        dst.close()
        return bytes(mem.read())


def _attachment(path: str | Path, filename: str = "scene.tif") -> dict[str, Any]:
    return {"id": "a1", "filename": filename, "mime_type": "image/tiff", "path": str(path)}


def _upload(client: TestClient) -> dict[str, Any]:
    r = client.post("/v1/upload", files={"file": ("scene.tif", _geotiff_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _used(client: TestClient) -> int:
    return int(client.get("/v1/session").json()["analyses"]["used"])


def _job(client: TestClient, *paths: str | Path) -> Any:
    return client.post("/v1/jobs", json={"prompt": "Calculate NDVI", "attachments": [_attachment(p) for p in paths]})


def _job_output(client: TestClient, outputs: Path) -> Path:
    """A finished-looking artifact of a job ``client`` owns: ``OUTPUTS_PATH/<job_id>/1_NDVIMap_ndvi.tif``."""
    r = _job(client, _upload(client)["path"])
    assert r.status_code == 202, r.text
    tif = outputs / r.json()["id"] / "1_NDVIMap_ndvi.tif"
    tif.parent.mkdir()
    tif.write_bytes(_geotiff_bytes())
    return tif


def _assert_refused(r: Any, expected: dict[str, Any], *paths: Path | str) -> None:
    """``r`` is ``expected`` (403 or 404) and its body holds none of ``paths``, nor their file names."""
    assert r.status_code == (403 if expected is REFUSED else 404), r.text
    assert r.json() == expected
    for p in paths:
        assert str(p) not in r.text and Path(p).name not in r.text


# --- Refused: POST /v1/jobs ------------------------------------------------------------------------


def test_guest_path_outside_the_roots_is_403_and_not_counted(guest: None, queue: JobQueue, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "scene.tif"
    outside.parent.mkdir()
    outside.write_bytes(_geotiff_bytes())
    client = _client()
    used, total, stored = _used(client), client.get("/v1/jobs").json()["total"], queue._store.count()

    r = _job(client, outside)

    _assert_refused(r, REFUSED, outside)
    assert _used(client) == used
    assert client.get("/v1/jobs").json()["total"] == total
    assert queue._store.count() == stored


def test_another_sessions_upload_is_404_and_not_counted(guest: None, queue: JobQueue) -> None:
    alice, bob = _client(), _client()
    up = _upload(alice)
    used, stored = _used(bob), queue._store.count()

    r = _job(bob, up["path"])

    _assert_refused(r, NOT_FOUND, up["path"])
    assert _used(bob) == used
    assert bob.get("/v1/jobs").json()["total"] == 0
    assert queue._store.count() == stored


def test_missing_file_inside_the_roots_is_404(guest: None, queue: JobQueue, uploads: Path) -> None:
    client = _client()
    missing = uploads / "abc12345_gone.tif"
    used = _used(client)

    r = _job(client, missing)

    _assert_refused(r, NOT_FOUND, missing)
    assert _used(client) == used
    assert queue._store.count() == 0


def test_a_directory_inside_the_roots_is_404(guest: None, queue: JobQueue, uploads: Path) -> None:
    folder = uploads / "abc12345_folder.tif"
    folder.mkdir()
    _assert_refused(_job(_client(), folder), NOT_FOUND, folder)
    assert queue._store.count() == 0


def test_traversal_out_of_the_roots_is_403(guest: None, queue: JobQueue, uploads: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "scene.tif"
    outside.parent.mkdir()
    outside.write_bytes(_geotiff_bytes())
    client = _client()
    traversal = f"{uploads}/../outside/scene.tif"

    _assert_refused(_job(client, traversal), REFUSED, traversal, outside)
    assert queue._store.count() == 0


def test_one_refused_file_refuses_the_request(guest: None, queue: JobQueue) -> None:
    client = _client()
    own = _upload(client)["path"]

    _assert_refused(_job(client, own, PROBE), REFUSED, PROBE)
    assert queue._store.count() == 0


def test_context_previous_attachments_are_checked_too(guest: None, queue: JobQueue) -> None:
    """With nothing attached, the worker reads ``context.previous_attachments``: the same rule applies."""
    client = _client()
    body = {"prompt": "Calculate NDVI", "context": {"previous_attachments": [_attachment(PROBE)]}}

    _assert_refused(client.post("/v1/jobs", json=body), REFUSED, PROBE)
    assert queue._store.count() == 0


# --- Accepted ----------------------------------------------------------------------------------------


def test_own_upload_is_accepted(guest: None, queue: JobQueue) -> None:
    client = _client()
    r = _job(client, _upload(client)["path"])
    assert r.status_code == 202, r.text
    assert _used(client) == 1


def test_the_sample_is_accepted(guest: None, queue: JobQueue) -> None:
    client = _client()
    sample = client.post("/v1/sample")
    assert sample.status_code == 200, sample.text
    paths = [a["path"] for a in sample.json()["attachments"]]

    r = _job(client, *paths)

    assert r.status_code == 202, r.text
    assert [a["path"] for a in queue._store.get(r.json()["id"]).attachments] == paths  # type: ignore[union-attr]


def test_own_job_output_is_accepted(guest: None, queue: JobQueue, _outputs_in_tmp: Path) -> None:
    client = _client()
    tif = _job_output(client, _outputs_in_tmp)

    r = _job(client, tif)

    assert r.status_code == 202, r.text


def test_another_owners_job_output_is_404_for_a_guest_and_accepted_for_an_operator(
    guest: None, queue: JobQueue, monkeypatch: pytest.MonkeyPatch, _outputs_in_tmp: Path
) -> None:
    tif = _job_output(_client(), _outputs_in_tmp)
    _assert_refused(_job(_client(), tif), NOT_FOUND, tif)

    monkeypatch.setenv("DEV_ROLE", "operator")
    operator = _client()
    r = _job(operator, tif)
    assert r.status_code == 202, r.text
    # Any file in the roots, not any file on the server
    _assert_refused(_job(operator, PROBE), REFUSED, PROBE)


def test_ownerless_roots_stay_usable(guest: None, queue: JobQueue, _outputs_in_tmp: Path) -> None:
    """``OUTPUTS_PATH/run-*`` and the temp dirs have no owner and stay usable, as for downloads."""
    run_tif = _outputs_in_tmp / "run-attachment-paths" / "x.tif"
    run_tif.parent.mkdir()
    run_tif.write_bytes(_geotiff_bytes())
    assert _job(_client(), run_tif).status_code == 202


def test_a_region_without_files_is_unaffected(guest: None, queue: JobQueue) -> None:
    region = {
        "bbox": [33.0, 46.6, 33.3, 46.8],
        "dataset": "HLS L30",
        "start_date": "2023-06-01",
        "end_date": "2023-06-09",
    }
    r = _client().post("/v1/jobs", json={"prompt": "Calculate NDVI", "region": region})
    assert r.status_code == 202, r.text


# --- POST /v1/chat: refused before classify_intent ---------------------------------------------------


@pytest.mark.parametrize("accept", [None, "text/event-stream"], ids=["json", "sse"])
@pytest.mark.parametrize("case", ["outside", "another-owner"])
def test_chat_refuses_before_classifying(guest: None, queue: JobQueue, case: str, accept: str | None) -> None:
    client = _client()
    target = PROBE if case == "outside" else _upload(_client())["path"]
    used = _used(client)
    body = {"message": "what is in this image?", "attachments": [_attachment(target, "hlsl_20230601.tif")]}
    headers = {"Accept": accept} if accept else {}

    with patch("server.routes.chat.classify_intent") as classify:
        r = client.post("/v1/chat", json=body, headers=headers)

    _assert_refused(r, REFUSED if case == "outside" else NOT_FOUND, target)
    classify.assert_not_called()
    assert _used(client) == used
    assert queue._store.count() == 0


def test_chat_resume_with_a_refused_file_is_refused(guest: None, queue: JobQueue) -> None:
    """``here`` with a pending analysis would submit at once: the check runs before that too."""
    client = _client()
    pending = {"prompt": "Calculate NDVI", "needs": ["RasterPath"], "asked_at": datetime.now(timezone.utc).isoformat()}
    body = {"message": "here", "attachments": [_attachment(PROBE)], "pending": pending}

    with patch("server.routes.chat.classify_intent") as classify:
        r = client.post("/v1/chat", json=body)

    _assert_refused(r, REFUSED, PROBE)
    classify.assert_not_called()
    assert queue._store.count() == 0


def test_chat_with_own_upload_reaches_the_classifier(guest: None, queue: JobQueue) -> None:
    client = _client()
    body = {"message": "What is NDVI?", "attachments": [_attachment(_upload(client)["path"])]}
    reply = {"intent": "conversation", "rule": "question", "response": "NDVI is a vegetation index."}

    with patch("server.routes.chat.classify_intent", MagicMock(return_value=reply)) as classify:
        r = client.post("/v1/chat", json=body)

    assert r.status_code == 200, r.text
    classify.assert_called_once()


# --- The probe: the raw sample under resources/kahovka_data/ ------------------------------------------


@pytest.mark.parametrize("role", ["guest", "operator"])
@pytest.mark.parametrize("endpoint", ["/v1/jobs", "/v1/chat"])
def test_the_raw_sample_path_is_refused(
    queue: JobQueue, uploads: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, role: str
) -> None:
    monkeypatch.setenv("DEV_ROLE", role)
    client = _client()
    used = _used(client)
    text = {"prompt": "Calculate NDVI"} if endpoint == "/v1/jobs" else {"message": "Calculate NDVI"}

    with patch("server.routes.chat.classify_intent") as classify:
        r = client.post(endpoint, json={**text, "attachments": [_attachment(PROBE, PROBE.name)]})

    _assert_refused(r, REFUSED, PROBE)
    classify.assert_not_called()
    assert _used(client) == used
    assert queue._store.count() == 0
