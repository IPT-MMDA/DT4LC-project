"""Files for the Library (v0.6.0): ``GET /v1/files`` kinds, groups and previews; ``DELETE /v1/files/{id}``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from server import sessions
from server.app import app
import server.paths as paths_mod
import server.routes.files as files_mod
import server.sample as sample_mod
from server.uploads import owner_file

SAMPLE_LINE = "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09 · EPSG:32636"


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(files_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(paths_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setenv("DEV_ROLE", "guest")
    return upload_dir


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _geotiff(tmp_path: Path) -> Path:
    path = tmp_path / "field_20230715.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint16",
        crs="EPSG:32636",
        transform=from_origin(650_000.0, 5_160_000.0, 10, 10),
    ) as dst:
        dst.write((np.arange(3 * 256).reshape(3, 16, 16) % 1000 + 1).astype("uint16"))
    return path


def _upload(client: TestClient, path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        r = client.post("/v1/upload", files={"file": (path.name, f, "image/tiff")})
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _uploads(client: TestClient) -> list[dict[str, Any]]:
    return [f for f in client.get("/v1/files").json()["files"] if f["source"] == "upload"]


def _bundled() -> list[tuple[Path, int, bytes]]:
    return [
        (p, p.stat().st_size, p.read_bytes()[:4096])
        for p in (sample_mod.SAMPLE_DIR / n for n in sample_mod.SAMPLE_FILES)
    ]


# --- GET /v1/files ---------------------------------------------------------------------------


def test_an_upload_is_listed_with_kind_preview_and_created_at(uploads: Path, tmp_path: Path) -> None:
    client = _client()
    uploaded = _upload(client, _geotiff(tmp_path))

    (listed,) = _uploads(client)

    assert listed["kind"] == "upload"
    assert listed["upload_id"] == uploaded["id"]
    assert listed["group"] is None
    assert listed["preview_url"] == uploaded["preview_url"]
    assert client.get(listed["preview_url"]).status_code == 200
    assert listed["size_bytes"] == Path(uploaded["path"]).stat().st_size
    assert abs((datetime.fromisoformat(listed["created_at"]) - datetime.now()).total_seconds()) < 60


def test_the_sample_pair_is_one_group(uploads: Path) -> None:
    client = _client()
    attached = client.post("/v1/sample").json()["attachments"]

    listed = sorted(_uploads(client), key=lambda f: f["filename"])

    assert [f["kind"] for f in listed] == ["sample", "sample"]
    assert [f["path"] for f in listed] == [a["path"] for a in attached]
    assert [f["preview_url"] for f in listed] == [a["preview_url"] for a in attached]
    assert {f["upload_id"] for f in listed} == {attached[0]["id"]}
    assert [f["group"] for f in listed] == [{"name": "kakhovka_hls_l30.tif", "line": SAMPLE_LINE}] * 2
    for f in listed:
        datetime.fromisoformat(f["created_at"])


def test_another_sessions_sample_id_is_an_upload_for_nobody_else(uploads: Path) -> None:
    """The sample id is derived from the owner: a copy renamed into another session is not a sample."""
    alice = _client()
    attached = alice.post("/v1/sample").json()["attachments"]
    bob = _client()
    bob.get("/v1/session")
    path = Path(attached[0]["path"])
    bob_id = sessions.decode_cookie(bob.cookies[sessions.COOKIE_NAME]).id  # type: ignore[union-attr]

    assert sample_mod.is_sample_copy(path, sessions.decode_cookie(alice.cookies[sessions.COOKIE_NAME]).id)  # type: ignore[union-attr]
    assert not sample_mod.is_sample_copy(path, bob_id)


# --- DELETE /v1/files/{id} -------------------------------------------------------------------


def test_delete_an_upload_removes_it_its_preview_and_sidecar(uploads: Path, tmp_path: Path) -> None:
    client = _client()
    uploaded = _upload(client, _geotiff(tmp_path))
    path = Path(uploaded["path"])

    r = client.delete(f"/v1/files/{uploaded['id']}")

    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "deleted": [path.name]}
    assert not path.exists() and not path.with_suffix(".png").exists() and not owner_file(path).exists()
    assert list(uploads.iterdir()) == []
    assert _uploads(client) == []
    assert client.delete(f"/v1/files/{uploaded['id']}").status_code == 404


def test_delete_the_sample_id_removes_the_pair_and_keeps_the_bundle(uploads: Path) -> None:
    bundled = _bundled()
    client = _client()
    attached = client.post("/v1/sample").json()["attachments"]

    r = client.delete(f"/v1/files/{attached[0]['id']}")

    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == [Path(a["path"]).name for a in attached]
    assert list(uploads.iterdir()) == []
    assert _bundled() == bundled
    # Attaching the sample again makes a fresh copy.
    again = client.post("/v1/sample").json()["attachments"]
    assert [Path(a["path"]).is_file() for a in again] == [True, True]


def test_delete_one_sample_date_by_its_file_id(uploads: Path) -> None:
    client = _client()
    client.post("/v1/sample")
    first, second = sorted(_uploads(client), key=lambda f: f["filename"])

    assert client.delete(f"/v1/files/{second['id']}").json()["deleted"] == [second["filename"]]

    (left,) = _uploads(client)
    assert left["path"] == first["path"]
    assert left["group"] == {
        "name": "kakhovka_hls_l30.tif",
        "line": "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 · EPSG:32636",
    }


def test_delete_another_sessions_file_is_404(uploads: Path, tmp_path: Path) -> None:
    alice, bob = _client(), _client()
    uploaded = _upload(alice, _geotiff(tmp_path))
    sample_id = alice.post("/v1/sample").json()["attachments"][0]["id"]
    before = sorted(p.name for p in uploads.iterdir())

    assert bob.delete(f"/v1/files/{uploaded['id']}").status_code == 404
    assert bob.delete(f"/v1/files/{sample_id}").status_code == 404
    assert bob.delete("/v1/files/..").status_code == 404

    assert sorted(p.name for p in uploads.iterdir()) == before
