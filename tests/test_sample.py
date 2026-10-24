"""``POST /v1/sample``: the bundled Kakhovka pair as the caller's uploads (v0.6.0 Workspace)."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import mercantile
from PIL import Image
import pytest

from server import sessions
from server.app import app
import server.paths as paths_mod
import server.routes.files as files_mod
import server.sample as sample_mod
from server.uploads import cleanup_guest_uploads, owner_file

TILE = mercantile.tile(32.5, 46.5, 10)
TILE_URL = f"/v1/tiles/{TILE.z}/{TILE.x}/{TILE.y}"
LINE = "HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09"


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


def _session_id(client: TestClient) -> str:
    return sessions.decode_cookie(client.cookies[sessions.COOKIE_NAME]).id  # type: ignore[union-attr]


def test_sample_attaches_two_owned_uploads_with_metadata_and_previews(uploads: Path) -> None:
    client = _client()

    r = client.post("/v1/sample")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "kakhovka_hls_l30"
    assert body["line"] == LINE
    owner = _session_id(client)
    first, second = body["attachments"]
    assert [a["filename"] for a in (first, second)] == ["hlsl_20230601.tif", "hlsl_20230609.tif"]
    assert first["id"] == second["id"]
    for att, day, size in ((first, "2023-06-01", "4.0 MB"), (second, "2023-06-09", "3.7 MB")):
        path = Path(att["path"])
        assert path.parent == uploads and path.name == f"{att['id']}_{att['filename']}"
        assert att["owner"] == owner
        assert json.loads(owner_file(path).read_text())["owner"] == owner
        assert att["size"] == [768, 768] and att["crs"] == "EPSG:32636"
        assert att["metadata"]["line"] == f"HLS L30 · B2–B6 · 30 m · {day}"
        assert att["metadata"]["size_label"] == size
        assert att["preview_png_base64"] == ""
        assert att["preview_url"] == "/v1/download?path=" + str(path.with_suffix(".png")).replace("/", "%2F")
        with Image.open(path.with_suffix(".png")) as img:
            assert (img.mode, img.size) == ("RGB", (768, 768))
    listed = {f["path"] for f in client.get("/v1/files").json()["files"] if f["source"] == "upload"}
    assert listed == {first["path"], second["path"]}


def test_sample_is_idempotent_per_session(uploads: Path) -> None:
    client = _client()
    first = client.post("/v1/sample").json()
    files = sorted(p.name for p in uploads.iterdir())
    mtimes = {p.name: p.stat().st_mtime_ns for p in uploads.iterdir()}

    again = client.post("/v1/sample").json()

    assert again == first
    assert sorted(p.name for p in uploads.iterdir()) == files
    assert {p.name: p.stat().st_mtime_ns for p in uploads.iterdir()} == mtimes
    assert len(files) == 6  # two GeoTIFFs, two previews, two sidecars
    # Another session gets its own pair.
    other = _client().post("/v1/sample").json()
    assert other["attachments"][0]["id"] != first["attachments"][0]["id"]


def test_only_the_owner_downloads_and_tiles_the_sample(uploads: Path) -> None:
    alice, bob = _client(), _client()
    attachments = alice.post("/v1/sample").json()["attachments"]
    bob.get("/v1/session")

    for att in attachments:
        tif = {"path": att["path"]}
        assert alice.get("/v1/download", params=tif).status_code == 200
        assert alice.get(att["preview_url"]).status_code == 200
        tile = alice.get(TILE_URL, params=tif)
        assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"

        assert bob.get("/v1/download", params=tif).status_code == 404
        assert bob.get(att["preview_url"]).status_code == 404
        assert bob.get(TILE_URL, params=tif).status_code == 404
    assert all(f["source"] != "upload" for f in bob.get("/v1/files").json()["files"])


def test_guest_cleanup_removes_the_sample_and_keeps_the_bundle(uploads: Path) -> None:
    bundled = [sample_mod.SAMPLE_DIR / name for name in sample_mod.SAMPLE_FILES]
    sizes = [p.stat().st_size for p in bundled]
    _client().post("/v1/sample")
    for sidecar in uploads.glob("*.owner.json"):
        record = json.loads(sidecar.read_text())
        record["created_at"] = (datetime.now() - timedelta(hours=25)).isoformat()
        sidecar.write_text(json.dumps(record))

    assert cleanup_guest_uploads(uploads, 24) == 2

    assert list(uploads.iterdir()) == []
    assert [p.stat().st_size for p in bundled] == sizes


def test_sample_is_copied_when_it_cannot_be_linked(uploads: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_link(src: object, dst: object) -> None:
        raise OSError("cross-device link")

    monkeypatch.setattr(sample_mod.os, "link", no_link)

    att = _client().post("/v1/sample").json()["attachments"][0]

    copied, bundled = Path(att["path"]), sample_mod.SAMPLE_DIR / att["filename"]
    assert not os.path.samefile(copied, bundled)
    assert copied.read_bytes() == bundled.read_bytes()


def test_sample_not_bundled(uploads: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sample_mod, "SAMPLE_DIR", tmp_path / "missing")

    r = _client().post("/v1/sample")

    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SAMPLE_NOT_BUNDLED"
    assert list(uploads.iterdir()) == []
