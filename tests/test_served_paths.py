"""Path allowlist shared by GET /v1/tiles and GET /v1/download (server/paths.py)."""

from pathlib import Path

from fastapi.testclient import TestClient
import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from server import paths
from server.app import app

# A tile inside the test raster (32-33 E, 46-47 N)
TILE = mercantile.tile(32.5, 46.5, 10)
TILE_URL = f"/v1/tiles/{TILE.z}/{TILE.x}/{TILE.y}"


def _write_geotiff(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(3 * 100 * 100, dtype=np.uint16).reshape(3, 100, 100)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(32.0, 47.0, 0.01, 0.01),
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every allowed root at a temp dir so nothing lands in resources/.cache."""
    names = {
        "DELINEATE_TEMP_DIR": "delineate",
        "UPLOAD_DIR": "uploads",
        "TEMP_PATH": "tmp",
        "OUTPUTS_PATH": "outputs",
        "GEE_EXPORT_DIR": "gee_exports",
    }
    result = {}
    for attr, name in names.items():
        root = tmp_path / "roots" / name
        root.mkdir(parents=True)
        monkeypatch.setattr(paths, attr, root)
        result[name] = root
    return result


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_tile_of_system_file_returns_403(client: TestClient, roots: dict[str, Path]) -> None:
    resp = client.get("/v1/tiles/10/0/0", params={"path": "/etc/hostname"})
    assert resp.status_code == 403


def test_tile_of_geotiff_outside_roots_returns_403(client: TestClient, roots: dict[str, Path], tmp_path: Path) -> None:
    outside = _write_geotiff(tmp_path / "outside" / "scene.tif")
    resp = client.get(TILE_URL, params={"path": str(outside)})
    assert resp.status_code == 403


def test_tile_traversal_out_of_root_returns_403(client: TestClient, roots: dict[str, Path], tmp_path: Path) -> None:
    _write_geotiff(tmp_path / "outside" / "scene.tif")
    traversal = roots["uploads"] / ".." / ".." / "outside" / "scene.tif"
    resp = client.get(TILE_URL, params={"path": str(traversal)})
    assert resp.status_code == 403


def test_tile_missing_file_outside_roots_returns_403(
    client: TestClient, roots: dict[str, Path], tmp_path: Path
) -> None:
    # No existence oracle for paths outside the roots
    resp = client.get(TILE_URL, params={"path": str(tmp_path / "missing.tif")})
    assert resp.status_code == 403


@pytest.mark.parametrize(
    ("root", "relpath"),
    [
        ("uploads", "abc_scene.tif"),
        ("outputs", "job-123/1_NDVIMap_ndvi.tif"),
        ("gee_exports", "gee_export_abc.tif"),
    ],
)
def test_tile_of_allowed_geotiff_returns_png(
    client: TestClient, roots: dict[str, Path], root: str, relpath: str
) -> None:
    tif = _write_geotiff(roots[root] / relpath)
    resp = client.get(TILE_URL, params={"path": str(tif)})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


def test_tile_of_directory_returns_404(client: TestClient, roots: dict[str, Path]) -> None:
    job_dir = roots["outputs"] / "job-123"
    job_dir.mkdir()
    resp = client.get(TILE_URL, params={"path": str(job_dir)})
    assert resp.status_code == 404


def test_download_of_gee_export_returns_200(client: TestClient, roots: dict[str, Path]) -> None:
    tif = roots["gee_exports"] / "gee_export_abc.tif"
    tif.write_bytes(b"tif")
    resp = client.get("/v1/download", params={"path": str(tif)})
    assert resp.status_code == 200
    assert resp.content == b"tif"


def test_download_of_directory_returns_404(client: TestClient, roots: dict[str, Path]) -> None:
    job_dir = roots["outputs"] / "job-123"
    job_dir.mkdir()
    resp = client.get("/v1/download", params={"path": str(job_dir)})
    assert resp.status_code == 404


def test_gee_export_dir_matches_gee_export_module() -> None:
    """server.paths derives EXPORT_DIR itself to avoid importing ``ee``; keep them in sync."""
    existed = paths.GEE_EXPORT_DIR.exists()
    from dta.dti.data_sources import gee_export

    try:
        assert gee_export.EXPORT_DIR == paths.GEE_EXPORT_DIR
    finally:
        # Importing gee_export creates EXPORT_DIR; do not leave it behind
        if not existed and not any(paths.GEE_EXPORT_DIR.iterdir()):
            paths.GEE_EXPORT_DIR.rmdir()
