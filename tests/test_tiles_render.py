"""GET /v1/tiles: RGBA tiles in the colours of the raster's style, transparent off the raster.

A tagged change raster goes lost -> no change -> gained in the map legend colours
(design/DESIGN.md §2), an index raster keeps its PNG's colormap, and an untagged
file keeps the old file-name colours.
"""

import io
from pathlib import Path

from fastapi.testclient import TestClient
import mercantile
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

from dta.dti import artifact_files
from server import paths
from server.app import app

LOST, NO_CHANGE, GAINED = (140, 74, 47), (196, 202, 214), (45, 95, 158)  # #8c4a2f, #c4cad6, #2d5f9e
TILE = mercantile.Tile(x=600, y=360, z=10)


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An OUTPUTS_PATH root for served job artifacts."""
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(paths, "OUTPUTS_PATH", root)
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _tile_raster(path: Path, data: np.ndarray, style: dict | None, *, cols: int = 256) -> Path:
    """A float raster in EPSG:3857 whose 256 rows span ``TILE`` and whose ``cols`` columns span its left part."""
    b = mercantile.xy_bounds(TILE)
    transform = transform_from_bounds(b.left, b.bottom, b.right, b.top, 256, 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_files.write_geotiff(path, data[:, :cols], "EPSG:3857", transform, style)
    return path


def _get_tile(client: TestClient, tif: Path, tile: mercantile.Tile = TILE) -> np.ndarray:
    resp = client.get(f"/v1/tiles/{tile.z}/{tile.x}/{tile.y}", params={"path": str(tif)})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(resp.content))
    assert img.mode == "RGBA" and img.size == (256, 256)
    return np.asarray(img)


def _thirds(lo: float, mid: float, hi: float) -> np.ndarray:
    data = np.full((256, 256), mid, dtype=np.float32)
    data[:, :85] = lo
    data[:, 171:] = hi
    data[:32, :] = np.nan  # nodata band along the top
    return data


def test_change_tile_uses_lost_same_gained_ramp(client: TestClient, outputs: Path) -> None:
    tif = _tile_raster(
        outputs / "run-1" / "2_ChangeMap_change.tif", _thirds(-0.5, 0.0, 0.5), artifact_files.change_style(0.5)
    )

    tile = _get_tile(client, tif)

    assert tuple(tile[128, 40]) == (*LOST, 255)
    assert tuple(tile[128, 128]) == (*NO_CHANGE, 255)
    assert tuple(tile[128, 215]) == (*GAINED, 255)
    assert (tile[:30, :, 3] == 0).all()  # NaN is transparent


def test_change_tile_clips_to_the_display_range(client: TestClient, outputs: Path) -> None:
    # Band math: the range is the 98th percentile of |change|; beyond it the end colours hold.
    tif = _tile_raster(
        outputs / "run-1" / "2_BandMathChangeMap_change.tif", _thirds(-0.9, 0.0, 0.9), artifact_files.change_style(0.2)
    )

    tile = _get_tile(client, tif)

    assert tuple(tile[128, 40, :3]) == LOST and tuple(tile[128, 215, :3]) == GAINED


def test_index_tile_keeps_the_png_colormap(client: TestClient, outputs: Path) -> None:
    from dta.dti.algorithms import ndwi, spectral_index

    style = spectral_index.display_style(ndwi._config())
    assert style["ramp"] == [artifact_files.hex_colour(c) for c in spectral_index.COLORMAPS["ndwi"]]
    lo, hi = style["range"]
    tif = _tile_raster(outputs / "run-1" / "1_NDWIMap_ndwi.tif", _thirds(lo, (lo + hi) / 2, hi), style)

    tile = _get_tile(client, tif)

    first, last = spectral_index.COLORMAPS["ndwi"][0], spectral_index.COLORMAPS["ndwi"][-1]
    assert tuple(tile[128, 40, :3]) == tuple(round(c * 255) for c in first)
    assert tuple(tile[128, 215, :3]) == tuple(round(c * 255) for c in last)


def test_untagged_tile_keeps_old_colours_with_transparent_nodata(client: TestClient, outputs: Path) -> None:
    from server.utils import apply_colormap

    tif = _tile_raster(outputs / "run-1" / "legacy_ndvi.tif", _thirds(-1.0, 0.0, 1.0), None)

    tile = _get_tile(client, tif)

    assert tuple(tile[128, 215, :3]) == tuple(apply_colormap(np.array([[255]], dtype=np.uint8), "RdYlGn")[0, 0])
    assert tile[128, 215, 3] == 255 and (tile[:30, :, 3] == 0).all()


def test_edge_tile_is_transparent_off_the_raster(client: TestClient, outputs: Path) -> None:
    # The raster covers the left 100 of the tile's 256 columns: nothing is stretched over the rest.
    tif = _tile_raster(
        outputs / "run-1" / "2_ChangeMap_change.tif",
        np.full((256, 256), 0.5, np.float32),
        artifact_files.change_style(0.5),
        cols=100,
    )

    tile = _get_tile(client, tif)

    assert (tile[:, :99, 3] == 255).all()
    assert (tile[:, 101:, 3] == 0).all()
    assert tuple(tile[128, 50, :3]) == GAINED


def test_edge_tile_of_rgb_upload_is_transparent_off_the_raster(client: TestClient, outputs: Path) -> None:
    b = mercantile.xy_bounds(TILE)
    tif = outputs / "run-1" / "scene.tif"
    tif.parent.mkdir()
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=100,
        height=256,
        count=3,
        dtype="uint16",
        crs="EPSG:3857",
        transform=transform_from_bounds(b.left, b.bottom, b.right, b.top, 256, 256),
    ) as dst:
        dst.write(np.arange(3 * 256 * 100, dtype=np.uint16).reshape(3, 256, 100))

    tile = _get_tile(client, tif)

    assert (tile[:, :99, 3] == 255).all() and (tile[:, 101:, 3] == 0).all()


def test_tile_outside_the_raster_is_a_transparent_png_not_500(client: TestClient, outputs: Path) -> None:
    # v0.5.1 follow-up: from_bounds gave a non-empty window off the raster and the read failed with 500.
    tif = _tile_raster(
        outputs / "run-1" / "2_ChangeMap_change.tif", _thirds(-0.5, 0.0, 0.5), artifact_files.change_style(0.5)
    )

    for far in (mercantile.Tile(0, 0, 10), mercantile.Tile(TILE.x + 1, TILE.y, TILE.z)):
        tile = _get_tile(client, tif, far)
        assert (tile[..., 3] == 0).all()


def test_tile_of_a_non_raster_file_is_415_without_the_path(client: TestClient, outputs: Path) -> None:
    """BE26: a served file that rasterio can't open is a 415, and the body doesn't echo the server path."""
    result = outputs / "run-1" / "result.json"
    result.parent.mkdir()
    result.write_text('{"ok": true}')

    resp = client.get(f"/v1/tiles/{TILE.z}/{TILE.x}/{TILE.y}", params={"path": str(result)})

    assert resp.status_code == 415
    assert resp.json() == {
        "ok": False,
        "error": {"code": "NOT_A_RASTER", "message": "This file isn't a raster the map can display.", "details": {}},
    }
    assert str(result.resolve()) not in resp.text and str(outputs) not in resp.text
