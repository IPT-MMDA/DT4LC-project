"""GET /v1/tiles over the UTM sample: tiles reprojected to Web Mercator.

Before v0.6.0 run 2 a tile read the UTM envelope of its bounds linearly, so the edges of adjacent
tiles did not meet (1-2 px steps at z12-14). The stitched tiles are compared here with an
independent reprojection: each mosaic pixel centre transformed to UTM by PROJ and sampled there.
"""

import io
from pathlib import Path

from fastapi.testclient import TestClient
import mercantile
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.warp import transform
from scipy.ndimage import map_coordinates

from dta.config import RESOURCES_PATH
from dta.dti import artifact_files
from dta.dti.algorithms import ndwi, spectral_index
from server import paths
from server.app import app

SAMPLE = RESOURCES_PATH / "kahovka_data" / "hlsl_20230601.tif"
CENTRE = (32.54, 46.56)  # lon, lat of the sample's centre

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="the Kakhovka sample is not bundled")


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An ownerless ``OUTPUTS_PATH/run-1`` the tiles may read."""
    monkeypatch.setattr(paths, "OUTPUTS_PATH", tmp_path / "outputs")
    run = tmp_path / "outputs" / "run-1"
    run.mkdir(parents=True)
    return run


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _tile(client: TestClient, tif: Path, tile: mercantile.Tile, **params: str) -> np.ndarray:
    resp = client.get(f"/v1/tiles/{tile.z}/{tile.x}/{tile.y}", params={"path": str(tif), **params})
    assert resp.status_code == 200, resp.text
    img = Image.open(io.BytesIO(resp.content))
    assert img.mode == "RGBA" and img.size == (256, 256)
    return np.asarray(img)


def _mosaic(client: TestClient, tif: Path, z: int) -> tuple[np.ndarray, mercantile.Bbox]:
    """The 3×3 tiles around the sample's centre at ``z``, stitched, and their Web Mercator bounds."""
    c = mercantile.tile(*CENTRE, z)
    rows = [
        np.concatenate([_tile(client, tif, mercantile.Tile(c.x + dx, c.y + dy, z)) for dx in (-1, 0, 1)], axis=1)
        for dy in (-1, 0, 1)
    ]
    ul, lr = mercantile.xy_bounds(c.x - 1, c.y - 1, z), mercantile.xy_bounds(c.x + 1, c.y + 1, z)
    return np.concatenate(rows, axis=0).astype(np.float64), mercantile.Bbox(ul.left, lr.bottom, lr.right, ul.top)


def _reference(array: np.ndarray, src: rasterio.DatasetReader, bounds: mercantile.Bbox, order: int) -> np.ndarray:
    """``array`` on ``src``'s grid at the 768×768 pixel centres of ``bounds``; ``order`` 0 nearest, 1 bilinear."""
    px = (bounds.right - bounds.left) / 768
    xs = bounds.left + (np.arange(768) + 0.5) * px
    ys = bounds.top - (np.arange(768) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    ux, uy = transform("EPSG:3857", src.crs, gx.ravel(), gy.ravel())
    col = (np.asarray(ux) - src.transform.c) / src.transform.a
    row = (np.asarray(uy) - src.transform.f) / src.transform.e
    inside = (col >= 0) & (col < src.width) & (row >= 0) & (row < src.height)
    if order == 0:
        r = np.clip(np.floor(row).astype(int), 0, src.height - 1)
        c = np.clip(np.floor(col).astype(int), 0, src.width - 1)
        values = array[r, c]
    else:
        values = map_coordinates(array.astype(np.float64), [row - 0.5, col - 0.5], order=1, mode="nearest")
    return np.where(inside, values, np.nan).reshape(768, 768)


def _seam_ratio(mosaic: np.ndarray) -> float:
    """Largest mean colour step across an internal tile edge, over the mean step one pixel either side of it."""
    ratios = []
    for img in (mosaic, mosaic.transpose(1, 0, 2)):
        for edge in (256, 512):
            opaque = (img[:, edge - 2 : edge + 2, 3] == 255).all(axis=1)

            def step(i: int, img: np.ndarray = img, opaque: np.ndarray = opaque) -> np.float64:
                return np.float64(np.abs(img[opaque, i, :3] - img[opaque, i + 1, :3]).mean())

            with np.errstate(divide="ignore", invalid="ignore"):  # a flat edge: 0/0 is skipped, x/0 is a step
                ratios.append(step(edge - 1) / ((step(edge - 2) + step(edge)) / 2))
    return float(np.nanmax(ratios))


@pytest.mark.parametrize("z", [12, 13, 14])
def test_adjacent_tiles_meet_without_steps(client: TestClient, run_dir: Path, z: int) -> None:
    """Measured on the sample: index |dRGB| 0.01-0.18 and class mismatch 0.00-0.05 % now, against
    2.9-5.6 and 0.7-2.0 % with the linear UTM read; seam ratio <= 1.18 now, 2.1 to inf before."""
    with rasterio.open(SAMPLE) as src:
        green, nir = src.read(2).astype(np.float64), src.read(4).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):  # a few pixels are 0 in both bands: NaN, transparent
            ndwi_values = ((green - nir) / (green + nir)).astype(np.float32)
        classes = np.where(ndwi_values > 0, 1, 2).astype(np.float32)
        index_style = spectral_index.display_style(ndwi._config())
        class_style = artifact_files.class_style({1: "#2d5f9e", 2: "#c4cad6"})
        index_tif, class_tif = run_dir / "1_NDWIMap_ndwi.tif", run_dir / "1_water_class.tif"
        artifact_files.write_geotiff(index_tif, ndwi_values, src.crs, src.transform, index_style)
        artifact_files.write_geotiff(class_tif, classes, src.crs, src.transform, class_style)

        index_mosaic, bounds = _mosaic(client, index_tif, z)
        class_mosaic, _ = _mosaic(client, class_tif, z)
        index_ref = artifact_files.colourize(_reference(ndwi_values, src, bounds, 1), index_style).astype(np.float64)
        class_ref = artifact_files.colourize(_reference(classes, src, bounds, 0), class_style).astype(np.float64)

    both = (index_mosaic[..., 3] == 255) & (index_ref[..., 3] == 255)
    assert both.mean() > 0.5
    assert np.abs(index_mosaic[both, :3] - index_ref[both, :3]).mean() < 1.0
    either = (class_mosaic[..., 3] == 255) | (class_ref[..., 3] == 255)
    assert (np.abs(class_mosaic - class_ref).sum(axis=2) > 0)[either].mean() < 0.005
    assert _seam_ratio(index_mosaic) < 1.5
