"""GET /v1/tiles over the UTM sample: tiles reprojected to Web Mercator, and the true-colour style.

Before v0.6.0 run 2 a tile read the UTM envelope of its bounds linearly, so the edges of adjacent
tiles did not meet (1-2 px steps at z12-14). The stitched tiles are compared here with an
independent reprojection: each mosaic pixel centre transformed to UTM by PROJ and sampled there.
"""

import io
import os
from pathlib import Path
import shutil

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
from server.routes import tiles
from server.sample import _true_colour_png

SAMPLE = RESOURCES_PATH / "kahovka_data" / "hlsl_20230601.tif"
CENTRE = (32.54, 46.56)  # lon, lat of the sample's centre
WATER = (32.4682, 46.6219)  # a Dnipro channel on 1 June (NDWI 0.38 over a 7×7 window)

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


def test_sample_renders_in_true_colour(client: TestClient, run_dir: Path) -> None:
    tif = Path(shutil.copy(SAMPLE, run_dir / SAMPLE.name))
    tile = mercantile.tile(*WATER, 13)
    b = mercantile.xy_bounds(tile)
    (x,), (y,) = transform("EPSG:4326", "EPSG:3857", [WATER[0]], [WATER[1]])
    col, row = int((x - b.left) / (b.right - b.left) * 256), int((b.top - y) / (b.top - b.bottom) * 256)

    rgba = _tile(client, tif, tile, style="truecolor")

    r, g, bl, a = (int(v) for v in rgba[row, col])
    assert a == 255
    assert bl > r  # red is B4 and blue B2, each stretched on its own; swapped, this pixel has B < R
    # The same colours as the sample's true-colour preview PNG (bilinear vs the source pixel: a few levels).
    with rasterio.open(tif) as src:
        preview = np.asarray(Image.open(io.BytesIO(_true_colour_png(src))))
        (ux,), (uy,) = transform("EPSG:4326", src.crs, [WATER[0]], [WATER[1]])
        src_row, src_col = src.index(ux, uy)
    assert np.abs(preview[src_row, src_col].astype(int) - [r, g, bl]).max() <= 8
    # Untagged and with resolvable red, green and blue bands: true colour is the default.
    assert (_tile(client, tif, tile) == rgba).all()


def _pixel(src: rasterio.DatasetReader, tile: mercantile.Tile, row: int, col: int) -> tuple[int, int]:
    """(row, col) in ``tile`` of the centre of source pixel (``row``, ``col``)."""
    (x,), (y,) = transform(src.crs, "EPSG:3857", *([v] for v in src.xy(row, col)))
    b = mercantile.xy_bounds(tile)
    return int((b.top - y) / (b.top - b.bottom) * 256), int((x - b.left) / (b.right - b.left) * 256)


def test_true_colour_nodata_is_transparent(client: TestClient, run_dir: Path) -> None:
    tif = run_dir / "holes.tif"
    with rasterio.open(SAMPLE) as src:
        data, profile, descriptions = src.read(), src.profile, src.descriptions
    assert profile["nodata"] == -9999
    data[2, 300:400, 300:400] = -9999  # red only: one band's nodata makes the pixel transparent
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = descriptions

    with rasterio.open(tif) as src:
        (lon,), (lat,) = transform(src.crs, "EPSG:4326", *([v] for v in src.xy(350, 350)))
        tile = mercantile.tile(lon, lat, 12)
        hole, beside = _pixel(src, tile, 350, 350), _pixel(src, tile, 350, 260)

    rgba = _tile(client, tif, tile, style="truecolor")

    assert rgba[hole][3] == 0
    assert rgba[beside][3] == 255 and rgba[beside][:3].any()


def test_true_colour_stretch_is_computed_once_per_file_version(client: TestClient, run_dir: Path) -> None:
    tif = Path(shutil.copy(SAMPLE, run_dir / SAMPLE.name))
    tiles.true_colour_stretch.cache_clear()
    c = mercantile.tile(*CENTRE, 13)

    for dx in (-1, 0, 1):
        _tile(client, tif, mercantile.Tile(c.x + dx, c.y, c.z), style="truecolor")

    info = tiles.true_colour_stretch.cache_info()
    assert (info.misses, info.hits) == (1, 2)
    bands, limits = tiles.true_colour_stretch(str(tif), tif.stat().st_mtime_ns)
    assert bands == [3, 2, 1]  # B4, B3, B2 from the band descriptions
    with rasterio.open(tif) as src:
        expected = [np.percentile(src.read(band).astype(np.float64), [2, 98]) for band in bands]
    assert np.allclose(limits, expected)

    stat = tif.stat()
    os.utime(tif, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # the file is rewritten
    _tile(client, tif, c, style="truecolor")
    assert tiles.true_colour_stretch.cache_info().misses == 2


def test_true_colour_needs_red_green_and_blue(client: TestClient, run_dir: Path) -> None:
    tif = run_dir / "1_NDWIMap_ndwi.tif"
    with rasterio.open(SAMPLE) as src:
        artifact_files.write_geotiff(tif, src.read(1).astype(np.float32), src.crs, src.transform)
    t = mercantile.tile(*CENTRE, 12)

    resp = client.get(f"/v1/tiles/{t.z}/{t.x}/{t.y}", params={"path": str(tif), "style": "truecolor"})

    assert resp.status_code == 422
    assert client.get(f"/v1/tiles/{t.z}/{t.x}/{t.y}", params={"path": str(tif), "style": "nope"}).status_code == 422
