"""Spectral-index scaling, nodata masking and range clipping (#64).

Constant scene from the JOSS review: blue 0.04, green 0.08, red 0.05,
NIR 0.40, SWIR1 0.20, as float32 reflectance and as int16 reflectance x10000.
"""

from pathlib import Path
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.config import ROOT_DIR
from dta.dti.algorithms import change_detection, evi, ndsi, ndvi, ndwi

REFLECTANCE = (0.04, 0.08, 0.05, 0.40, 0.20)  # blue, green, red, NIR, SWIR1
KAHOVKA_CROP = ROOT_DIR / "resources/kahovka_data/hlsl_20230601.tif"


def _write_constant_scene(path: Path, dtype: str) -> None:
    """8x8 5-band constant scene; the top-left 2x2 block is nodata in every band."""
    data = np.stack([np.full((8, 8), v, dtype=np.float64) for v in REFLECTANCE])
    nodata: float | None
    if dtype == "int16":
        data = np.round(data * 10000)
        data[:, :2, :2] = -9999
        nodata = -9999
    else:
        data[:, :2, :2] = np.nan  # NaN nodata without a nodata tag, like the full HLS scenes
        nodata = None
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=5,
        dtype=dtype,
        nodata=nodata,
        crs="EPSG:32636",
        transform=from_origin(0, 240, 30, 30),
    ) as dst:
        dst.write(data.astype(dtype))


@pytest.mark.parametrize("dtype", ["float32", "int16"])
@pytest.mark.parametrize(
    ("module", "expected"),
    [(evi, 0.625), (ndvi, 0.777778), (ndwi, -0.666667), (ndsi, -0.428571)],
    ids=["evi", "ndvi", "ndwi", "ndsi"],
)
def test_constant_scene_index_independent_of_scaling(tmp_path: Path, dtype: str, module, expected: float) -> None:
    path = tmp_path / f"scene_{dtype}.tif"
    _write_constant_scene(path, dtype)

    stats = module.run(str(path))["statistics"]

    assert round(stats["mean"], 6) == expected
    assert round(stats["min"], 6) == expected
    assert round(stats["max"], 6) == expected
    assert stats["valid_pixels"] == 64 - 4
    assert stats["total_pixels"] == 64


@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_change_detection_evi_independent_of_scaling(tmp_path: Path, dtype: str) -> None:
    path = str(tmp_path / f"scene_{dtype}.tif")
    _write_constant_scene(Path(path), dtype)

    with rasterio.open(change_detection.run(path, path, "evi")["rasters"]["index_before"]["path"]) as src:
        index_before = src.read(1).astype(float)

    assert round(float(np.nanmin(index_before)), 6) == round(float(np.nanmax(index_before)), 6) == 0.625


def test_kahovka_crop_ndvi_statistics_exclude_nodata(tmp_path: Path) -> None:
    """NDVI on the committed HLS L30 crop stays in [-1, 1] and skips nodata pixels."""
    path = tmp_path / KAHOVKA_CROP.name
    shutil.copy(KAHOVKA_CROP, path)
    with rasterio.open(path, "r+") as dst:
        assert dst.nodata == -9999
        red = dst.read(3)
        red[:100, :100] = -9999
        dst.write(red, 3)

    stats = ndvi.run(str(path))["statistics"]

    assert -1.0 <= stats["min"] < stats["max"] <= 1.0
    assert stats["total_pixels"] == 768 * 768
    assert stats["valid_pixels"] == 768 * 768 - 100 * 100
