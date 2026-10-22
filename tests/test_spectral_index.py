"""Spectral-index scaling, nodata masking and range clipping (#64).

Constant scene from the JOSS review: blue 0.04, green 0.08, red 0.05,
NIR 0.40, SWIR1 0.20, as float32 reflectance and as int16 reflectance x10000.
Dark scene from the v0.5.1 QA baseline (``write_scaled_scene``): blue 0.004,
green 0.008, red 0.005, NIR 0.040, SWIR1 0.020, so int16 x10000 stays < 1000.
"""

from pathlib import Path
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from dta.config import ROOT_DIR
from dta.dti.algorithms import change_detection, evi, ndsi, ndvi, ndwi, spectral_index
from dta.dti.raster import reflectance_divisor

REFLECTANCE = (0.04, 0.08, 0.05, 0.40, 0.20)  # blue, green, red, NIR, SWIR1
KAHOVKA_CROP = ROOT_DIR / "resources/kahovka_data/hlsl_20230601.tif"
# Git-ignored local data (~130 MB per scene); the crop is rows 640-1408, cols 256-1024 of it.
KAHOVKA_FULL = ROOT_DIR / "resources/kahovka_data_full/hlsl_20230601.tif"
KAHOVKA_WINDOW = Window(col_off=256, row_off=640, width=768, height=768)


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


@pytest.mark.parametrize(
    ("dtype", "data", "expected"),
    [
        ("uint8", [1.0, 10.0], 255.0),
        ("int16", [40.0, 400.0], 10000.0),
        ("uint16", [40.0, 400.0], 10000.0),
        ("int32", [40.0, 400.0], 10000.0),
        ("float32", [0.004, 0.04], 1.0),
        ("float64", [np.nan, 0.5], 1.0),
        ("float32", [np.nan, np.nan], 1.0),
        ("float32", [400.0, 4000.0], 10000.0),
    ],
)
def test_reflectance_divisor_by_dtype(dtype: str, data: list[float], expected: float) -> None:
    assert reflectance_divisor(dtype, np.array(data)) == expected


def test_reflectance_divisor_ignores_masked_values() -> None:
    data = np.ma.masked_array([0.5, 5000.0], mask=[False, True])
    assert reflectance_divisor("float64", data) == 1.0


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [("float32", 1), ("float64", 1), ("int16", 10000), ("uint16", 10000)],
)
@pytest.mark.parametrize(
    ("module", "expected"),
    [(evi, 0.084135), (ndvi, 0.777778), (ndwi, -0.666667), (ndsi, -0.428571)],
    ids=["evi", "ndvi", "ndwi", "ndsi"],
)
def test_dark_scene_index_scaled_by_dtype(write_scaled_scene, dtype: str, scale: int, module, expected: float) -> None:
    """Dark int16/uint16 x10000 equals its float twin; float 0-1 input is not rescaled."""
    stats = module.run(write_scaled_scene(dtype, scale))["statistics"]

    assert round(stats["mean"], 6) == round(stats["min"], 6) == round(stats["max"], 6) == expected


@pytest.mark.parametrize(
    ("dtype", "scale", "reflectance", "expected_evi"),
    [
        ("uint8", 255, (0.004, 0.008, 0.005, 0.040, 0.020), 0.085389),  # DN 1, 2, 1, 10, 5
        ("uint8", 255, (0.05, 0.08, 0.06, 0.40, 0.20), 0.622318),  # DN 13, 20, 15, 102, 51
        ("float32", 10000, REFLECTANCE, 0.625),  # legacy float-stored DN (max > 1000)
    ],
    ids=["uint8-dark", "uint8-bright", "float-dn"],
)
def test_evi_byte_and_float_dn_scaling(
    write_scaled_scene, dtype: str, scale: int, reflectance: tuple[float, ...], expected_evi: float
) -> None:
    stats = evi.run(write_scaled_scene(dtype, scale, reflectance))["statistics"]

    assert round(stats["mean"], 6) == expected_evi


def test_run_change_scales_both_dates_by_dtype(write_scaled_scene) -> None:
    """A dark int16 date against its float32 twin is no change."""
    before, after = write_scaled_scene("int16", 10000), write_scaled_scene("float32", 1)
    result = spectral_index.run_change(before, after, evi._config())

    assert round(result["statistics"]["mean_change"], 6) == 0.0
    assert round(result["statistics"]["max_change"], 6) == 0.0


@pytest.fixture(scope="module")
def kahovka_full_window(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The crop's window of the float64 original, kept as float64 with its NaN, transform and descriptions."""
    if not KAHOVKA_FULL.exists():
        pytest.skip("resources/kahovka_data_full/ absent (git-ignored local data, ~130 MB per scene)")
    path = tmp_path_factory.mktemp("kahovka") / "window_20230601.tif"
    with rasterio.open(KAHOVKA_FULL) as src:
        data = src.read(window=KAHOVKA_WINDOW)
        profile = {
            "driver": "GTiff",
            "height": KAHOVKA_WINDOW.height,
            "width": KAHOVKA_WINDOW.width,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "crs": src.crs,
            "transform": src.window_transform(KAHOVKA_WINDOW),
        }
        descriptions = src.descriptions
    assert profile["dtype"] == "float64"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(descriptions, start=1):
            dst.set_band_description(i, name)
    return str(path)


@pytest.mark.parametrize("module", [evi, ndvi, ndwi, ndsi], ids=["evi", "ndvi", "ndwi", "ndsi"])
def test_kahovka_crop_matches_float64_original(kahovka_full_window: str, module) -> None:
    """The committed int16 crop and the same window of the float64 original agree to 3 dp."""
    crop = module.run(str(KAHOVKA_CROP))["statistics"]
    full = module.run(kahovka_full_window)["statistics"]

    for key in ("mean", "std", "min", "max", "median"):
        assert crop[key] == pytest.approx(full[key], abs=5e-4), key
    assert crop["valid_pixels"] == full["valid_pixels"]
