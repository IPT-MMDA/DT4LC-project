"""Tests for change detection algorithm."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.dti.algorithms.change_detection import _create_change_visualization, run


@pytest.fixture
def dummy_tifs(tmp_path: Path):
    """Create two dummy TIFFs (before/after) for change detection."""
    before_path = tmp_path / "before.tif"
    after_path = tmp_path / "after.tif"

    transform = from_origin(0, 0, 10, 10)

    # Before: values 1.0 everywhere, 2 bands
    before_data = np.ones((2, 10, 10), dtype=np.float32)
    with rasterio.open(
        before_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=2,
        dtype=before_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(before_data)

    # After: values 2.0 everywhere, 2 bands
    after_data = np.full((2, 10, 10), 2.0, dtype=np.float32)
    with rasterio.open(
        after_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=2,
        dtype=after_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(after_data)

    return str(before_path), str(after_path)


def test_create_visualization(tmp_path: Path) -> None:
    """Test visualization generation."""
    change = np.array([[1.0, 0.0], [-0.5, 0.0]])

    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", True):
        res = _create_change_visualization(change, "ndvi")

    assert res is not None
    assert isinstance(res, bytes)
    assert res[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_change_detection(dummy_tifs) -> None:
    """Test full change detection execution."""
    before, after = dummy_tifs

    # Run change detection
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=before, RasterPathAfter=after, IndexType="ndvi")

    assert "change" in result["rasters"]

    # Check the change raster
    with rasterio.open(result["rasters"]["change"]["path"]) as src:
        change_data = src.read(1)
    assert np.all(change_data == 0.0)


@pytest.mark.parametrize(
    ("dtype", "scale", "expected"),
    [("float32", 1, 0.084135), ("int16", 10000, 0.084135), ("uint8", 255, 0.085389)],
)
def test_evi_scaled_by_source_dtype(write_scaled_scene, dtype: str, scale: int, expected: float) -> None:
    """EVI on the dark scene: int16 x10000 equals its float twin, uint8 is DN/255 (#64)."""
    before = write_scaled_scene(dtype, scale)
    after = write_scaled_scene("float32", 1)

    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=before, RasterPathAfter=after, IndexType="evi")

    with rasterio.open(result["rasters"]["index_before"]["path"]) as src:
        index_before = src.read(1).astype(float)
    assert round(float(np.min(index_before)), 6) == round(float(np.max(index_before)), 6) == expected
    if dtype != "uint8":
        assert round(result["statistics"]["mean_change"], 6) == 0.0


# --- Areas for the result card (v0.6.0 Workspace) ----------------------------------

SAMPLE_BEFORE = str(Path(__file__).resolve().parents[1] / "resources/kahovka_data/hlsl_20230601.tif")
SAMPLE_AFTER = str(Path(__file__).resolve().parents[1] / "resources/kahovka_data/hlsl_20230609.tif")


def test_ndwi_on_the_sample_reports_water_and_class_areas() -> None:
    """The Kakhovka pair (30 m, EPSG:32636): open water is NDWI > 0.2, one pixel is 0.0009 km².

    The numbers are what the code computes; DESIGN §8's 32 → 256 km², +428 / -43 are not reproduced.
    """
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=SAMPLE_BEFORE, RasterPathAfter=SAMPLE_AFTER, IndexType="ndwi")
    stats = result["statistics"]

    # Clipping each date to [-1, 1] moved 58 pixels to stable (51 from decrease, 7 from increase).
    assert stats["areas_km2"] == {"increase": 325.22, "decrease": 9.902, "stable": 195.718}
    assert stats["water"] == {
        "before_km2": 13.881,
        "after_km2": 170.039,
        "gained_km2": 156.761,
        "lost_km2": 0.603,
        "threshold": 0.2,
    }
    water = stats["water"]
    assert water["before_km2"] + water["gained_km2"] - water["lost_km2"] == pytest.approx(water["after_km2"], abs=1e-3)
    classes = result["classification"]
    increase = classes["moderate_water_gain"]["pixels"] + classes["strong_water_gain"]["pixels"]
    assert stats["areas_km2"]["increase"] == pytest.approx(increase * 0.0009, abs=1e-3)
    assert "vegetation" not in stats


def test_ndvi_on_the_sample_reports_vegetation() -> None:
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        stats = run(RasterPathBefore=SAMPLE_BEFORE, RasterPathAfter=SAMPLE_AFTER, IndexType="ndvi")["statistics"]

    assert stats["vegetation"] == {
        "before_km2": 437.651,
        "after_km2": 241.527,
        "gained_km2": 2.011,
        "lost_km2": 198.136,
        "threshold": 0.3,
    }
    assert "water" not in stats


def test_sample_indices_are_clipped_before_differencing() -> None:
    """Negative stored reflectance gave the sample's NDWI change -223 .. +318; each date is now in [-1, 1]."""
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=SAMPLE_BEFORE, RasterPathAfter=SAMPLE_AFTER, IndexType="ndwi")
    stats, index_stats = result["statistics"], result["index_statistics"]

    assert -2 <= stats["min_change"] < stats["max_change"] <= 2
    assert (round(stats["min_change"], 4), round(stats["max_change"], 4)) == (-1.6024, 2.0)
    assert round(stats["mean_change"], 4) == 0.3533
    for when in ("before", "after"):
        assert -1 <= index_stats[when]["min"] < index_stats[when]["max"] <= 1
    means = {name: c["mean_change"] for name, c in result["class_statistics"].items()}
    assert means == {"Water gained": 0.806, "Water lost": -0.233, "No change": 0.1642}
    assert result["rasters"]["change"]["stats"]["max"] <= 2


def test_index_beyond_its_range_is_clipped_and_zero_denominator_is_nan(tmp_path: Path) -> None:
    """NDVI of red -3, NIR 4 is 7 (clipped to 1); red -2, NIR 2 has no NDVI (NaN), as band math."""
    transform = from_origin(0, 0, 10, 10)
    paths = []
    for name, red, nir in (("before", [[-3.0, -2.0, 1.0]], [[4.0, 2.0, 3.0]]), ("after", [[1.0] * 3], [[1.0] * 3])):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=3,
            count=2,
            dtype="float32",
            crs="EPSG:32636",
            transform=transform,
        ) as dst:
            dst.write(np.array([red, nir], dtype=np.float32))
        paths.append(str(path))

    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=paths[0], RasterPathAfter=paths[1], IndexType="ndvi")
    with rasterio.open(result["rasters"]["index_before"]["path"]) as src:
        before = src.read(1, masked=True).filled(np.nan)
    with rasterio.open(result["rasters"]["change"]["path"]) as src:
        change = src.read(1, masked=True).filled(np.nan)

    np.testing.assert_allclose(before[0], [1.0, np.nan, 0.5])
    np.testing.assert_allclose(change[0], [-1.0, np.nan, -0.5])


def test_areas_are_null_for_a_geographic_crs(dummy_tifs) -> None:
    """Degrees have no fixed area: no km² for EPSG:4326."""
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        stats = run(RasterPathBefore=dummy_tifs[0], RasterPathAfter=dummy_tifs[1], IndexType="ndvi")["statistics"]

    assert stats["areas_km2"] is None
    assert stats["vegetation"] is None
