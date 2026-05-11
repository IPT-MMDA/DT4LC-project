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
    assert isinstance(res, str)
    assert len(res) > 100


def test_run_change_detection(dummy_tifs) -> None:
    """Test full change detection execution."""
    before, after = dummy_tifs

    # Run change detection
    with patch("dta.dti.algorithms.change_detection.HAS_MATPLOTLIB", False):
        result = run(RasterPathBefore=before, RasterPathAfter=after, IndexType="ndvi")

    assert "change_array" in result

    # Check the change array
    change_data = np.array(result["change_array"])
    assert np.all(change_data == 0.0)
