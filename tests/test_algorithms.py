"""Tests for algorithm functions.

Tests NDVI, NDWI, NDSI, LULC, snow classifier, and statistics
using synthetic GeoTIFF rasters created in-memory via rasterio.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


def _create_synthetic_geotiff(
    path: str,
    bands: int = 4,
    width: int = 32,
    height: int = 32,
    data: np.ndarray | None = None,
) -> str:
    """Create a synthetic GeoTIFF for testing.

    Args:
        path: Output file path
        bands: Number of bands
        width: Raster width
        height: Raster height
        data: Optional data array (shape: bands x height x width)

    Returns:
        Path to created file
    """
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.integers(100, 5000, (bands, height, width), dtype=np.int16)

    transform = from_bounds(0, 0, 1, 1, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def synthetic_tif(tmp_path: Path) -> str:
    """Create a 4-band synthetic GeoTIFF."""
    path = str(tmp_path / "synthetic.tif")
    return _create_synthetic_geotiff(path, bands=4)


@pytest.fixture
def synthetic_tif_2band(tmp_path: Path) -> str:
    """Create a 2-band synthetic GeoTIFF."""
    path = str(tmp_path / "synthetic_2b.tif")
    return _create_synthetic_geotiff(path, bands=2)


@pytest.fixture
def synthetic_tif_6band(tmp_path: Path) -> str:
    """Create a 6-band synthetic GeoTIFF."""
    path = str(tmp_path / "synthetic_6b.tif")
    return _create_synthetic_geotiff(path, bands=6)


@pytest.fixture
def synthetic_tif_7band(tmp_path: Path) -> str:
    """Create a 7-band synthetic GeoTIFF (Landsat-like)."""
    path = str(tmp_path / "synthetic_7b.tif")
    return _create_synthetic_geotiff(path, bands=7)


class TestNDVIAlgorithm:
    """Tests for NDVI calculation."""

    def test_ndvi_basic(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        result = calculate_ndvi(synthetic_tif)
        assert "statistics" in result
        assert "metadata" in result
        assert "visualizations" in result
        assert result["statistics"]["min"] >= -1.0
        assert result["statistics"]["max"] <= 1.0

    def test_ndvi_2band(self, synthetic_tif_2band: str) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        result = calculate_ndvi(synthetic_tif_2band)
        assert result["statistics"]["valid_pixels"] > 0

    def test_ndvi_7band(self, synthetic_tif_7band: str) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        result = calculate_ndvi(synthetic_tif_7band)
        assert "statistics" in result

    def test_ndvi_6band(self, synthetic_tif_6band: str) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        result = calculate_ndvi(synthetic_tif_6band)
        assert "statistics" in result

    def test_ndvi_file_not_found(self) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        with pytest.raises(FileNotFoundError):
            calculate_ndvi("/nonexistent/path.tif")

    def test_ndvi_insufficient_bands(self, tmp_path: Path) -> None:
        from dta.dti.algorithms.ndvi import calculate_ndvi

        path = str(tmp_path / "single_band.tif")
        _create_synthetic_geotiff(path, bands=1)

        with pytest.raises((IndexError, ValueError)):
            calculate_ndvi(path)

    def test_ndvi_run_function(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.ndvi import run

        result = run(synthetic_tif)
        assert "statistics" in result

    def test_ndvi_visualization(self, synthetic_tif: str) -> None:
        # _create_ndvi_visualization is a private implementation detail of spectral_index;
        # test the public interface instead.
        from dta.dti.algorithms.ndvi import calculate_ndvi

        result = calculate_ndvi(synthetic_tif)
        assert "visualizations" in result

    def test_ndvi_change(self) -> None:
        from dta.dti.algorithms.ndvi import ndvi_change

        before = {
            "ndvi_array": np.random.uniform(0.2, 0.6, (32, 32)),
            "metadata": {"crs": "EPSG:4326"},
            "path": "/before.tif",
        }
        after = {
            "ndvi_array": np.random.uniform(0.3, 0.7, (32, 32)),
            "metadata": {"crs": "EPSG:4326"},
            "path": "/after.tif",
        }
        result = ndvi_change(before, after)
        assert "change_array" in result
        assert "statistics" in result
        assert result["statistics"]["total_valid_pixels"] > 0


class TestNDWIAlgorithm:
    """Tests for NDWI calculation."""

    def test_ndwi_basic(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.ndwi import calculate_ndwi

        result = calculate_ndwi(synthetic_tif)
        assert "statistics" in result
        assert result["statistics"]["min"] >= -1.0
        assert result["statistics"]["max"] <= 1.0

    def test_ndwi_file_not_found(self) -> None:
        from dta.dti.algorithms.ndwi import calculate_ndwi

        with pytest.raises(FileNotFoundError):
            calculate_ndwi("/nonexistent.tif")

    def test_ndwi_run_function(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.ndwi import run

        result = run(synthetic_tif)
        assert "statistics" in result


class TestEVIAlgorithm:
    """Tests for EVI algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_evi_calculation_with_real_data(self) -> None:
        """Test EVI algorithm with real data."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.evi import calculate_evi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_evi(str(tif_files[0]))

        assert "evi_array" in result
        assert "metadata" in result
        assert "statistics" in result

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_evi_statistics_valid(self) -> None:
        """Test that EVI statistics are valid."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.evi import calculate_evi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_evi(str(tif_files[0]))
        stats = result["statistics"]

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["valid_pixels"] > 0

    def test_evi_file_not_found(self) -> None:
        """Test EVI raises error for non-existent file."""
        from dta.dti.algorithms.evi import calculate_evi

        with pytest.raises(FileNotFoundError):
            calculate_evi("/nonexistent/path.tif")

    def test_evi_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.evi import run

        assert callable(run)


class TestNDSIAlgorithm:
    """Tests for NDSI calculation."""

    def test_ndsi_basic(self, synthetic_tif_6band: str) -> None:
        from dta.dti.algorithms.ndsi import calculate_ndsi

        result = calculate_ndsi(synthetic_tif_6band)
        assert "statistics" in result
        assert result["statistics"]["min"] >= -1.0
        assert result["statistics"]["max"] <= 1.0

    def test_ndsi_file_not_found(self) -> None:
        from dta.dti.algorithms.ndsi import calculate_ndsi

        with pytest.raises(FileNotFoundError):
            calculate_ndsi("/nonexistent.tif")

    def test_ndsi_run_function(self, synthetic_tif_6band: str) -> None:
        from dta.dti.algorithms.ndsi import run

        result = run(synthetic_tif_6band)
        assert "statistics" in result


class TestLULCClassifier:
    """Tests for LULC classification."""

    def test_lulc_basic(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.lulc_classifier import classify_land_cover

        result = classify_land_cover(synthetic_tif)
        assert "statistics" in result
        assert "class_names" in result or "classes" in result or "classification" in result

    def test_lulc_file_not_found(self) -> None:
        from dta.dti.algorithms.lulc_classifier import classify_land_cover

        with pytest.raises(FileNotFoundError):
            classify_land_cover("/nonexistent.tif")

    def test_lulc_run_function(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.lulc_classifier import run

        result = run(synthetic_tif)
        assert isinstance(result, dict)


class TestSnowClassifier:
    """Tests for snow/ice classification."""

    def test_snow_basic(self, synthetic_tif_6band: str) -> None:
        from dta.dti.algorithms.snow_classifier import classify_snow

        result = classify_snow(synthetic_tif_6band)
        assert "statistics" in result

    def test_snow_file_not_found(self) -> None:
        from dta.dti.algorithms.snow_classifier import classify_snow

        with pytest.raises(FileNotFoundError):
            classify_snow("/nonexistent.tif")

    def test_snow_run_function(self, synthetic_tif_6band: str) -> None:
        from dta.dti.algorithms.snow_classifier import run

        result = run(synthetic_tif_6band)
        assert isinstance(result, dict)


class TestStatisticsAlgorithm:
    """Tests for statistics extraction."""

    def test_statistics_basic(self, synthetic_tif: str) -> None:
        from dta.dti.algorithms.statistics import run

        result = run(synthetic_tif)
        assert isinstance(result, dict)
        # Should contain per-band statistics
        assert "statistics" in result or "bands" in result or "per_band" in result

    def test_statistics_2band(self, synthetic_tif_2band: str) -> None:
        from dta.dti.algorithms.statistics import run

        result = run(synthetic_tif_2band)
        assert isinstance(result, dict)


class TestChangeDetection:
    """Tests for change detection algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_change_detection_with_real_data(self) -> None:
        """Test change detection with before/after images."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.change_detection import calculate_change

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = sorted(data_dir.glob("*.tif"))

        if len(tif_files) < 2:
            pytest.skip("Need at least 2 GeoTIFF files for change detection")

        result = calculate_change(str(tif_files[0]), str(tif_files[1]))

        # Check result contains expected keys
        assert "change_map" in result or "change_array" in result or "outputs" in result
        assert "statistics" in result or "metadata" in result

    def test_change_detection(self, tmp_path: Path) -> None:
        from dta.dti.algorithms.change_detection import run

        before = str(tmp_path / "before.tif")
        after = str(tmp_path / "after.tif")
        _create_synthetic_geotiff(before, bands=4)
        _create_synthetic_geotiff(after, bands=4)

        result = run(RasterPathBefore=before, RasterPathAfter=after)
        assert isinstance(result, dict)
        assert "statistics" in result or "change" in str(result).lower()

    def test_change_detection_file_not_found(self) -> None:
        from dta.dti.algorithms.change_detection import run

        with pytest.raises((FileNotFoundError, Exception)):
            run(RasterPathBefore="/nonexistent_before.tif", RasterPathAfter="/nonexistent_after.tif")
