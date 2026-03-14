"""Tests for algorithm implementations.

Tests NDVI, Statistics, and Change Detection algorithms.
"""

from pathlib import Path

import pytest


class TestNDVIAlgorithm:
    """Tests for NDVI algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_ndvi_calculation_with_real_data(self) -> None:
        """Test NDVI algorithm with real data."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.ndvi import calculate_ndvi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_ndvi(str(tif_files[0]))

        assert "ndvi_array" in result
        assert "metadata" in result
        assert "statistics" in result

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_ndvi_statistics_valid(self) -> None:
        """Test that NDVI statistics are valid."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.ndvi import calculate_ndvi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_ndvi(str(tif_files[0]))
        stats = result["statistics"]

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["valid_pixels"] > 0

    def test_ndvi_file_not_found(self) -> None:
        """Test NDVI raises error for non-existent file."""
        from dta.dti.algorithms.ndvi import calculate_ndvi

        with pytest.raises(FileNotFoundError):
            calculate_ndvi("/nonexistent/path.tif")

    def test_ndvi_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.ndvi import run

        assert callable(run)


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
    """Tests for NDSI (snow index) algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_ndsi_calculation_with_real_data(self) -> None:
        """Test NDSI algorithm with real data."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.ndsi import calculate_ndsi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_ndsi(str(tif_files[0]))

        assert "ndsi_array" in result
        assert "snow_mask" in result
        assert "metadata" in result
        assert "statistics" in result

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_ndsi_statistics_valid(self) -> None:
        """Test that NDSI statistics are valid and include snow metrics."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.ndsi import calculate_ndsi

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_ndsi(str(tif_files[0]))
        stats = result["statistics"]

        # Standard statistics
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["valid_pixels"] > 0

        # Snow-specific statistics
        assert "snow_threshold" in stats
        assert stats["snow_threshold"] == 0.42
        assert "snow_pixels" in stats
        assert "non_snow_pixels" in stats
        assert "snow_coverage_percent" in stats

    def test_ndsi_file_not_found(self) -> None:
        """Test NDSI raises error for non-existent file."""
        from dta.dti.algorithms.ndsi import calculate_ndsi

        with pytest.raises(FileNotFoundError):
            calculate_ndsi("/nonexistent/path.tif")

    def test_ndsi_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.ndsi import run

        assert callable(run)


class TestSnowClassifierAlgorithm:
    """Tests for snow classifier algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_snow_classifier_with_real_data(self) -> None:
        """Test snow classifier with real data."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.snow_classifier import classify_snow

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = classify_snow(str(tif_files[0]))

        assert "snow_mask" in result
        assert "ndsi" in result
        assert "ndvi" in result
        assert "brightness" in result
        assert "metadata" in result
        assert "statistics" in result

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_snow_classifier_statistics(self) -> None:
        """Test that snow classifier statistics include all criteria info."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.snow_classifier import classify_snow

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = classify_snow(str(tif_files[0]))
        stats = result["statistics"]

        # Basic stats
        assert "snow_pixels" in stats
        assert "non_snow_pixels" in stats
        assert "snow_coverage_percent" in stats

        # Criteria stats
        assert "pixels_meeting_ndsi_threshold" in stats
        assert "pixels_meeting_ndvi_criterion" in stats
        assert "pixels_meeting_brightness_threshold" in stats

        # Thresholds
        assert "thresholds" in stats
        assert stats["thresholds"]["ndsi"] == 0.4
        assert stats["thresholds"]["ndvi_center"] == 0.1
        assert stats["thresholds"]["brightness"] == 0.3

    def test_snow_classifier_file_not_found(self) -> None:
        """Test snow classifier raises error for non-existent file."""
        from dta.dti.algorithms.snow_classifier import classify_snow

        with pytest.raises(FileNotFoundError):
            classify_snow("/nonexistent/path.tif")

    def test_snow_classifier_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.snow_classifier import run

        assert callable(run)


class TestStatisticsAlgorithm:
    """Tests for statistics algorithm."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_statistics_calculation(self) -> None:
        """Test statistics calculation with real data."""
        from dta.config import ROOT_DIR
        from dta.dti.algorithms.statistics import calculate_statistics

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        result = calculate_statistics(str(tif_files[0]))

        # Statistics module returns 'bands' with per-band statistics
        assert "bands" in result
        assert "metadata" in result or "path" in result

    def test_statistics_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.statistics import run

        assert callable(run)


class TestChangeDetectionAlgorithm:
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
        tif_files = sorted(list(data_dir.glob("*.tif")))

        if len(tif_files) < 2:
            pytest.skip("Need at least 2 GeoTIFF files for change detection")

        result = calculate_change(str(tif_files[0]), str(tif_files[1]))

        # Check result contains expected keys
        assert "change_map" in result or "change_array" in result or "outputs" in result
        assert "statistics" in result or "metadata" in result

    def test_change_detection_run_function_exists(self) -> None:
        """Test that run() function exists for registry integration."""
        from dta.dti.algorithms.change_detection import run

        assert callable(run)


class TestAlgorithmEntrypoints:
    """Tests for algorithm entrypoint consistency."""

    def test_all_algorithms_have_run_function(self) -> None:
        """Test that all algorithms have run() function."""
        from dta.dti.algorithms import change_detection, ndsi, ndvi, snow_classifier, statistics

        assert hasattr(ndvi, "run")
        assert hasattr(ndsi, "run")
        assert hasattr(snow_classifier, "run")
        assert hasattr(statistics, "run")
        assert hasattr(change_detection, "run")

    def test_algorithm_files_exist(self) -> None:
        """Test that algorithm files exist."""
        from dta.config import ROOT_DIR

        algorithms_dir = ROOT_DIR / "dta/dti/algorithms"

        assert (algorithms_dir / "ndvi.py").exists()
        assert (algorithms_dir / "ndsi.py").exists()
        assert (algorithms_dir / "snow_classifier.py").exists()
        assert (algorithms_dir / "statistics.py").exists()
        assert (algorithms_dir / "change_detection.py").exists()
