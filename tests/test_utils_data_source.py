"""Tests for data source detection utilities."""

from pathlib import Path

import pytest

from dta.dti.utils.data_source import detect_data_source, get_filtering_thresholds


class TestDetectDataSource:
    """Tests for detect_data_source function."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_detect_sentinel_from_kahovka(self) -> None:
        """Test detection of Sentinel-like resolution from Kahovka data."""
        from dta.config import ROOT_DIR

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        # Kahovka data is Sentinel-2 (10m resolution)
        source = detect_data_source(str(tif_files[0]))
        assert source == "sentinel", f"Expected 'sentinel' for Kahovka data, got '{source}'"

    def test_detect_nonexistent_file_defaults_to_sentinel(self) -> None:
        """Test that non-existent file returns default 'sentinel'."""
        source = detect_data_source("/nonexistent/path.tif")
        assert source == "sentinel"


class TestGetFilteringThresholds:
    """Tests for get_filtering_thresholds function."""

    def test_sentinel_thresholds(self) -> None:
        """Test Sentinel-2 filtering thresholds."""
        thresholds = get_filtering_thresholds("sentinel")
        assert thresholds["minimum_area_m2"] == 2500
        assert thresholds["minimum_hole_area_m2"] == 2500

    def test_planet_thresholds(self) -> None:
        """Test Planet filtering thresholds."""
        thresholds = get_filtering_thresholds("planet")
        assert thresholds["minimum_area_m2"] == 1000
        assert thresholds["minimum_hole_area_m2"] == 1000

    def test_maxar_thresholds(self) -> None:
        """Test Maxar filtering thresholds."""
        thresholds = get_filtering_thresholds("maxar")
        assert thresholds["minimum_area_m2"] == 500
        assert thresholds["minimum_hole_area_m2"] == 500

    def test_unknown_source_defaults_to_sentinel(self) -> None:
        """Test unknown data source returns Sentinel defaults."""
        thresholds = get_filtering_thresholds("unknown")
        assert thresholds["minimum_area_m2"] == 2500
        assert thresholds["minimum_hole_area_m2"] == 2500
