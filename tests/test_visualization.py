"""Tests for visualization and post-processing.

Tests NDVI map rendering, change map rendering, statistics charts, and insight generation.
"""

import base64

import numpy as np
import pytest

from dta.dti.post_processing import InsightGenerator, Visualizer, format_statistics


@pytest.fixture
def sample_ndvi_array() -> np.ndarray:
    """Create sample NDVI array."""
    np.random.seed(42)
    return np.random.uniform(-0.2, 0.9, (100, 100))


@pytest.fixture
def sample_change_array() -> np.ndarray:
    """Create sample change array."""
    np.random.seed(42)
    return np.random.uniform(-0.3, 0.3, (100, 100))


@pytest.fixture
def sample_stats() -> dict:
    """Create sample statistics."""
    return {
        "mean": 0.542,
        "std": 0.123,
        "min": 0.1,
        "max": 0.9,
        "count": 10000,
    }


class TestVisualizerInitialization:
    """Tests for visualizer initialization."""

    def test_default_initialization(self) -> None:
        """Test visualizer can be initialized with defaults."""
        viz = Visualizer()
        assert viz.dpi == 100
        assert viz.figsize == (10, 8)

    def test_custom_initialization(self) -> None:
        """Test visualizer with custom parameters."""
        viz = Visualizer(dpi=150, figsize=(12, 10))
        assert viz.dpi == 150
        assert viz.figsize == (12, 10)


class TestNDVIRendering:
    """Tests for NDVI map rendering."""

    def test_render_ndvi_map(self, sample_ndvi_array: np.ndarray) -> None:
        """Test NDVI map rendering."""
        viz = Visualizer()
        result = viz.render_ndvi_map(sample_ndvi_array)

        assert "image" in result
        assert "format" in result
        assert "colormap" in result
        assert "statistics" in result

        assert isinstance(result["image"], str)
        assert len(result["image"]) > 0

        try:
            base64.b64decode(result["image"])
        except Exception:
            pytest.fail("Image is not valid base64")

        assert result["format"] == "png"
        assert result["colormap"] == "ndvi"

        stats = result["statistics"]
        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats

        assert -1 <= stats["min"] <= 1
        assert -1 <= stats["max"] <= 1

    def test_render_ndvi_with_metadata(self, sample_ndvi_array: np.ndarray) -> None:
        """Test NDVI rendering with metadata."""
        viz = Visualizer()
        metadata = {"crs": "EPSG:4326", "bounds": [0, 0, 100, 100]}

        result = viz.render_ndvi_map(sample_ndvi_array, metadata)

        assert result["metadata"] == metadata


class TestChangeMapRendering:
    """Tests for change map rendering."""

    def test_render_change_map(self, sample_change_array: np.ndarray) -> None:
        """Test change map rendering."""
        viz = Visualizer()
        result = viz.render_change_map(sample_change_array)

        assert "image" in result
        assert "format" in result
        assert "colormap" in result
        assert "statistics" in result

        assert isinstance(result["image"], str)
        assert len(result["image"]) > 0

        assert result["format"] == "png"
        assert result["colormap"] == "RdBu_r"

        stats = result["statistics"]
        assert "mean_change" in stats
        assert "min_change" in stats
        assert "max_change" in stats
        assert "total_decrease" in stats
        assert "total_increase" in stats


class TestStatisticsChart:
    """Tests for statistics chart rendering."""

    def test_render_bar_chart(self, sample_stats: dict) -> None:
        """Test bar chart rendering."""
        viz = Visualizer()
        result = viz.render_statistics_chart(sample_stats, chart_type="bar")

        assert "image" in result
        assert "format" in result
        assert "chart_type" in result

        assert result["format"] == "png"
        assert result["chart_type"] == "bar"

        try:
            base64.b64decode(result["image"])
        except Exception:
            pytest.fail("Chart image is not valid base64")

    def test_render_histogram(self) -> None:
        """Test histogram rendering."""
        viz = Visualizer()
        stats = {"values": np.random.normal(0.5, 0.2, 1000)}

        result = viz.render_statistics_chart(stats, chart_type="histogram")

        assert result["chart_type"] == "histogram"
        assert "image" in result


class TestGeoJSONConversion:
    """Tests for GeoJSON conversion."""

    def test_to_geojson(self, sample_ndvi_array: np.ndarray) -> None:
        """Test GeoJSON conversion."""
        viz = Visualizer()
        result = viz.to_geojson(sample_ndvi_array, transform=None)

        assert "type" in result
        assert result["type"] == "FeatureCollection"
        assert "features" in result
        assert "crs" in result

        assert len(result["features"]) > 0
        assert result["features"][0]["type"] == "Feature"


class TestStatisticsFormatting:
    """Tests for statistics formatting."""

    def test_format_statistics(self, sample_stats: dict) -> None:
        """Test statistics formatting."""
        formatted = format_statistics(sample_stats)

        assert "**Statistics:**" in formatted
        assert "Mean:" in formatted
        assert "0.542" in formatted
        assert "Count:" in formatted
        assert "10,000" in formatted


class TestInsightGenerator:
    """Tests for insight generator."""

    def test_initialization(self) -> None:
        """Test insight generator initialization."""
        gen = InsightGenerator()
        assert gen.llm_router is None

    def test_template_ndvi_insights(self) -> None:
        """Test template-based NDVI insights."""
        gen = InsightGenerator()

        ndvi_data = {
            "statistics": {
                "mean": 0.65,
                "std": 0.15,
                "min": 0.2,
                "max": 0.9,
            }
        }

        insights = gen._template_ndvi_insights(ndvi_data)

        assert isinstance(insights, str)
        assert len(insights) > 0
        assert "0.65" in insights
        assert "vegetation" in insights.lower()

    def test_template_change_insights(self) -> None:
        """Test template-based change insights."""
        gen = InsightGenerator()

        change_data = {
            "statistics": {
                "mean_change": -0.15,
                "total_decrease": 500.0,
                "total_increase": 200.0,
            }
        }

        insights = gen._template_change_insights(change_data)

        assert isinstance(insights, str)
        assert len(insights) > 0
        assert "-0.15" in insights
        assert "500.00" in insights

    def test_ndvi_insights_fallback(self) -> None:
        """Test NDVI insights with fallback when LLM unavailable."""
        gen = InsightGenerator(llm_router=None)

        ndvi_data = {
            "statistics": {
                "mean": 0.55,
                "std": 0.12,
                "min": 0.1,
                "max": 0.85,
            }
        }

        insights = gen.generate_ndvi_insights(ndvi_data)

        assert isinstance(insights, str)
        assert len(insights) > 0
        assert "0.55" in insights or "vegetation" in insights.lower()

    def test_change_insights_fallback(self) -> None:
        """Test change insights with fallback when LLM unavailable."""
        gen = InsightGenerator(llm_router=None)

        change_data = {
            "statistics": {
                "mean_change": 0.08,
                "total_decrease": 100.0,
                "total_increase": 300.0,
            }
        }

        insights = gen.generate_change_insights(change_data)

        assert isinstance(insights, str)
        assert len(insights) > 0


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    def test_render_ndvi(self, sample_ndvi_array: np.ndarray) -> None:
        """Test render_ndvi convenience function."""
        from dta.dti.post_processing.visualization import render_ndvi

        result = render_ndvi(sample_ndvi_array)
        assert "image" in result
        assert result["colormap"] == "ndvi"

    def test_render_change(self, sample_change_array: np.ndarray) -> None:
        """Test render_change convenience function."""
        from dta.dti.post_processing.visualization import render_change

        result = render_change(sample_change_array)
        assert "image" in result
        assert result["colormap"] == "RdBu_r"

    def test_render_chart(self) -> None:
        """Test render_chart convenience function."""
        from dta.dti.post_processing.visualization import render_chart

        stats = {"mean": 0.5, "max": 1.0}
        result = render_chart(stats)
        assert "image" in result
