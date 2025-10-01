"""Phase 3 Tests - Visualization & Post-Processing."""

import base64

import numpy as np
import pytest

from dta.dti.post_processing import InsightGenerator, Visualizer, format_statistics


@pytest.fixture
def sample_ndvi_array() -> np.ndarray:
    """Create sample NDVI array."""
    # 100x100 array with realistic NDVI values
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


def test_visualizer_initialization() -> None:
    """Test visualizer can be initialized."""
    viz = Visualizer()
    assert viz.dpi == 100
    assert viz.figsize == (10, 8)

    viz_custom = Visualizer(dpi=150, figsize=(12, 10))
    assert viz_custom.dpi == 150
    assert viz_custom.figsize == (12, 10)


def test_render_ndvi_map(sample_ndvi_array: np.ndarray) -> None:
    """Test NDVI map rendering."""
    viz = Visualizer()
    result = viz.render_ndvi_map(sample_ndvi_array)

    # Check structure
    assert "image" in result
    assert "format" in result
    assert "colormap" in result
    assert "statistics" in result

    # Check image is base64
    assert isinstance(result["image"], str)
    assert len(result["image"]) > 0

    # Verify it's valid base64
    try:
        base64.b64decode(result["image"])
    except Exception:
        pytest.fail("Image is not valid base64")

    # Check format
    assert result["format"] == "png"
    assert result["colormap"] == "ndvi"

    # Check statistics
    stats = result["statistics"]
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats

    assert -1 <= stats["min"] <= 1
    assert -1 <= stats["max"] <= 1


def test_render_ndvi_with_metadata(sample_ndvi_array: np.ndarray) -> None:
    """Test NDVI rendering with metadata."""
    viz = Visualizer()
    metadata = {"crs": "EPSG:4326", "bounds": [0, 0, 100, 100]}

    result = viz.render_ndvi_map(sample_ndvi_array, metadata)

    assert result["metadata"] == metadata


def test_render_change_map(sample_change_array: np.ndarray) -> None:
    """Test change map rendering."""
    viz = Visualizer()
    result = viz.render_change_map(sample_change_array)

    # Check structure
    assert "image" in result
    assert "format" in result
    assert "colormap" in result
    assert "statistics" in result

    # Check image
    assert isinstance(result["image"], str)
    assert len(result["image"]) > 0

    # Check format
    assert result["format"] == "png"
    assert result["colormap"] == "RdBu_r"

    # Check statistics
    stats = result["statistics"]
    assert "mean_change" in stats
    assert "min_change" in stats
    assert "max_change" in stats
    assert "total_decrease" in stats
    assert "total_increase" in stats


def test_render_statistics_chart_bar(sample_stats: dict) -> None:
    """Test bar chart rendering."""
    viz = Visualizer()
    result = viz.render_statistics_chart(sample_stats, chart_type="bar")

    assert "image" in result
    assert "format" in result
    assert "chart_type" in result

    assert result["format"] == "png"
    assert result["chart_type"] == "bar"

    # Verify base64
    try:
        base64.b64decode(result["image"])
    except Exception:
        pytest.fail("Chart image is not valid base64")


def test_render_statistics_chart_histogram() -> None:
    """Test histogram rendering."""
    viz = Visualizer()
    stats = {"values": np.random.normal(0.5, 0.2, 1000)}

    result = viz.render_statistics_chart(stats, chart_type="histogram")

    assert result["chart_type"] == "histogram"
    assert "image" in result


def test_to_geojson(sample_ndvi_array: np.ndarray) -> None:
    """Test GeoJSON conversion."""
    viz = Visualizer()
    result = viz.to_geojson(sample_ndvi_array, transform=None)

    # Check structure
    assert "type" in result
    assert result["type"] == "FeatureCollection"
    assert "features" in result
    assert "crs" in result

    # Check features
    assert len(result["features"]) > 0
    assert result["features"][0]["type"] == "Feature"


def test_format_statistics(sample_stats: dict) -> None:
    """Test statistics formatting."""
    formatted = format_statistics(sample_stats)

    assert "**Statistics:**" in formatted
    assert "Mean:" in formatted
    assert "0.542" in formatted
    assert "Count:" in formatted
    assert "10,000" in formatted


def test_insight_generator_initialization() -> None:
    """Test insight generator initialization."""
    gen = InsightGenerator()
    assert gen.llm_router is None

    # Should create router on demand
    gen2 = InsightGenerator()
    # Don't call _get_router() yet - just check initialization


def test_insight_generator_template_ndvi() -> None:
    """Test template-based NDVI insights (fallback)."""
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
    assert "0.65" in insights  # Mean value
    assert "vegetation" in insights.lower()


def test_insight_generator_template_change() -> None:
    """Test template-based change insights (fallback)."""
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
    assert "-0.15" in insights  # Mean change
    assert "500.00" in insights  # Decrease


def test_convenience_functions(sample_ndvi_array: np.ndarray, sample_change_array: np.ndarray) -> None:
    """Test convenience wrapper functions."""
    from dta.dti.post_processing.visualization import render_change, render_chart, render_ndvi

    # Test render_ndvi
    result = render_ndvi(sample_ndvi_array)
    assert "image" in result
    assert result["colormap"] == "ndvi"

    # Test render_change
    result2 = render_change(sample_change_array)
    assert "image" in result2
    assert result2["colormap"] == "RdBu_r"

    # Test render_chart
    stats = {"mean": 0.5, "max": 1.0}
    result3 = render_chart(stats)
    assert "image" in result3


def test_ndvi_insights_fallback() -> None:
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

    # This should fallback to template since no router
    insights = gen.generate_ndvi_insights(ndvi_data)

    assert isinstance(insights, str)
    assert len(insights) > 0
    assert "0.55" in insights or "vegetation" in insights.lower()


def test_change_insights_fallback() -> None:
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
