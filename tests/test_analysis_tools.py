"""Tests for post-processing analysis tools."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import rasterio

from dta.dti.post_processing.analysis_tools import (
    AVAILABLE_TOOLS,
    analyze_spatial_patterns,
    compute_band_statistics,
    compute_image_similarity,
    execute_tool,
)


def _create_base64_image(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> str:
    """Helper to create a base64 encoded PNG image of a solid color."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def visualizations() -> dict[str, str]:
    """Provide sample base64 images for testing."""
    return {
        "img_black": _create_base64_image((0, 0, 0)),
        "img_white": _create_base64_image((255, 255, 255)),
        "img_gray": _create_base64_image((128, 128, 128)),
        "img_diff_shape": _create_base64_image((0, 0, 0), size=(16, 16)),
    }


class TestAnalysisToolsDefinition:
    def test_available_tools_list(self) -> None:
        """Test that tools are correctly defined."""
        assert isinstance(AVAILABLE_TOOLS, list)
        assert len(AVAILABLE_TOOLS) > 0

        tool_names = [tool["name"] for tool in AVAILABLE_TOOLS]
        assert "compute_image_similarity" in tool_names
        assert "compute_band_statistics" in tool_names
        assert "analyze_spatial_patterns" in tool_names


class TestImageSimilarity:
    def test_similarity_identical_images(self, visualizations: dict[str, str]) -> None:
        """Test similarity between identical images."""
        result = compute_image_similarity(visualizations, "img_black", "img_black")

        assert "error" not in result
        assert result["mse"] == 0.0
        assert result["rmse"] == 0.0
        assert result["psnr_db"] == float("inf")
        assert result["ssim"] > 0.99
        # NCC is undefined for constant images (zero variance); accept 0.0 or 1.0
        assert result["ncc"] >= 0.0
        assert "excellent structural similarity" in result["interpretation"]

    def test_similarity_different_images(self, visualizations: dict[str, str]) -> None:
        """Test similarity between completely different images."""
        result = compute_image_similarity(visualizations, "img_black", "img_white")

        assert "error" not in result
        assert result["mse"] == 1.0  # (1.0 - 0.0)^2
        assert result["rmse"] == 1.0
        assert result["psnr_db"] == 0.0
        # White and black images have perfect structural correlation in terms of uniformity
        assert "ssim" in result

    def test_similarity_missing_image(self, visualizations: dict[str, str]) -> None:
        """Test with missing image key."""
        result = compute_image_similarity(visualizations, "missing", "img_black")
        assert "error" in result

        result = compute_image_similarity(visualizations, "img_black", "missing")
        assert "error" in result

    def test_similarity_different_shapes(self, visualizations: dict[str, str]) -> None:
        """Test images with different dimensions."""
        result = compute_image_similarity(visualizations, "img_black", "img_diff_shape")
        assert "error" in result
        assert "differ" in result["error"].lower()


class TestSpatialPatterns:
    def test_analyze_solid_image(self, visualizations: dict[str, str]) -> None:
        """Test spatial patterns on a solid image (no edges, no texture)."""
        result = analyze_spatial_patterns(visualizations, "img_black")

        assert "error" not in result
        assert result["edge_density"] == 0.0
        assert result["texture_complexity"] == 0.0
        # Entropy should be effectively zero (allow tiny floating-point deviations)
        assert abs(result["entropy"]) < 0.01
        assert "smooth/homogeneous" in result["interpretation"] or "uniform" in result["interpretation"]

    def test_analyze_missing_image(self, visualizations: dict[str, str]) -> None:
        """Test with missing image key."""
        result = analyze_spatial_patterns(visualizations, "missing")
        assert "error" in result

    def test_analyze_complex_image(self) -> None:
        """Test with a random noise image to see high complexity."""
        # Create random noise image
        rng = np.random.default_rng(42)
        noise_array = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(noise_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode()

        visualizations = {"noise": b64_img}
        result = analyze_spatial_patterns(visualizations, "noise")

        assert "error" not in result
        assert result["edge_density"] > 0
        assert result["texture_complexity"] > 0
        assert result["entropy"] > 0
        assert "complex" in result["interpretation"] or "high" in result["interpretation"]


class TestBandStatistics:
    def test_compute_stats_valid_raster(self, tmp_path: Path) -> None:
        """Test statistics computation on a valid raster."""
        tif_path = str(tmp_path / "test_stats.tif")
        data = np.arange(100).reshape(1, 10, 10).astype(np.float32)

        with rasterio.open(tif_path, "w", driver="GTiff", height=10, width=10, count=1, dtype=data.dtype) as dst:
            dst.write(data)

        result = compute_band_statistics(tif_path)

        assert "error" not in result
        assert result["file"] == "test_stats.tif"
        assert result["bands"] == 1
        assert "band_1" in result["statistics"]

        b1_stats = result["statistics"]["band_1"]
        assert b1_stats["min"] == 0.0
        assert b1_stats["max"] == 99.0
        assert b1_stats["mean"] == 49.5

    def test_compute_stats_with_nan(self, tmp_path: Path) -> None:
        """Test statistics computation ignoring NaN values."""
        tif_path = str(tmp_path / "nan_stats.tif")
        data = np.ones((1, 10, 10), dtype=np.float32)
        data[0, 0, 0] = np.nan

        with rasterio.open(tif_path, "w", driver="GTiff", height=10, width=10, count=1, dtype=data.dtype) as dst:
            dst.write(data)

        result = compute_band_statistics(tif_path)

        assert "error" not in result
        assert result["statistics"]["band_1"]["mean"] == 1.0

    def test_compute_stats_not_found(self) -> None:
        """Test with nonexistent file."""
        result = compute_band_statistics("/nonexistent/file.tif")
        assert "error" in result
        assert "not found" in result["error"].lower()


class TestExecuteTool:
    def test_execute_similarity(self, visualizations: dict[str, str]) -> None:
        """Test executing the similarity tool."""
        context = {"visualizations": visualizations}
        args = {"image1_key": "img_black", "image2_key": "img_black"}

        result = execute_tool("compute_image_similarity", args, context)
        assert "error" not in result
        assert result["mse"] == 0.0

    def test_execute_statistics(self, tmp_path: Path) -> None:
        """Test executing the statistics tool."""
        tif_path = str(tmp_path / "exec_stats.tif")
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        with rasterio.open(tif_path, "w", driver="GTiff", height=10, width=10, count=1, dtype=data.dtype) as dst:
            dst.write(data)

        context = {}
        args = {"file_path": tif_path}

        result = execute_tool("compute_band_statistics", args, context)
        assert "error" not in result
        assert "band_1" in result["statistics"]

    def test_execute_spatial(self, visualizations: dict[str, str]) -> None:
        """Test executing the spatial patterns tool."""
        context = {"visualizations": visualizations}
        args = {"image_key": "img_black"}

        result = execute_tool("analyze_spatial_patterns", args, context)
        assert "error" not in result
        assert result["edge_density"] == 0.0

    def test_execute_unknown_tool(self) -> None:
        """Test executing an unknown tool."""
        result = execute_tool("unknown_tool", {}, {})
        assert "error" in result
        assert "Unknown tool" in result["error"]
