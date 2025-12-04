"""Analysis tools for agentic interpretation.

Provides tool functions that the LLM can request during interpretation.
These tools compute metrics, statistics, and other derived values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Tool definitions for LLM function calling
AVAILABLE_TOOLS = [
    {
        "name": "compute_image_similarity",
        "description": "Compute similarity metrics (MSE, SSIM, PSNR) between two images. "
        "Useful for evaluating reconstruction quality.",
        "parameters": {
            "type": "object",
            "properties": {
                "image1_key": {
                    "type": "string",
                    "description": "Key for the first image in visualizations (e.g., 'original_rgb_t0')",
                },
                "image2_key": {
                    "type": "string",
                    "description": "Key for the second image in visualizations (e.g., 'predicted_rgb_t0')",
                },
            },
            "required": ["image1_key", "image2_key"],
        },
    },
    {
        "name": "compute_band_statistics",
        "description": "Compute statistics (mean, std, min, max, percentiles) for a raster file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the raster file",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "analyze_spatial_patterns",
        "description": "Analyze spatial autocorrelation and texture metrics in an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_key": {
                    "type": "string",
                    "description": "Key for the image in visualizations",
                },
            },
            "required": ["image_key"],
        },
    },
]


def compute_image_similarity(
    visualizations: dict[str, str],
    image1_key: str,
    image2_key: str,
) -> dict[str, Any]:
    """Compute similarity metrics between two images.

    Args:
        visualizations: Dict of image key -> base64 PNG data
        image1_key: Key for first image
        image2_key: Key for second image

    Returns:
        Dictionary with MSE, SSIM, PSNR metrics
    """
    try:
        import base64
        import io

        from PIL import Image

        if image1_key not in visualizations:
            return {"error": f"Image '{image1_key}' not found. Available: {list(visualizations.keys())}"}
        if image2_key not in visualizations:
            return {"error": f"Image '{image2_key}' not found. Available: {list(visualizations.keys())}"}

        # Decode base64 images
        img1_data = base64.b64decode(visualizations[image1_key])
        img2_data = base64.b64decode(visualizations[image2_key])

        img1 = np.array(Image.open(io.BytesIO(img1_data)).convert("RGB"))
        img2 = np.array(Image.open(io.BytesIO(img2_data)).convert("RGB"))

        # Ensure same shape
        if img1.shape != img2.shape:
            return {"error": f"Image shapes differ: {img1.shape} vs {img2.shape}"}

        # Normalize to 0-1 for metrics
        img1_norm = img1.astype(np.float64) / 255.0
        img2_norm = img2.astype(np.float64) / 255.0

        # MSE (Mean Squared Error)
        mse = np.mean((img1_norm - img2_norm) ** 2)

        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = float("inf")

        # SSIM (Structural Similarity Index) - simplified implementation
        ssim = _compute_ssim(img1_norm, img2_norm)

        # Normalized cross-correlation
        ncc = _compute_ncc(img1_norm, img2_norm)

        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "psnr_db": float(psnr),
            "ssim": float(ssim),
            "ncc": float(ncc),
            "interpretation": _interpret_similarity(mse, ssim, psnr),
        }

    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}
    except Exception as e:
        logger.exception("Error computing image similarity")
        return {"error": str(e)}


def _compute_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 7) -> float:
    """Compute SSIM between two images.

    Simplified SSIM implementation without scipy dependency.
    """
    # Constants for stability (standard SSIM values)
    c1 = 0.01**2
    c2 = 0.03**2

    # Convert to grayscale if RGB
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)

    # Local means using uniform filter (simplified)
    kernel_size = window_size
    pad = kernel_size // 2

    def local_mean(img: np.ndarray) -> np.ndarray:
        """Compute local mean using convolution."""
        result = np.zeros_like(img)
        padded = np.pad(img, pad, mode="reflect")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                result[i, j] = np.mean(padded[i : i + kernel_size, j : j + kernel_size])
        return result

    # For efficiency, use global statistics as approximation
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return float(ssim)


def _compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute normalized cross-correlation."""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    img1_norm = img1_flat - np.mean(img1_flat)
    img2_norm = img2_flat - np.mean(img2_flat)

    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def _interpret_similarity(mse: float, ssim: float, psnr: float) -> str:
    """Generate brief interpretation of similarity metrics."""
    quality = []

    if ssim > 0.95:
        quality.append("excellent structural similarity")
    elif ssim > 0.85:
        quality.append("good structural similarity")
    elif ssim > 0.7:
        quality.append("moderate structural similarity")
    else:
        quality.append("low structural similarity")

    if psnr > 35:
        quality.append("high fidelity")
    elif psnr > 25:
        quality.append("acceptable fidelity")
    else:
        quality.append("noticeable distortion")

    return ", ".join(quality)


def compute_band_statistics(file_path: str) -> dict[str, Any]:
    """Compute statistics for a raster file.

    Args:
        file_path: Path to raster file

    Returns:
        Dictionary with per-band statistics
    """
    try:
        import rasterio

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        with rasterio.open(path) as src:
            stats = {}
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                valid_data = data[~np.isnan(data)]

                if len(valid_data) == 0:
                    continue

                stats[f"band_{band_idx}"] = {
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "median": float(np.median(valid_data)),
                    "p5": float(np.percentile(valid_data, 5)),
                    "p95": float(np.percentile(valid_data, 95)),
                }

            return {
                "file": str(path.name),
                "bands": src.count,
                "shape": f"{src.width}x{src.height}",
                "crs": str(src.crs) if src.crs else None,
                "statistics": stats,
            }

    except ImportError:
        return {"error": "rasterio not installed"}
    except Exception as e:
        logger.exception("Error computing band statistics")
        return {"error": str(e)}


def analyze_spatial_patterns(visualizations: dict[str, str], image_key: str) -> dict[str, Any]:
    """Analyze spatial patterns in an image.

    Args:
        visualizations: Dict of image key -> base64 PNG data
        image_key: Key for the image

    Returns:
        Dictionary with spatial pattern metrics
    """
    try:
        import base64
        import io

        from PIL import Image

        if image_key not in visualizations:
            return {"error": f"Image '{image_key}' not found. Available: {list(visualizations.keys())}"}

        img_data = base64.b64decode(visualizations[image_key])
        img = np.array(Image.open(io.BytesIO(img_data)).convert("L"))  # Grayscale
        img_norm = img.astype(np.float64) / 255.0

        # Edge density (gradient magnitude)
        gy, gx = np.gradient(img_norm)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        edge_density = float(np.mean(gradient_magnitude))

        # Local variance (texture measure)
        local_var = _local_variance(img_norm, window_size=5)
        texture_complexity = float(np.mean(local_var))

        # Entropy (information content)
        hist, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = float(-np.sum(hist * np.log2(hist)))

        return {
            "edge_density": edge_density,
            "texture_complexity": texture_complexity,
            "entropy": entropy,
            "interpretation": _interpret_spatial(edge_density, texture_complexity, entropy),
        }

    except Exception as e:
        logger.exception("Error analyzing spatial patterns")
        return {"error": str(e)}


def _local_variance(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local variance using sliding window."""
    pad = window_size // 2
    padded = np.pad(img, pad, mode="reflect")
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i : i + window_size, j : j + window_size]
            result[i, j] = np.var(window)

    return result


def _interpret_spatial(edge_density: float, texture: float, entropy: float) -> str:
    """Generate brief interpretation of spatial patterns."""
    observations = []

    if edge_density > 0.1:
        observations.append("high edge content (sharp features)")
    elif edge_density < 0.03:
        observations.append("smooth/homogeneous regions")

    if texture > 0.05:
        observations.append("complex texture")
    elif texture < 0.01:
        observations.append("uniform texture")

    if entropy > 6:
        observations.append("high information density")
    elif entropy < 4:
        observations.append("low information density")

    return ", ".join(observations) if observations else "typical spatial characteristics"


def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool by name with given arguments.

    Args:
        tool_name: Name of tool to execute
        args: Tool arguments
        context: Execution context with artifacts, visualizations, etc.

    Returns:
        Tool result dictionary
    """
    visualizations = context.get("visualizations", {})

    if tool_name == "compute_image_similarity":
        return compute_image_similarity(
            visualizations,
            args.get("image1_key", ""),
            args.get("image2_key", ""),
        )
    elif tool_name == "compute_band_statistics":
        return compute_band_statistics(args.get("file_path", ""))
    elif tool_name == "analyze_spatial_patterns":
        return analyze_spatial_patterns(visualizations, args.get("image_key", ""))
    else:
        return {"error": f"Unknown tool: {tool_name}"}
