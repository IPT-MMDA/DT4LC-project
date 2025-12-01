"""Change Detection Algorithm - NDVI-based vegetation change analysis.

Compares two raster images from different time periods to detect
vegetation changes using NDVI differencing.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# Try to import matplotlib for visualization, graceful fallback
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _calculate_ndvi_array(src: rasterio.DatasetReader) -> np.ndarray:
    """Calculate NDVI array from rasterio dataset.

    Args:
        src: Open rasterio dataset

    Returns:
        NDVI array with values in [-1, 1] range
    """
    if src.count < 2:
        raise ValueError(f"NDVI requires at least 2 bands, got {src.count}")

    # HLS/Landsat-like: Red=Band 4, NIR=Band 5 (for 5+ bands)
    # Otherwise: Red=Band 1, NIR=Band 2
    if src.count >= 4:
        red_band = src.read(4, masked=True).astype(np.float32)
        nir_band = src.read(5, masked=True).astype(np.float32)
    else:
        red_band = src.read(1, masked=True).astype(np.float32)
        nir_band = src.read(2, masked=True).astype(np.float32)

    # NDVI = (NIR - Red) / (NIR + Red)
    denominator = nir_band + red_band
    ndvi = np.where(
        denominator != 0,
        (nir_band - red_band) / denominator,
        np.nan,
    )

    # Handle masked arrays
    if hasattr(ndvi, "filled"):
        ndvi = ndvi.filled(np.nan)

    return ndvi


def _create_change_visualization(
    change_array: np.ndarray,
    title: str = "Vegetation Change",
) -> str | None:
    """Create a colored visualization of NDVI change.

    Args:
        change_array: NDVI difference array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Create custom colormap: Red (loss) -> White (stable) -> Green (gain)
    colors = [
        (0.8, 0.0, 0.0),  # Dark red - severe loss
        (1.0, 0.4, 0.4),  # Light red - moderate loss
        (1.0, 1.0, 1.0),  # White - stable
        (0.4, 0.8, 0.4),  # Light green - moderate gain
        (0.0, 0.6, 0.0),  # Dark green - strong gain
    ]
    cmap = LinearSegmentedColormap.from_list("vegetation_change", colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Clip change values for better visualization (-0.5 to 0.5 range)
    vmin, vmax = -0.5, 0.5
    im = ax.imshow(change_array, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("NDVI Change", fontsize=10)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels(["Loss", "", "Stable", "", "Gain"])

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_ndvi_visualization(
    ndvi_array: np.ndarray,
    title: str = "NDVI",
) -> str | None:
    """Create a colored visualization of NDVI.

    Args:
        ndvi_array: NDVI array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # NDVI colormap: Brown/Red (low) -> Yellow -> Green (high)
    colors = [
        (0.6, 0.3, 0.1),  # Brown - bare soil/water
        (0.8, 0.6, 0.2),  # Tan - sparse vegetation
        (1.0, 1.0, 0.4),  # Yellow - moderate vegetation
        (0.6, 0.8, 0.2),  # Yellow-green
        (0.2, 0.6, 0.2),  # Green - healthy vegetation
        (0.0, 0.4, 0.0),  # Dark green - dense vegetation
    ]
    cmap = LinearSegmentedColormap.from_list("ndvi", colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(ndvi_array, cmap=cmap, vmin=-0.2, vmax=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("NDVI", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_change(
    before_path: str,
    after_path: str,
) -> dict[str, Any]:
    """Calculate vegetation change between two raster images.

    Computes NDVI for both images and calculates the difference
    (after - before). Positive values indicate vegetation gain,
    negative values indicate vegetation loss.

    Args:
        before_path: Path to the earlier date GeoTIFF
        after_path: Path to the later date GeoTIFF

    Returns:
        Dictionary containing:
            - change_array: NDVI difference (after - before) as list
            - ndvi_before: NDVI array for before image
            - ndvi_after: NDVI array for after image
            - statistics: Change statistics
            - classification: Pixel classification counts
            - metadata: Raster metadata
            - visualizations: Base64 PNG images (if matplotlib available)

    Raises:
        FileNotFoundError: If either raster file not found
        ValueError: If rasters have different dimensions
    """
    before_path_obj = Path(before_path)
    after_path_obj = Path(after_path)

    if not before_path_obj.exists():
        raise FileNotFoundError(f"Before raster not found: {before_path}")
    if not after_path_obj.exists():
        raise FileNotFoundError(f"After raster not found: {after_path}")

    # Open both rasters
    with rasterio.open(before_path) as src_before, rasterio.open(after_path) as src_after:
        # Validate dimensions match
        if (src_before.width, src_before.height) != (src_after.width, src_after.height):
            raise ValueError(
                f"Raster dimensions must match. "
                f"Before: {src_before.width}x{src_before.height}, "
                f"After: {src_after.width}x{src_after.height}"
            )

        # Calculate NDVI for both
        ndvi_before = _calculate_ndvi_array(src_before)
        ndvi_after = _calculate_ndvi_array(src_after)

        # Calculate change (after - before)
        change = ndvi_after - ndvi_before

        # Statistics on valid pixels
        valid_mask = np.isfinite(change)
        valid_change = change[valid_mask]

        statistics = {}
        classification = {}

        if valid_change.size > 0:
            statistics = {
                "min_change": float(np.nanmin(valid_change)),
                "max_change": float(np.nanmax(valid_change)),
                "mean_change": float(np.nanmean(valid_change)),
                "std_change": float(np.nanstd(valid_change)),
                "median_change": float(np.nanmedian(valid_change)),
            }

            # Classification with multiple thresholds
            severe_loss = np.sum(valid_change < -0.2)
            moderate_loss = np.sum((valid_change >= -0.2) & (valid_change < -0.05))
            stable = np.sum((valid_change >= -0.05) & (valid_change <= 0.05))
            moderate_gain = np.sum((valid_change > 0.05) & (valid_change <= 0.2))
            strong_gain = np.sum(valid_change > 0.2)

            total = valid_change.size
            classification = {
                "severe_vegetation_loss": {
                    "pixels": int(severe_loss),
                    "percentage": round(100 * severe_loss / total, 2),
                },
                "moderate_vegetation_loss": {
                    "pixels": int(moderate_loss),
                    "percentage": round(100 * moderate_loss / total, 2),
                },
                "stable": {
                    "pixels": int(stable),
                    "percentage": round(100 * stable / total, 2),
                },
                "moderate_vegetation_gain": {
                    "pixels": int(moderate_gain),
                    "percentage": round(100 * moderate_gain / total, 2),
                },
                "strong_vegetation_gain": {
                    "pixels": int(strong_gain),
                    "percentage": round(100 * strong_gain / total, 2),
                },
                "total_valid_pixels": int(total),
            }

        # NDVI statistics for each image
        valid_before = ndvi_before[np.isfinite(ndvi_before)]
        valid_after = ndvi_after[np.isfinite(ndvi_after)]

        ndvi_stats = {
            "before": {
                "mean": float(np.nanmean(valid_before)) if valid_before.size > 0 else None,
                "std": float(np.nanstd(valid_before)) if valid_before.size > 0 else None,
                "min": float(np.nanmin(valid_before)) if valid_before.size > 0 else None,
                "max": float(np.nanmax(valid_before)) if valid_before.size > 0 else None,
            },
            "after": {
                "mean": float(np.nanmean(valid_after)) if valid_after.size > 0 else None,
                "std": float(np.nanstd(valid_after)) if valid_after.size > 0 else None,
                "min": float(np.nanmin(valid_after)) if valid_after.size > 0 else None,
                "max": float(np.nanmax(valid_after)) if valid_after.size > 0 else None,
            },
        }

        metadata = {
            "crs": src_after.crs.to_string() if src_after.crs else None,
            "transform": list(src_after.transform) if src_after.transform else None,
            "bounds": [
                src_after.bounds.left,
                src_after.bounds.bottom,
                src_after.bounds.right,
                src_after.bounds.top,
            ],
            "width": src_after.width,
            "height": src_after.height,
            "before_path": str(before_path),
            "after_path": str(after_path),
        }

        # Generate visualizations
        visualizations = {}
        if HAS_MATPLOTLIB:
            visualizations["change_map"] = _create_change_visualization(
                change,
                title="Vegetation Change Detection",
            )
            visualizations["ndvi_before"] = _create_ndvi_visualization(
                ndvi_before,
                title="NDVI - Before",
            )
            visualizations["ndvi_after"] = _create_ndvi_visualization(
                ndvi_after,
                title="NDVI - After",
            )

        return {
            "change_array": change.tolist(),
            "ndvi_before": ndvi_before.tolist(),
            "ndvi_after": ndvi_after.tolist(),
            "statistics": statistics,
            "ndvi_statistics": ndvi_stats,
            "classification": classification,
            "metadata": metadata,
            "visualizations": visualizations,
        }


def run(RasterPathBefore: str, RasterPathAfter: str) -> dict[str, Any]:
    """Registry-compatible change detection.

    Args:
        RasterPathBefore: Path to before image (registry type)
        RasterPathAfter: Path to after image (registry type)

    Returns:
        Change detection result dictionary
    """
    return calculate_change(RasterPathBefore, RasterPathAfter)
