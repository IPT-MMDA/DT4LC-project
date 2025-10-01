"""NDVI (Normalized Difference Vegetation Index) calculation.

NDVI is computed as (NIR - Red) / (NIR + Red) and ranges from -1 to 1.
Higher values indicate healthier vegetation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def calculate_ndvi(raster_path: str) -> dict[str, Any]:
    """Calculate NDVI from a multispectral raster.

    Expects a raster with at least NIR and Red bands. Uses standard band
    ordering: Red=Band 1 (or 4), NIR=Band 2 (or 5) depending on sensor.

    Args:
        raster_path: Path to GeoTIFF with multispectral data

    Returns:
        Dictionary containing:
            - ndvi_array: Computed NDVI array
            - metadata: Raster metadata (CRS, transform, etc.)
            - statistics: Basic stats (min, max, mean, std)
            - path: Input path

    Raises:
        ValueError: If raster has insufficient bands
        FileNotFoundError: If raster file not found
    """
    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.count < 2:
            raise ValueError(f"NDVI requires at least 2 bands, got {src.count}")

        # Heuristic: if 4+ bands, assume Landsat-like (R=4, NIR=5)
        # Otherwise assume R=1, NIR=2
        if src.count >= 4:
            red_band = src.read(4, masked=True).astype(float)
            nir_band = src.read(5, masked=True).astype(float)
        else:
            red_band = src.read(1, masked=True).astype(float)
            nir_band = src.read(2, masked=True).astype(float)

        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        denominator = nir_band + red_band
        ndvi = np.where(
            denominator != 0,
            (nir_band - red_band) / denominator,
            0.0,  # Avoid division by zero
        )

        # Mask out invalid values
        if hasattr(ndvi, "filled"):
            ndvi_filled = ndvi.filled(np.nan)
        else:
            ndvi_filled = ndvi

        # Compute statistics on valid pixels
        valid_mask = np.isfinite(ndvi_filled)
        valid_ndvi = ndvi_filled[valid_mask]

        stats = {}
        if valid_ndvi.size > 0:
            stats = {
                "min": float(np.min(valid_ndvi)),
                "max": float(np.max(valid_ndvi)),
                "mean": float(np.mean(valid_ndvi)),
                "std": float(np.std(valid_ndvi)),
                "median": float(np.median(valid_ndvi)),
                "valid_pixels": int(valid_ndvi.size),
                "total_pixels": int(ndvi_filled.size),
            }

        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }

        return {
            "ndvi_array": ndvi_filled,
            "metadata": metadata,
            "statistics": stats,
            "path": str(raster_path),
        }


def ndvi_change(
    ndvi_before: dict[str, Any],
    ndvi_after: dict[str, Any],
) -> dict[str, Any]:
    """Calculate NDVI change between two time periods.

    Args:
        ndvi_before: NDVI result from earlier date
        ndvi_after: NDVI result from later date

    Returns:
        Dictionary containing:
            - change_array: NDVI difference (after - before)
            - statistics: Change statistics
            - metadata: Combined metadata

    Raises:
        ValueError: If arrays have different shapes
    """
    arr_before = ndvi_before["ndvi_array"]
    arr_after = ndvi_after["ndvi_array"]

    if arr_before.shape != arr_after.shape:
        raise ValueError(f"NDVI arrays must have same shape. Got {arr_before.shape} and {arr_after.shape}")

    # Calculate change
    change = arr_after - arr_before

    # Statistics on valid pixels
    valid_mask = np.isfinite(change)
    valid_change = change[valid_mask]

    stats = {}
    if valid_change.size > 0:
        stats = {
            "min_change": float(np.min(valid_change)),
            "max_change": float(np.max(valid_change)),
            "mean_change": float(np.mean(valid_change)),
            "std_change": float(np.std(valid_change)),
            "median_change": float(np.median(valid_change)),
            # Thresholds for interpretation
            "increase_pixels": int(np.sum(valid_change > 0.1)),  # Significant increase
            "decrease_pixels": int(np.sum(valid_change < -0.1)),  # Significant decrease
            "stable_pixels": int(np.sum(np.abs(valid_change) <= 0.1)),  # Stable
            "total_valid_pixels": int(valid_change.size),
        }

    return {
        "change_array": change,
        "statistics": stats,
        "metadata": ndvi_after["metadata"],  # Use later date metadata
        "before_path": ndvi_before.get("path"),
        "after_path": ndvi_after.get("path"),
    }


# Convenience function for registry integration
def run(RasterPath: str) -> dict[str, Any]:
    """Registry-compatible NDVI calculation.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        NDVI result dictionary
    """
    return calculate_ndvi(RasterPath)
