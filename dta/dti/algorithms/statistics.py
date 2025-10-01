"""Statistical analysis for geospatial raster data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def calculate_statistics(raster_path: str) -> dict[str, Any]:
    """Calculate comprehensive statistics for a raster.

    Computes per-band and overall statistics including:
    - Basic stats (min, max, mean, std, median)
    - Percentiles (5th, 25th, 75th, 95th)
    - Histogram data
    - Nodata information

    Args:
        raster_path: Path to GeoTIFF

    Returns:
        Dictionary with statistics for each band and overall metadata

    Raises:
        FileNotFoundError: If raster file not found
    """
    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        stats: dict[str, Any] = {
            "path": str(raster_path),
            "metadata": {
                "crs": src.crs.to_string() if src.crs else None,
                "bounds": list(src.bounds),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else None,
                "nodata": src.nodata,
            },
            "bands": {},
        }

        # Per-band statistics
        for band_idx in range(1, src.count + 1):
            band_data = src.read(band_idx, masked=True)

            # Convert to float for calculations
            band_float = band_data.astype(float)
            if hasattr(band_float, "filled"):
                band_array = band_float.filled(np.nan)
            else:
                band_array = band_float

            # Valid pixels only
            valid_mask = np.isfinite(band_array)
            valid_data = band_array[valid_mask]

            band_stats: dict[str, Any] = {}
            if valid_data.size > 0:
                band_stats = {
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "median": float(np.median(valid_data)),
                    "percentiles": {
                        "p5": float(np.percentile(valid_data, 5)),
                        "p25": float(np.percentile(valid_data, 25)),
                        "p75": float(np.percentile(valid_data, 75)),
                        "p95": float(np.percentile(valid_data, 95)),
                    },
                    "valid_pixels": int(valid_data.size),
                    "total_pixels": int(band_array.size),
                    "nodata_pixels": int(band_array.size - valid_data.size),
                }

                # Histogram (10 bins)
                hist, bin_edges = np.histogram(valid_data, bins=10)
                band_stats["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                band_stats = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "median": None,
                    "valid_pixels": 0,
                    "total_pixels": int(band_array.size),
                    "nodata_pixels": int(band_array.size),
                }

            stats["bands"][f"band_{band_idx}"] = band_stats

        # Overall statistics (if multi-band, average across bands)
        if src.count > 1:
            all_means = [
                stats["bands"][f"band_{i}"]["mean"]
                for i in range(1, src.count + 1)
                if stats["bands"][f"band_{i}"]["mean"] is not None
            ]
            if all_means:
                stats["overall_mean"] = float(np.mean(all_means))

        return stats


# Registry-compatible wrapper
def run(RasterPath: str) -> dict[str, Any]:
    """Registry-compatible statistics calculation.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        Statistics result dictionary
    """
    return calculate_statistics(RasterPath)
