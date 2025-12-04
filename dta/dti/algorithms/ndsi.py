"""NDSI (Normalized Difference Snow Index) calculation.

NDSI is computed as (Green - SWIR) / (Green + SWIR) and ranges from -1 to 1.
Values above 0.42 typically indicate snow/ice coverage.

For Sentinel-2: NDSI = (B3 - B11) / (B3 + B11)
For Landsat 8/9: NDSI = (B3 - B6) / (B3 + B6)

References:
- https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndsi/
- https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/snow_classifier/
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# Try to import matplotlib for visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Standard snow threshold
SNOW_THRESHOLD = 0.42


def _create_ndsi_visualization(ndsi_array: np.ndarray, title: str = "NDSI") -> str | None:
    """Create a colored visualization of NDSI.

    Args:
        ndsi_array: NDSI array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # NDSI colormap: Brown/Gray (no snow) -> Light Blue -> White (snow/ice)
    colors = [
        (0.4, 0.3, 0.2),  # Brown - bare ground
        (0.5, 0.5, 0.5),  # Gray - rock/soil
        (0.6, 0.7, 0.8),  # Light gray-blue - transition
        (0.7, 0.85, 0.95),  # Light blue - some snow
        (0.85, 0.92, 1.0),  # Very light blue - snow
        (1.0, 1.0, 1.0),  # White - dense snow/ice
    ]
    cmap = LinearSegmentedColormap.from_list("ndsi", colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(ndsi_array, cmap=cmap, vmin=-0.5, vmax=1.0)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("NDSI", fontsize=10)

    # Add snow threshold line on colorbar
    cbar.ax.axhline(y=SNOW_THRESHOLD, color="red", linewidth=1.5, linestyle="--")
    cbar.ax.text(1.5, SNOW_THRESHOLD, f"Snow threshold ({SNOW_THRESHOLD})", va="center", fontsize=8, color="red")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_snow_mask_visualization(ndsi_array: np.ndarray, title: str = "Snow Classification") -> str | None:
    """Create a binary snow/no-snow visualization.

    Args:
        ndsi_array: NDSI array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Create binary snow mask
    snow_mask = ndsi_array >= SNOW_THRESHOLD

    # Custom colormap: brown (no snow) -> white (snow)
    colors = [(0.5, 0.4, 0.3), (1.0, 1.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list("snow", colors, N=2)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(snow_mask.astype(float), cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=(0.5, 0.4, 0.3), label="No Snow"),
        Patch(facecolor=(1.0, 1.0, 1.0), edgecolor="black", label="Snow/Ice"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_ndsi(raster_path: str) -> dict[str, Any]:
    """Calculate NDSI from a multispectral raster.

    Expects a raster with Green and SWIR bands. Uses standard band ordering
    based on sensor type detection.

    For Sentinel-2: Green=B3, SWIR=B11
    For Landsat 8/9: Green=B3, SWIR=B6

    Args:
        raster_path: Path to GeoTIFF with multispectral data

    Returns:
        Dictionary containing:
            - ndsi_array: Computed NDSI array
            - snow_mask: Binary snow classification (NDSI >= 0.42)
            - metadata: Raster metadata (CRS, transform, etc.)
            - statistics: Basic stats (min, max, mean, std, snow coverage)
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
            raise ValueError(f"NDSI requires at least 2 bands, got {src.count}")

        # Band selection based on band count (heuristic for different sensors)
        # Sentinel-2 (13 bands): Green=B3 (band 3), SWIR1=B11 (band 11)
        # Landsat 8/9 (7+ bands): Green=B3, SWIR1=B6
        # 6 bands (Sentinel-2 subset B2,B3,B4,B8,B11,B12): Green=2, SWIR=5
        # Generic: assume Green=2, SWIR=last band
        if src.count >= 11:
            # Full Sentinel-2: Green=Band3, SWIR1=Band11
            green_band = src.read(3, masked=True).astype(float)
            swir_band = src.read(11, masked=True).astype(float)
        elif src.count >= 7:
            # Landsat 8/9: Green=Band3, SWIR1=Band6
            green_band = src.read(3, masked=True).astype(float)
            swir_band = src.read(6, masked=True).astype(float)
        elif src.count == 6:
            # Sentinel-2 subset (B2,B3,B4,B8,B11,B12): Green=Band2, SWIR=Band5
            green_band = src.read(2, masked=True).astype(float)
            swir_band = src.read(5, masked=True).astype(float)
        elif src.count >= 4:
            # 4-5 bands: assume Green=Band2, SWIR=last band
            green_band = src.read(2, masked=True).astype(float)
            swir_band = src.read(src.count, masked=True).astype(float)
        else:
            # 2-3 bands: assume Green=Band1, SWIR=Band2
            green_band = src.read(1, masked=True).astype(float)
            swir_band = src.read(2, masked=True).astype(float)

        # Calculate NDSI: (Green - SWIR) / (Green + SWIR)
        denominator = green_band + swir_band
        ndsi = np.where(
            denominator != 0,
            (green_band - swir_band) / denominator,
            0.0,  # Avoid division by zero
        )

        # Mask out invalid values
        if hasattr(ndsi, "filled"):
            ndsi_filled = ndsi.filled(np.nan)
        else:
            ndsi_filled = ndsi

        # Create snow mask (NDSI >= 0.42)
        snow_mask = ndsi_filled >= SNOW_THRESHOLD

        # Compute statistics on valid pixels
        valid_mask = np.isfinite(ndsi_filled)
        valid_ndsi = ndsi_filled[valid_mask]

        stats = {}
        if valid_ndsi.size > 0:
            snow_pixels = int(np.sum(snow_mask[valid_mask]))
            total_valid = int(valid_ndsi.size)

            stats = {
                "min": float(np.min(valid_ndsi)),
                "max": float(np.max(valid_ndsi)),
                "mean": float(np.mean(valid_ndsi)),
                "std": float(np.std(valid_ndsi)),
                "median": float(np.median(valid_ndsi)),
                "valid_pixels": total_valid,
                "total_pixels": int(ndsi_filled.size),
                # Snow-specific statistics
                "snow_threshold": SNOW_THRESHOLD,
                "snow_pixels": snow_pixels,
                "non_snow_pixels": total_valid - snow_pixels,
                "snow_coverage_percent": float(snow_pixels / total_valid * 100) if total_valid > 0 else 0.0,
            }

        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }

        # Generate visualizations
        visualizations = {}
        if HAS_MATPLOTLIB:
            visualizations["ndsi_map"] = _create_ndsi_visualization(
                ndsi_filled,
                title=f"NDSI - {Path(raster_path).stem}",
            )
            visualizations["snow_classification"] = _create_snow_mask_visualization(
                ndsi_filled,
                title=f"Snow Classification - {Path(raster_path).stem}",
            )

        return {
            "ndsi_array": ndsi_filled.tolist(),  # Convert to list for JSON serialization
            "snow_mask": snow_mask.tolist(),  # Binary snow classification
            "metadata": metadata,
            "statistics": stats,
            "visualizations": visualizations,
            "path": str(raster_path),
        }


# Convenience function for registry integration
def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    """Registry-compatible NDSI calculation.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        NDSI result dictionary
    """
    return calculate_ndsi(RasterPath)
