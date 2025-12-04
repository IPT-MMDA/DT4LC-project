"""Snow Classifier - Multi-criteria snow detection algorithm.

A more robust snow classification compared to basic NDSI thresholding.
Combines multiple spectral indices and brightness thresholds to reduce
false positives and improve accuracy across varying terrain and atmospheric conditions.

Classification rules:
1. NDSI >= 0.4 (snow absorbs SWIR, reflects visible)
2. NDVI close to 0.1 (±0.025) - vegetation check
3. Green band brightness > 0.3 - brightness confirmation

For Sentinel-2:
- NDSI = (B3 - B11) / (B3 + B11)
- NDVI = (B8 - B4) / (B8 + B4)
- Brightness = B3

References:
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

# Classification thresholds
NDSI_THRESHOLD = 0.4
NDVI_CENTER = 0.1
NDVI_TOLERANCE = 0.025
BRIGHTNESS_THRESHOLD = 0.3


def _create_snow_classification_visualization(
    snow_mask: np.ndarray,
    title: str = "Snow Classification",
) -> str | None:
    """Create visualization of snow classification result.

    Args:
        snow_mask: Binary snow classification array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Custom colormap: terrain color (no snow) -> white/orange (snow)
    colors = [(0.4, 0.5, 0.3), (1.0, 0.8, 0.4)]  # Green-brown to orange-yellow (snow)
    cmap = LinearSegmentedColormap.from_list("snow_class", colors, N=2)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(snow_mask.astype(float), cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=(0.4, 0.5, 0.3), label="No Snow"),
        Patch(facecolor=(1.0, 0.8, 0.4), edgecolor="black", label="Snow"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_criteria_visualization(
    ndsi: np.ndarray,
    ndvi: np.ndarray,
    brightness: np.ndarray,
    snow_mask: np.ndarray,
    title: str = "Snow Classification Criteria",
) -> str | None:
    """Create multi-panel visualization showing all classification criteria.

    Args:
        ndsi: NDSI values
        ndvi: NDVI values
        brightness: Green band brightness
        snow_mask: Final snow classification
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # NDSI
    im1 = axes[0, 0].imshow(ndsi, cmap="RdYlBu", vmin=-0.5, vmax=1.0)
    axes[0, 0].set_title(f"NDSI (threshold ≥ {NDSI_THRESHOLD})")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # NDVI
    im2 = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    axes[0, 1].set_title(f"NDVI (target ≈ {NDVI_CENTER})")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # Brightness
    im3 = axes[1, 0].imshow(brightness, cmap="gray", vmin=0, vmax=1.0)
    axes[1, 0].set_title(f"Green Brightness (threshold > {BRIGHTNESS_THRESHOLD})")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    # Final classification
    colors = [(0.4, 0.5, 0.3), (1.0, 0.8, 0.4)]
    cmap = LinearSegmentedColormap.from_list("snow", colors, N=2)
    axes[1, 1].imshow(snow_mask.astype(float), cmap=cmap, vmin=0, vmax=1)
    axes[1, 1].set_title("Snow Classification Result")
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def classify_snow(raster_path: str) -> dict[str, Any]:
    """Multi-criteria snow classification from multispectral imagery.

    Uses three criteria for robust snow detection:
    1. NDSI >= 0.4 (primary snow indicator)
    2. NDVI close to 0.1 (±0.025) - excludes vegetation
    3. Green brightness > 0.3 - confirms high reflectance

    A pixel is classified as snow if:
    - (NDSI >= 0.4 AND brightness > 0.3) OR
    - (|NDVI - 0.1| <= 0.025 AND brightness > 0.3)

    Args:
        raster_path: Path to GeoTIFF with multispectral data

    Returns:
        Dictionary containing:
            - snow_mask: Binary snow classification
            - ndsi: NDSI values
            - ndvi: NDVI values
            - brightness: Green band brightness
            - metadata: Raster metadata
            - statistics: Classification statistics
            - path: Input path

    Raises:
        ValueError: If raster has insufficient bands
        FileNotFoundError: If raster file not found
    """
    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.count < 4:
            raise ValueError(f"Snow classifier requires at least 4 bands, got {src.count}")

        # Band selection based on band count
        # Sentinel-2 (13 bands): Green=B3, Red=B4, NIR=B8, SWIR=B11
        # Landsat 8/9 (7+ bands): Green=B3, Red=B4, NIR=B5, SWIR=B6
        # 6 bands (Sentinel-2 subset): Green=B2, Red=B3, NIR=B4, SWIR=B5
        if src.count >= 11:
            # Full Sentinel-2
            green = src.read(3, masked=True).astype(float)
            red = src.read(4, masked=True).astype(float)
            nir = src.read(8, masked=True).astype(float)
            swir = src.read(11, masked=True).astype(float)
        elif src.count >= 7:
            # Landsat 8/9
            green = src.read(3, masked=True).astype(float)
            red = src.read(4, masked=True).astype(float)
            nir = src.read(5, masked=True).astype(float)
            swir = src.read(6, masked=True).astype(float)
        elif src.count >= 6:
            # Sentinel-2 subset (B2,B3,B4,B8,B11,B12)
            green = src.read(2, masked=True).astype(float)
            red = src.read(3, masked=True).astype(float)
            nir = src.read(4, masked=True).astype(float)
            swir = src.read(5, masked=True).astype(float)
        else:
            # 4-5 bands: assume Green=2, Red=3, NIR=4, SWIR=last
            green = src.read(2, masked=True).astype(float)
            red = src.read(3, masked=True).astype(float)
            nir = src.read(4, masked=True).astype(float)
            swir = src.read(src.count, masked=True).astype(float)

        # Normalize to 0-1 range if needed (detect by max value)
        max_val = max(
            np.nanmax(green) if np.any(np.isfinite(green)) else 0,
            np.nanmax(red) if np.any(np.isfinite(red)) else 0,
            np.nanmax(nir) if np.any(np.isfinite(nir)) else 0,
            np.nanmax(swir) if np.any(np.isfinite(swir)) else 0,
        )

        if max_val > 100:  # Likely in DN or scaled reflectance (0-10000)
            scale_factor = 10000.0 if max_val > 1000 else 255.0
            green = green / scale_factor
            red = red / scale_factor
            nir = nir / scale_factor
            swir = swir / scale_factor

        # Calculate NDSI: (Green - SWIR) / (Green + SWIR)
        ndsi_denom = green + swir
        ndsi = np.where(ndsi_denom != 0, (green - swir) / ndsi_denom, 0.0)

        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        ndvi_denom = nir + red
        ndvi = np.where(ndvi_denom != 0, (nir - red) / ndvi_denom, 0.0)

        # Brightness from green band
        brightness = green

        # Fill masked arrays
        if hasattr(ndsi, "filled"):
            ndsi = ndsi.filled(np.nan)
        if hasattr(ndvi, "filled"):
            ndvi = ndvi.filled(np.nan)
        if hasattr(brightness, "filled"):
            brightness = brightness.filled(np.nan)

        # Classification logic:
        # Snow if: (NDSI >= 0.4 AND brightness > 0.3) OR
        #          (|NDVI - 0.1| <= 0.025 AND brightness > 0.3)
        condition_ndsi = ndsi >= NDSI_THRESHOLD
        condition_ndvi = np.abs(ndvi - NDVI_CENTER) <= NDVI_TOLERANCE
        condition_brightness = brightness > BRIGHTNESS_THRESHOLD

        snow_mask = (condition_ndsi | condition_ndvi) & condition_brightness

        # Handle NaN values
        valid_mask = np.isfinite(ndsi) & np.isfinite(ndvi) & np.isfinite(brightness)
        snow_mask = snow_mask & valid_mask

        # Compute statistics
        total_valid = int(np.sum(valid_mask))
        snow_pixels = int(np.sum(snow_mask))

        stats = {
            "total_pixels": int(ndsi.size),
            "valid_pixels": total_valid,
            "snow_pixels": snow_pixels,
            "non_snow_pixels": total_valid - snow_pixels,
            "snow_coverage_percent": float(snow_pixels / total_valid * 100) if total_valid > 0 else 0.0,
            # Criteria statistics
            "pixels_meeting_ndsi_threshold": int(np.sum(condition_ndsi & valid_mask)),
            "pixels_meeting_ndvi_criterion": int(np.sum(condition_ndvi & valid_mask)),
            "pixels_meeting_brightness_threshold": int(np.sum(condition_brightness & valid_mask)),
            # Thresholds used
            "thresholds": {
                "ndsi": NDSI_THRESHOLD,
                "ndvi_center": NDVI_CENTER,
                "ndvi_tolerance": NDVI_TOLERANCE,
                "brightness": BRIGHTNESS_THRESHOLD,
            },
            # Index statistics
            "ndsi_stats": {
                "min": float(np.nanmin(ndsi)) if np.any(np.isfinite(ndsi)) else None,
                "max": float(np.nanmax(ndsi)) if np.any(np.isfinite(ndsi)) else None,
                "mean": float(np.nanmean(ndsi)) if np.any(np.isfinite(ndsi)) else None,
            },
            "ndvi_stats": {
                "min": float(np.nanmin(ndvi)) if np.any(np.isfinite(ndvi)) else None,
                "max": float(np.nanmax(ndvi)) if np.any(np.isfinite(ndvi)) else None,
                "mean": float(np.nanmean(ndvi)) if np.any(np.isfinite(ndvi)) else None,
            },
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
            visualizations["snow_classification"] = _create_snow_classification_visualization(
                snow_mask,
                title=f"Snow Classification - {Path(raster_path).stem}",
            )
            visualizations["criteria_analysis"] = _create_criteria_visualization(
                ndsi,
                ndvi,
                brightness,
                snow_mask,
                title=f"Classification Criteria - {Path(raster_path).stem}",
            )

        return {
            "snow_mask": snow_mask.tolist(),
            "ndsi": ndsi.tolist(),
            "ndvi": ndvi.tolist(),
            "brightness": brightness.tolist(),
            "metadata": metadata,
            "statistics": stats,
            "visualizations": visualizations,
            "path": str(raster_path),
        }


# Convenience function for registry integration
def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    """Registry-compatible snow classification.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        Snow classification result dictionary
    """
    return classify_snow(RasterPath)
