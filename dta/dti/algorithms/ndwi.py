"""NDWI (Normalized Difference Water Index) calculation.

NDWI is computed as (Green - NIR) / (Green + NIR) and ranges from -1 to 1.
Positive values typically indicate water bodies.

Reference:
- McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI)
  in the delineation of open water features.
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


def _create_ndwi_visualization(ndwi_array: np.ndarray, title: str = "NDWI") -> str | None:
    """Create a colored visualization of NDWI.

    Args:
        ndwi_array: NDWI array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # NDWI colormap: Brown (dry/vegetation) -> White -> Blue (water)
    colors = [
        (0.6, 0.4, 0.2),  # Brown - dry land/vegetation
        (0.8, 0.7, 0.5),  # Tan
        (0.95, 0.95, 0.95),  # Near white - transition
        (0.6, 0.8, 1.0),  # Light blue - shallow/mixed
        (0.2, 0.5, 0.9),  # Medium blue - water
        (0.0, 0.2, 0.6),  # Dark blue - deep water
    ]
    cmap = LinearSegmentedColormap.from_list("ndwi", colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(ndwi_array, cmap=cmap, vmin=-0.5, vmax=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("NDWI", fontsize=10)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels(["Dry", "", "Mixed", "", "Water"])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_water_mask_visualization(
    water_mask: np.ndarray,
    title: str = "Water Detection",
) -> str | None:
    """Create visualization of water mask.

    Args:
        water_mask: Binary water mask array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Simple two-color map: land (tan) and water (blue)
    colors = [(0.85, 0.75, 0.6), (0.1, 0.4, 0.8)]
    cmap = LinearSegmentedColormap.from_list("water_mask", colors, N=2)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(water_mask.astype(float), cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor="black", label="Land"),
        Patch(facecolor=colors[1], edgecolor="black", label="Water"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_ndwi(raster_path: str) -> dict[str, Any]:
    """Calculate NDWI from a multispectral raster.

    NDWI = (Green - NIR) / (Green + NIR)

    Positive values indicate water bodies. Typical thresholds:
    - NDWI > 0.3: Open water
    - NDWI > 0.0: Possible water/wetland
    - NDWI < 0.0: Non-water (vegetation, soil)

    Args:
        raster_path: Path to GeoTIFF with multispectral data

    Returns:
        Dictionary containing:
            - ndwi_array: Computed NDWI array
            - water_mask: Binary water detection (NDWI > 0.3)
            - metadata: Raster metadata
            - statistics: NDWI and water coverage stats
            - visualizations: NDWI map and water mask

    Raises:
        ValueError: If raster has insufficient bands
        FileNotFoundError: If raster file not found
    """
    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.count < 4:
            raise ValueError(f"NDWI requires at least 4 bands (with NIR), got {src.count}")

        # Band selection based on band count
        # Same logic as other algorithms for consistency
        if src.count >= 7:
            # Landsat 8/9: B3=Green, B5=NIR
            green_band = src.read(3, masked=True).astype(np.float32)
            nir_band = src.read(5, masked=True).astype(np.float32)
        elif src.count >= 5:
            # Sentinel-2 or similar: B2=Green, B4=NIR
            green_band = src.read(2, masked=True).astype(np.float32)
            nir_band = src.read(4, masked=True).astype(np.float32)
        else:
            # 4 bands (RGBN): B2=Green, B4=NIR
            green_band = src.read(2, masked=True).astype(np.float32)
            nir_band = src.read(4, masked=True).astype(np.float32)

        # Handle masked arrays
        if hasattr(green_band, "filled"):
            green_band = green_band.filled(np.nan)
        if hasattr(nir_band, "filled"):
            nir_band = nir_band.filled(np.nan)

        # Calculate NDWI: (Green - NIR) / (Green + NIR)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = green_band + nir_band
            ndwi = np.where(denominator != 0, (green_band - nir_band) / denominator, 0.0)

        # Create water mask (NDWI > 0.3 is typical threshold for open water)
        water_threshold = 0.3
        water_mask = ndwi > water_threshold

        # Compute statistics on valid pixels
        valid_mask = np.isfinite(ndwi)
        valid_ndwi = ndwi[valid_mask]

        stats: dict[str, Any] = {}
        if valid_ndwi.size > 0:
            water_pixels = int(np.sum(water_mask[valid_mask]))
            total_valid = int(valid_ndwi.size)

            stats = {
                "min": float(np.min(valid_ndwi)),
                "max": float(np.max(valid_ndwi)),
                "mean": float(np.mean(valid_ndwi)),
                "std": float(np.std(valid_ndwi)),
                "median": float(np.median(valid_ndwi)),
                "valid_pixels": total_valid,
                "total_pixels": int(ndwi.size),
                "water_pixels": water_pixels,
                "water_percentage": round(100 * water_pixels / total_valid, 2),
                "water_threshold": water_threshold,
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
            visualizations["ndwi_map"] = _create_ndwi_visualization(
                ndwi,
                title=f"NDWI - {Path(raster_path).stem}",
            )
            visualizations["water_mask"] = _create_water_mask_visualization(
                water_mask,
                title=f"Water Detection (NDWI > {water_threshold})",
            )

        return {
            "ndwi_array": ndwi.tolist(),
            "water_mask": water_mask.tolist(),
            "metadata": metadata,
            "statistics": stats,
            "visualizations": visualizations,
            "path": str(raster_path),
        }


def run(RasterPath: str) -> dict[str, Any]:
    """Registry-compatible NDWI calculation.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        NDWI result dictionary
    """
    return calculate_ndwi(RasterPath)
