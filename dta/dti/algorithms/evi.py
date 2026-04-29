"""EVI (Enhanced Vegetation Index) calculation.

EVI is computed as 2.5 * (NIR - RED) / ((NIR + 6*RED - 7.5*BLUE) + 1) and ranges from -1 to 1.
Higher values indicate healthier vegetation.
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


def _create_evi_visualization(evi_array: np.ndarray, title: str = "EVI") -> str | None:
    """Create a colored visualization of EVI.

    Args:
        evi_array: EVI array
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # EVI colormap: Brown/Red (low) -> Yellow -> Green (high)
    colors = [
        (0.6, 0.3, 0.1),  # Brown - bare soil/water
        (0.8, 0.6, 0.2),  # Tan - sparse vegetation
        (1.0, 1.0, 0.4),  # Yellow - moderate vegetation
        (0.6, 0.8, 0.2),  # Yellow-green
        (0.2, 0.6, 0.2),  # Green - healthy vegetation
        (0.0, 0.4, 0.0),  # Dark green - dense vegetation
    ]
    cmap = LinearSegmentedColormap.from_list("evi", colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(evi_array, cmap=cmap, vmin=-0.2, vmax=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("EVI", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_evi(raster_path: str) -> dict[str, Any]:
    """Calculate EVI from a multispectral raster.

    Expects a raster with at least NIR and Red bands. Uses standard band
    ordering: Red=Band 1 (or 4), Blue=Band 2 (or 3), NIR=Band 2 (or 5) depending on sensor.

    Args:
        raster_path: Path to GeoTIFF with multispectral data

    Returns:
        Dictionary containing:
            - evi_array: Computed EVI array
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
        if src.count < 4:
            raise ValueError(f"EVI requires at least 4 bands (with NIR), got {src.count}")

        # Band selection based on band count (heuristic for different sensors)
        # 7+ bands: Landsat 8/9 (B1-B7 + QA) -> Blue=2, Red=4, NIR=5
        # 6 bands: Sentinel-2 subset (B2,B3,B4,B8,B11,B12) -> Red=3, NIR=4
        # 5 bands: Generic (B,G,R,NIR,SWIR) -> Red=3, NIR=4
        # 4 bands: RGBN -> Red=1, Blue=3, NIR=4
        # 2-3 bands: Simple R,NIR or R,G,NIR -> Red=1, NIR=2
        if src.count >= 7:
            # Landsat 8/9: Blue=Band2 (SR_B2), Red=Band4 (SR_B4), NIR=Band5 (SR_B5)
            red_band = src.read(4, masked=True).astype(float)
            blue_band = src.read(2, masked=True).astype(float)
            nir_band = src.read(5, masked=True).astype(float)
        elif src.count >= 5:
            # Sentinel-2 or similar: Blue=Band2, Red=Band3, NIR=Band4
            red_band = src.read(3, masked=True).astype(float)
            blue_band = src.read(2, masked=True).astype(float)
            nir_band = src.read(4, masked=True).astype(float)
        else:
            # RGBN format: Red=Band1, Blue=Band3, NIR=Band4
            red_band = src.read(1, masked=True).astype(float)
            blue_band = src.read(3, masked=True).astype(float)
            nir_band = src.read(4, masked=True).astype(float)

        # Calculate EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
        evi = np.where(
            denominator != 0,
            (2.5 * (nir_band - red_band)) / denominator,
            0.0,  # Avoid division by zero
        )

        # Mask out invalid values
        if hasattr(evi, "filled"):
            evi_filled = evi.filled(np.nan)
        else:
            evi_filled = evi

        # Compute statistics on valid pixels
        valid_mask = np.isfinite(evi_filled)
        valid_evi = evi_filled[valid_mask]

        stats = {}
        if valid_evi.size > 0:
            stats = {
                "min": float(np.min(valid_evi)),
                "max": float(np.max(valid_evi)),
                "mean": float(np.mean(valid_evi)),
                "std": float(np.std(valid_evi)),
                "median": float(np.median(valid_evi)),
                "valid_pixels": int(valid_evi.size),
                "total_pixels": int(evi_filled.size),
            }

        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }

        # Generate visualization
        visualizations = {}
        if HAS_MATPLOTLIB:
            visualizations["evi_map"] = _create_evi_visualization(
                evi_filled,
                title=f"EVI - {Path(raster_path).stem}",
            )

        return {
            "evi_array": evi_filled.tolist(),  # Convert to list for JSON serialization
            "metadata": metadata,
            "statistics": stats,
            "visualizations": visualizations,
            "path": str(raster_path),
        }


def evi_change(
    evi_before: dict[str, Any],
    evi_after: dict[str, Any],
) -> dict[str, Any]:
    """Calculate EVI change between two time periods.

    Args:
        evi_before: EVI result from earlier date
        evi_after: EVI result from later date

    Returns:
        Dictionary containing:
            - change_array: EVI difference (after - before)
            - statistics: Change statistics
            - metadata: Combined metadata

    Raises:
        ValueError: If arrays have different shapes
    """
    arr_before = np.array(evi_before["evi_array"])
    arr_after = np.array(evi_after["evi_array"])

    if arr_before.shape != arr_after.shape:
        raise ValueError(f"EVI arrays must have same shape. Got {arr_before.shape} and {arr_after.shape}")

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
        "metadata": evi_after["metadata"],  # Use later date metadata
        "before_path": evi_before.get("path"),
        "after_path": evi_after.get("path"),
    }


# Convenience function for registry integration
def run(RasterPath: str) -> dict[str, Any]:
    """Registry-compatible EVI calculation.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        EVI result dictionary
    """
    return calculate_evi(RasterPath)
