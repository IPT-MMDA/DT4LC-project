"""Change Detection Algorithm - Multi-index temporal change analysis.

Compares two raster images from different time periods to detect
changes using spectral indices (NDVI, NDSI, NDWI, EVI).

Supported indices:
- NDVI: Vegetation change detection
- NDSI: Snow/ice change detection
- NDWI: Water body change detection
- EVI: Vegetation change detection
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

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


# Index type definition
IndexType = Literal["ndvi", "ndsi", "ndwi", "evi"]

# Index-specific configurations
INDEX_CONFIG = {
    "ndvi": {
        "name": "NDVI",
        "full_name": "Normalized Difference Vegetation Index",
        "subject": "Vegetation",
        "loss_label": "Vegetation Loss",
        "gain_label": "Vegetation Gain",
        "colors": [
            (0.8, 0.0, 0.0),  # Dark red - severe loss
            (1.0, 0.4, 0.4),  # Light red - moderate loss
            (1.0, 1.0, 1.0),  # White - stable
            (0.4, 0.8, 0.4),  # Light green - moderate gain
            (0.0, 0.6, 0.0),  # Dark green - strong gain
        ],
        "index_colors": [
            (0.6, 0.3, 0.1),  # Brown - bare soil/water
            (0.8, 0.6, 0.2),  # Tan - sparse vegetation
            (1.0, 1.0, 0.4),  # Yellow - moderate vegetation
            (0.6, 0.8, 0.2),  # Yellow-green
            (0.2, 0.6, 0.2),  # Green - healthy vegetation
            (0.0, 0.4, 0.0),  # Dark green - dense vegetation
        ],
        "vmin": -0.2,
        "vmax": 0.8,
    },
    "ndsi": {
        "name": "NDSI",
        "full_name": "Normalized Difference Snow Index",
        "subject": "Snow/Ice",
        "loss_label": "Snow/Ice Loss",
        "gain_label": "Snow/Ice Gain",
        "colors": [
            (0.6, 0.3, 0.1),  # Brown - severe loss (melting)
            (0.8, 0.6, 0.4),  # Tan - moderate loss
            (0.9, 0.9, 0.9),  # Light gray - stable
            (0.7, 0.85, 1.0),  # Light blue - moderate gain
            (0.3, 0.5, 0.9),  # Blue - strong gain (freezing)
        ],
        "index_colors": [
            (0.4, 0.3, 0.2),  # Dark brown - no snow
            (0.6, 0.5, 0.4),  # Brown
            (0.8, 0.8, 0.8),  # Gray - mixed
            (0.9, 0.95, 1.0),  # Very light blue
            (0.7, 0.85, 1.0),  # Light blue - snow
            (0.4, 0.6, 0.95),  # Blue - dense snow
        ],
        "vmin": -0.5,
        "vmax": 1.0,
    },
    "ndwi": {
        "name": "NDWI",
        "full_name": "Normalized Difference Water Index",
        "subject": "Water",
        "loss_label": "Water Loss",
        "gain_label": "Water Gain",
        "colors": [
            (0.7, 0.5, 0.3),  # Brown - severe loss (drying)
            (0.85, 0.7, 0.5),  # Tan - moderate loss
            (0.95, 0.95, 0.95),  # Near white - stable
            (0.6, 0.8, 1.0),  # Light blue - moderate gain
            (0.2, 0.5, 0.9),  # Blue - strong gain (flooding)
        ],
        "index_colors": [
            (0.6, 0.4, 0.2),  # Brown - dry land
            (0.8, 0.7, 0.5),  # Tan
            (0.95, 0.95, 0.95),  # Near white
            (0.6, 0.8, 1.0),  # Light blue
            (0.2, 0.5, 0.9),  # Medium blue
            (0.0, 0.2, 0.6),  # Dark blue - water
        ],
        "vmin": -0.5,
        "vmax": 0.5,
    },
    "evi": {
        "name": "EVI",
        "full_name": "Enhanced Vegetation Index",
        "subject": "Vegetation",
        "loss_label": "Vegetation Loss",
        "gain_label": "Vegetation Gain",
        "colors": [
            (0.8, 0.0, 0.0),  # Dark red - severe loss
            (1.0, 0.4, 0.4),  # Light red - moderate loss
            (1.0, 1.0, 1.0),  # White - stable
            (0.4, 0.8, 0.4),  # Light green - moderate gain
            (0.0, 0.6, 0.0),  # Dark green - strong gain
        ],
        "index_colors": [
            (0.6, 0.3, 0.1),  # Brown - bare soil/water
            (0.8, 0.6, 0.2),  # Tan - sparse vegetation
            (1.0, 1.0, 0.4),  # Yellow - moderate vegetation
            (0.6, 0.8, 0.2),  # Yellow-green
            (0.2, 0.6, 0.2),  # Green - healthy vegetation
            (0.0, 0.4, 0.0),  # Dark green - dense vegetation
        ],
        "vmin": -0.2,
        "vmax": 0.8,
    },
}


def _get_band_indices(src: rasterio.DatasetReader) -> dict[str, int]:
    """Get band indices for different sensors.

    Args:
        src: Open rasterio dataset

    Returns:
        Dictionary mapping band names to 1-based indices
    """
    band_count = src.count

    if band_count >= 7:
        # Landsat 8/9: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1
        return {"red": 4, "nir": 5, "green": 3, "swir": 6}
    elif band_count >= 6:
        # Sentinel-2 subset or similar
        return {"red": 3, "nir": 4, "green": 2, "swir": 5}
    elif band_count >= 5:
        # Generic (B,G,R,NIR,SWIR)
        return {"red": 3, "nir": 4, "green": 2, "swir": 5}
    elif band_count == 4:
        # RGBN format
        return {"red": 1, "nir": 4, "green": 2, "swir": None}
    else:
        # 2-3 bands: assume Red=1, NIR=2
        return {"red": 1, "nir": 2, "green": 1, "swir": None}


def _calculate_index_array(
    src: rasterio.DatasetReader,
    index_type: IndexType,
) -> np.ndarray:
    """Calculate spectral index array from rasterio dataset.

    Args:
        src: Open rasterio dataset
        index_type: Type of index to calculate

    Returns:
        Index array with values in appropriate range
    """
    bands = _get_band_indices(src)

    if index_type == "ndvi":
        if src.count < 2:
            raise ValueError(f"NDVI requires at least 2 bands, got {src.count}")

        red_band = src.read(bands["red"], masked=True).astype(np.float32)
        nir_band = src.read(bands["nir"], masked=True).astype(np.float32)

        # Handle masked arrays
        if hasattr(red_band, "filled"):
            red_band = red_band.filled(np.nan)
        if hasattr(nir_band, "filled"):
            nir_band = nir_band.filled(np.nan)

        # NDVI = (NIR - Red) / (NIR + Red)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = nir_band + red_band
            index = np.where(denominator != 0, (nir_band - red_band) / denominator, np.nan)

    elif index_type == "ndsi":
        if src.count < 5 or bands["swir"] is None:
            raise ValueError(f"NDSI requires SWIR band (5+ bands), got {src.count}")

        green_band = src.read(bands["green"], masked=True).astype(np.float32)
        swir_band = src.read(bands["swir"], masked=True).astype(np.float32)

        if hasattr(green_band, "filled"):
            green_band = green_band.filled(np.nan)
        if hasattr(swir_band, "filled"):
            swir_band = swir_band.filled(np.nan)

        # NDSI = (Green - SWIR) / (Green + SWIR)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = green_band + swir_band
            index = np.where(denominator != 0, (green_band - swir_band) / denominator, np.nan)

    elif index_type == "ndwi":
        if src.count < 4:
            raise ValueError(f"NDWI requires at least 4 bands (with NIR), got {src.count}")

        green_band = src.read(bands["green"], masked=True).astype(np.float32)
        nir_band = src.read(bands["nir"], masked=True).astype(np.float32)

        if hasattr(green_band, "filled"):
            green_band = green_band.filled(np.nan)
        if hasattr(nir_band, "filled"):
            nir_band = nir_band.filled(np.nan)

        # NDWI = (Green - NIR) / (Green + NIR)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = green_band + nir_band
            index = np.where(denominator != 0, (green_band - nir_band) / denominator, np.nan)

    elif index_type == "evi":
        if src.count < 3:
            raise ValueError(f"EVI requires at least 3 bands, got {src.count}")

        green_band = src.read(bands["blue"], masked=True).astype(np.float32)
        red_band = src.read(bands["red"], masked=True).astype(np.float32)
        nir_band = src.read(bands["nir"], masked=True).astype(np.float32)

        if hasattr(green_band, "filled"):
            blue_band = green_band.filled(np.nan)
        if hasattr(red_band, "filled"):
            red_band = red_band.filled(np.nan)
        if hasattr(nir_band, "filled"):
            nir_band = nir_band.filled(np.nan)

        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = nir_band + 6*red_band - 7.5*blue_band + 1
            index = np.where(denominator != 0, (2.5 * (nir_band - red_band)) / denominator, np.nan)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    return index


def _create_change_visualization(
    change_array: np.ndarray,
    index_type: IndexType,
    title: str | None = None,
) -> str | None:
    """Create a colored visualization of index change.

    Args:
        change_array: Index difference array
        index_type: Type of index for colormap selection
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    config = INDEX_CONFIG[index_type]

    if title is None:
        title = f"{config['subject']} Change Detection"

    cmap = LinearSegmentedColormap.from_list(f"{index_type}_change", config["colors"], N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Clip change values for better visualization
    vmin, vmax = -0.5, 0.5
    im = ax.imshow(change_array, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(f"{config['name']} Change", fontsize=10)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels([config["loss_label"], "", "Stable", "", config["gain_label"]])

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_index_visualization(
    index_array: np.ndarray,
    index_type: IndexType,
    title: str | None = None,
) -> str | None:
    """Create a colored visualization of spectral index.

    Args:
        index_array: Index array
        index_type: Type of index for colormap selection
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    config = INDEX_CONFIG[index_type]

    if title is None:
        title = config["name"]

    cmap = LinearSegmentedColormap.from_list(index_type, config["index_colors"], N=256)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(index_array, cmap=cmap, vmin=config["vmin"], vmax=config["vmax"])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(config["name"], fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def calculate_change(
    before_path: str,
    after_path: str,
    index_type: IndexType = "ndvi",
) -> dict[str, Any]:
    """Calculate change between two raster images using spectral index.

    Computes the specified index for both images and calculates the difference
    (after - before). Positive values indicate gain, negative values indicate loss.

    Args:
        before_path: Path to the earlier date GeoTIFF
        after_path: Path to the later date GeoTIFF
        index_type: Type of index to use ("ndvi", "ndsi", "ndwi")

    Returns:
        Dictionary containing:
            - change_array: Index difference (after - before) as list
            - index_before: Index array for before image
            - index_after: Index array for after image
            - statistics: Change statistics
            - classification: Pixel classification counts
            - metadata: Raster metadata
            - visualizations: Base64 PNG images (if matplotlib available)
            - index_type: The index type used

    Raises:
        FileNotFoundError: If either raster file not found
        ValueError: If rasters have different dimensions or invalid index type
    """
    before_path_obj = Path(before_path)
    after_path_obj = Path(after_path)

    if not before_path_obj.exists():
        raise FileNotFoundError(f"Before raster not found: {before_path}")
    if not after_path_obj.exists():
        raise FileNotFoundError(f"After raster not found: {after_path}")

    # Validate index type
    if index_type not in INDEX_CONFIG:
        raise ValueError(f"Invalid index type: {index_type}. Must be one of: {list(INDEX_CONFIG.keys())}")

    config = INDEX_CONFIG[index_type]

    # Open both rasters
    with rasterio.open(before_path) as src_before, rasterio.open(after_path) as src_after:
        # Validate dimensions match
        if (src_before.width, src_before.height) != (src_after.width, src_after.height):
            raise ValueError(
                f"Raster dimensions must match. "
                f"Before: {src_before.width}x{src_before.height}, "
                f"After: {src_after.width}x{src_after.height}"
            )

        # Calculate index for both
        index_before = _calculate_index_array(src_before, index_type)
        index_after = _calculate_index_array(src_after, index_type)

        # Calculate change (after - before)
        change = index_after - index_before

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
            subject = config["subject"].lower()

            classification = {
                f"severe_{subject}_loss": {
                    "pixels": int(severe_loss),
                    "percentage": round(100 * severe_loss / total, 2),
                },
                f"moderate_{subject}_loss": {
                    "pixels": int(moderate_loss),
                    "percentage": round(100 * moderate_loss / total, 2),
                },
                "stable": {
                    "pixels": int(stable),
                    "percentage": round(100 * stable / total, 2),
                },
                f"moderate_{subject}_gain": {
                    "pixels": int(moderate_gain),
                    "percentage": round(100 * moderate_gain / total, 2),
                },
                f"strong_{subject}_gain": {
                    "pixels": int(strong_gain),
                    "percentage": round(100 * strong_gain / total, 2),
                },
                "total_valid_pixels": int(total),
            }

        # Index statistics for each image
        valid_before = index_before[np.isfinite(index_before)]
        valid_after = index_after[np.isfinite(index_after)]

        index_stats = {
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
                index_type,
                title=f"{config['subject']} Change Detection",
            )
            visualizations[f"{index_type}_before"] = _create_index_visualization(
                index_before,
                index_type,
                title=f"{config['name']} - Before",
            )
            visualizations[f"{index_type}_after"] = _create_index_visualization(
                index_after,
                index_type,
                title=f"{config['name']} - After",
            )

        # For backwards compatibility, also include ndvi_* keys if using NDVI
        result = {
            "change_array": change.tolist(),
            "index_before": index_before.tolist(),
            "index_after": index_after.tolist(),
            "statistics": statistics,
            "index_statistics": index_stats,
            "classification": classification,
            "metadata": metadata,
            "visualizations": visualizations,
            "index_type": index_type,
            "index_name": config["name"],
        }

        # Backwards compatibility for NDVI
        if index_type == "ndvi":
            result["ndvi_before"] = result["index_before"]
            result["ndvi_after"] = result["index_after"]
            result["ndvi_statistics"] = result["index_statistics"]

        return result


def run(
    RasterPathBefore: str,
    RasterPathAfter: str,
    IndexType: str = "ndvi",
) -> dict[str, Any]:
    """Registry-compatible change detection.

    Args:
        RasterPathBefore: Path to before image (registry type)
        RasterPathAfter: Path to after image (registry type)
        IndexType: Type of index to use ("ndvi", "ndsi", "ndwi", "evi")

    Returns:
        Change detection result dictionary
    """
    # Normalize index type
    index_type = IndexType.lower() if IndexType else "ndvi"
    if index_type not in INDEX_CONFIG:
        index_type = "ndvi"  # Default fallback

    return calculate_change(RasterPathBefore, RasterPathAfter, index_type)
