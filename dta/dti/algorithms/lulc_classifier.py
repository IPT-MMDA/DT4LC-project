"""Land Use / Land Cover (LULC) Classification.

Threshold-based classification using spectral indices (NDVI, NDWI, NDSI).
Provides rapid land cover assessment without requiring training data.

Classes:
- Water: NDWI > 0.3 or NDVI < -0.1
- Snow/Ice: NDSI > 0.4 and brightness > 0.3
- Bare Soil: NDVI < 0.1 and NDWI < 0
- Sparse Vegetation: 0.1 <= NDVI < 0.3
- Cropland/Grassland: 0.3 <= NDVI < 0.5
- Dense Vegetation: NDVI >= 0.5

References:
- McFeeters (1996) - NDWI for water body delineation
- Zha et al. (2003) - Built-up index concepts
- Tucker (1979) - NDVI thresholds for vegetation
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
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# LULC class definitions
LULC_CLASSES = {
    0: {"name": "No Data", "color": "#000000"},
    1: {"name": "Water", "color": "#0077be"},
    2: {"name": "Snow/Ice", "color": "#ffffff"},
    3: {"name": "Bare Soil", "color": "#c2b280"},
    4: {"name": "Sparse Vegetation", "color": "#d4e157"},
    5: {"name": "Cropland/Grassland", "color": "#8bc34a"},
    6: {"name": "Dense Vegetation", "color": "#2e7d32"},
}


def _get_band_indices(band_count: int) -> dict[str, int | None]:
    """Determine band indices for different sensors.

    Args:
        band_count: Number of bands in the raster

    Returns:
        Dictionary mapping band names to 1-based band indices (None if unavailable)
    """
    if band_count >= 7:
        # Landsat 8/9: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1
        return {
            "green": 3,
            "red": 4,
            "nir": 5,
            "swir": 6,
        }
    elif band_count >= 6:
        # Sentinel-2 subset (B2,B3,B4,B8,B11,B12) or similar
        return {
            "green": 2,
            "red": 3,
            "nir": 4,
            "swir": 5,
        }
    elif band_count >= 5:
        # Generic (B,G,R,NIR,SWIR)
        return {
            "green": 2,
            "red": 3,
            "nir": 4,
            "swir": 5,
        }
    elif band_count == 4:
        # RGBN - no SWIR available
        return {
            "green": 2,
            "red": 1,
            "nir": 4,
            "swir": None,
        }
    else:
        # Minimal bands
        return {
            "green": 1,
            "red": 1,
            "nir": 2,
            "swir": None,
        }


def _calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculate NDVI: (NIR - Red) / (NIR + Red)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
    return ndvi


def _calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculate NDWI: (Green - NIR) / (Green + NIR).

    McFeeters (1996) water index. Positive values indicate water.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = green + nir
        ndwi = np.where(denominator != 0, (green - nir) / denominator, 0.0)
    return ndwi


def _calculate_ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Calculate NDSI: (Green - SWIR) / (Green + SWIR).

    Snow index. Values > 0.4 typically indicate snow/ice.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = green + swir
        ndsi = np.where(denominator != 0, (green - swir) / denominator, 0.0)
    return ndsi


def _create_lulc_visualization(
    classification: np.ndarray,
    title: str = "Land Cover Classification",
) -> str | None:
    """Create colored visualization of LULC classification.

    Args:
        classification: Classification array with class values 0-6
        title: Title for the image

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Create colormap from class colors
    colors = [LULC_CLASSES[i]["color"] for i in range(7)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(classification, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Create legend
    legend_elements = [
        Patch(facecolor=LULC_CLASSES[i]["color"], edgecolor="black", label=LULC_CLASSES[i]["name"])
        for i in range(1, 7)  # Skip "No Data"
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def _create_histogram_visualization(
    class_stats: dict[str, dict[str, Any]],
    title: str = "Land Cover Distribution",
) -> str | None:
    """Create bar chart of land cover distribution.

    Args:
        class_stats: Dictionary with class statistics
        title: Title for the chart

    Returns:
        Base64 encoded PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    # Extract data (skip No Data class)
    classes = []
    percentages = []
    colors = []

    for class_id in range(1, 7):
        class_name = LULC_CLASSES[class_id]["name"]
        if class_name in class_stats:
            classes.append(class_name)
            percentages.append(class_stats[class_name]["percentage"])
            colors.append(LULC_CLASSES[class_id]["color"])

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(classes, percentages, color=colors, edgecolor="black", linewidth=0.5)

    # Add percentage labels
    for bar, pct in zip(bars, percentages, strict=False):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Coverage (%)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(percentages) * 1.15 if percentages else 100)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def classify_land_cover(raster_path: str) -> dict[str, Any]:
    """Classify land cover using spectral index thresholds.

    Uses NDVI, NDWI, and optionally NDSI to classify pixels into
    land cover categories. This is a rule-based approach suitable
    for rapid assessment without training data.

    Args:
        raster_path: Path to multispectral GeoTIFF

    Returns:
        Dictionary containing:
            - classification: 2D array of class values (0-6)
            - class_statistics: Per-class pixel counts and percentages
            - indices: Computed spectral indices (NDVI, NDWI, NDSI)
            - metadata: Raster metadata
            - visualizations: Classification map and histogram

    Raises:
        FileNotFoundError: If raster file not found
        ValueError: If raster has insufficient bands
    """
    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.count < 4:
            raise ValueError(f"LULC classification requires at least 4 bands, got {src.count}")

        # Get band indices for this sensor
        bands = _get_band_indices(src.count)

        # Read required bands
        green = src.read(bands["green"], masked=True).astype(np.float32)
        red = src.read(bands["red"], masked=True).astype(np.float32)
        nir = src.read(bands["nir"], masked=True).astype(np.float32)

        # Handle masked arrays
        if hasattr(green, "filled"):
            green = green.filled(np.nan)
        if hasattr(red, "filled"):
            red = red.filled(np.nan)
        if hasattr(nir, "filled"):
            nir = nir.filled(np.nan)

        # Calculate indices
        ndvi = _calculate_ndvi(red, nir)
        ndwi = _calculate_ndwi(green, nir)

        # SWIR-based indices (if available)
        has_swir = bands["swir"] is not None
        if has_swir:
            swir = src.read(bands["swir"], masked=True).astype(np.float32)
            if hasattr(swir, "filled"):
                swir = swir.filled(np.nan)
            ndsi = _calculate_ndsi(green, swir)
        else:
            ndsi = np.zeros_like(ndvi)

        # Calculate brightness (mean of visible bands)
        brightness = (green + red) / 2.0
        # Normalize to 0-1 range if needed
        if np.nanmax(brightness) > 1.0:
            brightness = brightness / np.nanmax(brightness)

        # Initialize classification array
        classification = np.zeros_like(ndvi, dtype=np.uint8)

        # Valid data mask
        valid_mask = np.isfinite(ndvi) & np.isfinite(ndwi)

        # Apply classification rules
        # Priority order: Water/Snow first (most distinct), then vegetation classes
        # This prevents misclassification of water as vegetation

        # Initialize with default class
        classification[valid_mask] = 5  # Default: Cropland/Grassland

        # 1. WATER - Check first (highest priority for distinct features)
        # NDWI > 0.2 captures most water bodies
        # Also include very low NDVI (< -0.1) which indicates water
        # Additional: low NIR reflectance is characteristic of water
        water_mask = (
            (ndwi > 0.2)  # Primary water indicator
            | (ndvi < -0.1)  # Very negative NDVI indicates water
            | ((ndwi > 0.0) & (ndvi < 0.0))  # Mixed signal - likely water/wet
        )
        classification[valid_mask & water_mask] = 1

        # 2. SNOW/ICE - High NDSI with high brightness (if SWIR available)
        if has_swir:
            snow_mask = (ndsi > 0.4) & (brightness > 0.3) & ~water_mask
            classification[valid_mask & snow_mask] = 2

        # 3. BARE SOIL - Low NDVI, negative NDWI (dry), not water
        bare_soil_mask = (ndvi < 0.15) & (ndwi < -0.1) & ~water_mask
        classification[valid_mask & bare_soil_mask] = 3

        # 4. SPARSE VEGETATION - Low to moderate NDVI
        sparse_veg_mask = (ndvi >= 0.1) & (ndvi < 0.3) & ~water_mask & ~bare_soil_mask
        classification[valid_mask & sparse_veg_mask] = 4

        # 5. CROPLAND/GRASSLAND - Moderate NDVI (default for valid non-water pixels)
        # Already set as default, will remain for 0.3 <= NDVI < 0.5

        # 6. DENSE VEGETATION - High NDVI
        dense_veg_mask = (ndvi >= 0.5) & ~water_mask
        classification[valid_mask & dense_veg_mask] = 6

        # No Data: invalid pixels
        classification[~valid_mask] = 0

        # Calculate class statistics
        total_valid = np.sum(valid_mask)
        class_statistics = {}

        for class_id in range(1, 7):
            class_name = LULC_CLASSES[class_id]["name"]
            pixel_count = int(np.sum(classification == class_id))
            percentage = round(100 * pixel_count / total_valid, 2) if total_valid > 0 else 0.0

            class_statistics[class_name] = {
                "class_id": class_id,
                "pixels": pixel_count,
                "percentage": percentage,
            }

        # Index statistics
        valid_ndvi = ndvi[valid_mask]
        valid_ndwi = ndwi[valid_mask]

        index_stats = {
            "ndvi": {
                "min": float(np.nanmin(valid_ndvi)) if valid_ndvi.size > 0 else None,
                "max": float(np.nanmax(valid_ndvi)) if valid_ndvi.size > 0 else None,
                "mean": float(np.nanmean(valid_ndvi)) if valid_ndvi.size > 0 else None,
                "std": float(np.nanstd(valid_ndvi)) if valid_ndvi.size > 0 else None,
            },
            "ndwi": {
                "min": float(np.nanmin(valid_ndwi)) if valid_ndwi.size > 0 else None,
                "max": float(np.nanmax(valid_ndwi)) if valid_ndwi.size > 0 else None,
                "mean": float(np.nanmean(valid_ndwi)) if valid_ndwi.size > 0 else None,
                "std": float(np.nanstd(valid_ndwi)) if valid_ndwi.size > 0 else None,
            },
        }

        if has_swir:
            valid_ndsi = ndsi[valid_mask]
            index_stats["ndsi"] = {
                "min": float(np.nanmin(valid_ndsi)) if valid_ndsi.size > 0 else None,
                "max": float(np.nanmax(valid_ndsi)) if valid_ndsi.size > 0 else None,
                "mean": float(np.nanmean(valid_ndsi)) if valid_ndsi.size > 0 else None,
                "std": float(np.nanstd(valid_ndsi)) if valid_ndsi.size > 0 else None,
            }

        # Metadata
        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "has_swir": has_swir,
        }

        # Summary statistics
        statistics = {
            "total_pixels": int(classification.size),
            "valid_pixels": int(total_valid),
            "nodata_pixels": int(classification.size - total_valid),
            "dominant_class": max(class_statistics.items(), key=lambda x: x[1]["pixels"])[0]
            if class_statistics
            else None,
        }

        # Generate visualizations
        visualizations = {}
        if HAS_MATPLOTLIB:
            visualizations["classification_map"] = _create_lulc_visualization(
                classification,
                title=f"Land Cover - {Path(raster_path).stem}",
            )
            visualizations["distribution_chart"] = _create_histogram_visualization(
                class_statistics,
                title="Land Cover Distribution",
            )

        return {
            "classification": classification.tolist(),
            "class_statistics": class_statistics,
            "index_statistics": index_stats,
            "statistics": statistics,
            "metadata": metadata,
            "visualizations": visualizations,
            "path": str(raster_path),
        }


def run(RasterPath: str) -> dict[str, Any]:
    """Registry-compatible LULC classification.

    Args:
        RasterPath: Path to raster file (registry type)

    Returns:
        LULC classification result dictionary
    """
    return classify_land_cover(RasterPath)
