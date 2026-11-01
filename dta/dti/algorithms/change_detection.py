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

import io
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import rasterio

from dta.dti import artifact_files
from dta.dti.algorithms.spectral_index import RANGES
from dta.dti.raster import reflectance_divisor

# Try to import matplotlib for visualization, graceful fallback
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Index type definition
IndexType = Literal["ndvi", "ndsi", "ndwi", "evi"]

# Index-specific configurations
INDEX_CONFIG: dict[str, dict[str, Any]] = {
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


# Cover classes whose before/after/gained/lost areas the result reports, per index:
# (result key, threshold). Open water is NDWI > 0.2; vegetation is NDVI > 0.3, the lower
# bound of the LULC classifier's cropland/grassland and dense-vegetation classes.
COVER_THRESHOLDS: dict[str, tuple[str, float]] = {"ndwi": ("water", 0.2), "ndvi": ("vegetation", 0.3)}


def _pixel_km2(src: rasterio.DatasetReader) -> float | None:
    """Area of one pixel in km² for a projected CRS, else ``None``."""
    if src.crs is None or not src.crs.is_projected:
        return None
    return float(abs(src.transform.a * src.transform.e) * src.crs.linear_units_factor[1] ** 2 / 1e6)


def _cover_areas(
    index_before: np.ndarray, index_after: np.ndarray, valid: np.ndarray, threshold: float, pixel_km2: float
) -> dict[str, float]:
    """Cover (index > ``threshold``) in km² before and after, and gained / lost, on pixels valid on both dates."""
    with np.errstate(invalid="ignore"):
        before = valid & (index_before > threshold)
        after = valid & (index_after > threshold)
    return {
        "before_km2": round(int(before.sum()) * pixel_km2, 3),
        "after_km2": round(int(after.sum()) * pixel_km2, 3),
        "gained_km2": round(int((after & ~before).sum()) * pixel_km2, 3),
        "lost_km2": round(int((before & ~after).sum()) * pixel_km2, 3),
        "threshold": threshold,
    }


# |change| above this counts as an increase / decrease (``areas_km2`` and the Increase / Decrease classes).
STABLE_BAND = 0.05
# Class names (gained, lost, unchanged) of ``class_statistics`` for an index with a cover class.
COVER_CLASS_NAMES: dict[str, tuple[str, str, str]] = {
    "ndwi": ("Water gained", "Water lost", "No change"),
    "ndvi": ("Vegetation gained", "Vegetation lost", "No change"),
}
INCREASE_CLASS_NAMES = ("Increase", "Decrease", "Stable")


def change_class_statistics(
    change: np.ndarray,
    gained: np.ndarray,
    lost: np.ndarray,
    names: tuple[str, str, str],
    pixel_km2: float | None,
) -> dict[str, dict[str, Any]]:
    """``class_statistics`` of a change result, shaped as LULC's: per class name ``{id, name, colour,
    pixels, percent, area_km2, mean_change}``.

    The classes are ``gained`` (id 1), ``lost`` (id 2) and every other valid pixel (id 0), in that
    order, coloured as the change legend (:data:`artifact_files.CHANGE_RAMP`). ``percent`` is of the
    valid pixels (finite ``change``), ``area_km2`` is ``None`` without a projected CRS and
    ``mean_change`` is the mean of ``change`` inside the class (``None`` for an empty class).
    """
    valid = np.isfinite(change)
    gained, lost = valid & gained, valid & lost & ~gained
    lost_colour, stable_colour, gained_colour = artifact_files.CHANGE_RAMP
    total = int(valid.sum())
    classes = (
        (1, names[0], gained_colour, gained),
        (2, names[1], lost_colour, lost),
        (0, names[2], stable_colour, valid & ~gained & ~lost),
    )
    out: dict[str, dict[str, Any]] = {}
    for class_id, name, colour, mask in classes:
        pixels = int(mask.sum())
        out[name] = {
            "id": class_id,
            "name": name,
            "colour": colour,
            "pixels": pixels,
            "percent": round(100 * pixels / total, 2) if total else 0.0,
            "area_km2": round(pixels * pixel_km2, 3) if pixel_km2 is not None else None,
            "mean_change": round(float(change[mask].mean()), 4) if pixels else None,
        }
    return out


def _get_band_indices(src: rasterio.DatasetReader) -> dict[str, int | None]:
    """Get band indices for different sensors.

    Delegates to :func:`dta.dti.raster.band_indices` so the per-index files
    and change-detection share one source of truth for band layouts.
    """
    from dta.dti.raster import band_indices

    return band_indices(src)


def _calculate_index_array(
    src: rasterio.DatasetReader,
    index_type: IndexType,
) -> np.ndarray:
    """Calculate spectral index array from rasterio dataset.

    Args:
        src: Open rasterio dataset
        index_type: Type of index to calculate

    Returns:
        Index array clipped to the formula's range (``spectral_index.RANGES``, as the per-index
        analyses), NaN where a band is nodata or the denominator is zero
    """
    bands = _get_band_indices(src)
    # src.read(None) would read every band; fail clearly instead.
    required = {
        "ndvi": ("red", "nir"),
        "ndsi": ("green", "swir"),
        "ndwi": ("green", "nir"),
        "evi": ("blue", "red", "nir"),
    }
    missing = [role for role in required.get(index_type, ()) if bands[role] is None]
    if missing:
        raise ValueError(
            f"{index_type.upper()} requires band(s) {missing}, but the raster ({src.count} bands) doesn't expose them."
        )

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

        # Stored values -> 0-1 reflectance for EVI's "+1", by source dtype, as in spectral_index.run.
        dtype = np.result_type(*(src.dtypes[cast(int, bands[role]) - 1] for role in ("blue", "red", "nir")))
        divisor = reflectance_divisor(dtype, np.stack([blue_band, red_band, nir_band]))
        if divisor != 1.0:
            blue_band, red_band, nir_band = blue_band / divisor, red_band / divisor, nir_band / divisor

        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
            index = np.where(denominator != 0, (2.5 * (nir_band - red_band)) / denominator, np.nan)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Near-zero denominators (negative stored reflectance) give |index| far beyond the range.
    return np.clip(index, *RANGES[index_type])


def _create_change_visualization(
    change_array: np.ndarray,
    index_type: IndexType,
    title: str | None = None,
) -> bytes | None:
    """Create a colored visualization of index change.

    Args:
        change_array: Index difference array
        index_type: Type of index for colormap selection
        title: Title for the image

    Returns:
        PNG bytes, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    config = INDEX_CONFIG[index_type]

    if title is None:
        title = f"{config['subject']} Change Detection"

    cmap = LinearSegmentedColormap.from_list(f"{index_type}_change", config["colors"], N=256)

    fig = Figure(figsize=(10, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()

    # Clip change values for better visualization
    vmin, vmax = -0.5, 0.5
    im = ax.imshow(change_array, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(f"{config['name']} Change", fontsize=10)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_ticklabels([config["loss_label"], "", "Stable", "", config["gain_label"]])

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")

    return buf.getvalue()


def _create_index_visualization(
    index_array: np.ndarray,
    index_type: IndexType,
    title: str | None = None,
) -> bytes | None:
    """Create a colored visualization of spectral index.

    Args:
        index_array: Index array
        index_type: Type of index for colormap selection
        title: Title for the image

    Returns:
        PNG bytes, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None

    config = INDEX_CONFIG[index_type]

    if title is None:
        title = config["name"]

    cmap = LinearSegmentedColormap.from_list(index_type, config["index_colors"], N=256)

    fig = Figure(figsize=(10, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()

    im = ax.imshow(index_array, cmap=cmap, vmin=config["vmin"], vmax=config["vmax"])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(config["name"], fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")

    return buf.getvalue()


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
            - rasters: GeoTIFF references ``change`` (after - before),
              ``index_before`` and ``index_after``
            - statistics: Change statistics; with a projected CRS also
              ``areas_km2: {increase, decrease, stable}`` (change > 0.05,
              < -0.05, within ±0.05) and, for NDWI ``water`` and for NDVI
              ``vegetation``, ``{before_km2, after_km2, gained_km2, lost_km2,
              threshold}`` (see :data:`COVER_THRESHOLDS`); ``None`` otherwise
            - classification: Pixel classification counts
            - class_statistics: :func:`change_class_statistics`: for NDWI and NDVI
              ``Water`` / ``Vegetation gained``, ``... lost`` (the cover class appeared or
              disappeared) and ``No change``; for other indices ``Increase`` / ``Decrease`` /
              ``Stable`` (change beyond ±:data:`STABLE_BAND`)
            - metadata: Raster metadata
            - visualizations: PNG references (if matplotlib available) and the
              map-only ``change_map_bare``, ``<index>_before_bare``, ``<index>_after_bare``
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

        statistics: dict[str, Any] = {}
        classification = {}
        class_statistics: dict[str, dict[str, Any]] = {}

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

            # Areas of the gain / loss / stable classes above, and of the index's cover class.
            pixel_km2 = _pixel_km2(src_after)
            with np.errstate(invalid="ignore"):
                if index_type in COVER_THRESHOLDS:
                    cover = COVER_THRESHOLDS[index_type][1]
                    before_cover, after_cover = index_before > cover, index_after > cover
                    gained, lost = after_cover & ~before_cover, before_cover & ~after_cover
                    names = COVER_CLASS_NAMES[index_type]
                else:
                    gained, lost, names = change > STABLE_BAND, change < -STABLE_BAND, INCREASE_CLASS_NAMES
            class_statistics = change_class_statistics(change, gained, lost, names, pixel_km2)
            statistics["areas_km2"] = (
                {
                    "increase": round(int(moderate_gain + strong_gain) * pixel_km2, 3),
                    "decrease": round(int(severe_loss + moderate_loss) * pixel_km2, 3),
                    "stable": round(int(stable) * pixel_km2, 3),
                }
                if pixel_km2 is not None
                else None
            )
            if index_type in COVER_THRESHOLDS:
                key, threshold = COVER_THRESHOLDS[index_type]
                statistics[key] = (
                    _cover_areas(index_before, index_after, valid_mask, threshold, pixel_km2)
                    if pixel_km2 is not None
                    else None
                )

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

        result = {
            "statistics": statistics,
            "index_statistics": index_stats,
            "classification": classification,
            "class_statistics": class_statistics,
            "metadata": metadata,
            "visualizations": visualizations,
            "index_type": index_type,
            "index_name": config["name"],
        }

        # Backwards compatibility for NDVI
        if index_type == "ndvi":
            result["ndvi_statistics"] = result["index_statistics"]

        rasters = {"change": change, "index_before": index_before, "index_after": index_after}
        # Tiles and bare PNGs: change on the legend ramp over the change PNG's ±0.5, indices as their PNGs.
        index_look = artifact_files.index_style(config["index_colors"], config["vmin"], config["vmax"])
        styles = {"change": artifact_files.change_style(0.5), "index_before": index_look, "index_after": index_look}
        bare = {"change_map": "change", f"{index_type}_before": "index_before", f"{index_type}_after": "index_after"}
        return artifact_files.save(
            result,
            rasters,
            visualizations,
            crs=src_after.crs,
            transform=src_after.transform,
            styles=styles,
            bare=bare,
        )


def run(
    RasterPathBefore: str,  # noqa: N803  # CamelCase mirrors registry input key
    RasterPathAfter: str,  # noqa: N803  # CamelCase mirrors registry input key
    IndexType: str = "ndvi",  # noqa: N803  # CamelCase mirrors registry input key
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
    index_type_str = IndexType.lower() if IndexType else "ndvi"
    if index_type_str not in INDEX_CONFIG:
        index_type_str = "ndvi"  # Default fallback
    index_type: Literal["ndvi", "ndsi", "ndwi", "evi"] = cast(Literal["ndvi", "ndsi", "ndwi", "evi"], index_type_str)

    return calculate_change(RasterPathBefore, RasterPathAfter, index_type)
