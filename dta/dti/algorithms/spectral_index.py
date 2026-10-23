"""Parameterized spectral-index runner.

One module shared by every spectral-index algorithm (NDVI, NDWI, NDSI, EVI).
The formula, required bands, colormap, and value range come from the caller's
``index_config`` dict — typically loaded from each algorithm's ``config``
field in ``registry.yaml``.

Adding a new spectral index = one new entry in ``FORMULAS`` (and optionally
``COLORMAPS``) plus a registry-yaml stanza pointing at this module. No new
file in ``algorithms/``.
"""

from __future__ import annotations

from collections.abc import Callable
import io
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from dta.dti import artifact_files
from dta.dti.raster import band_indices, reflectance_divisor

# Try to import matplotlib for visualization; degrade gracefully if absent.
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --- Formulas: name → callable(bands_dict) → index array ---------------------
# Each callable receives a dict with float NumPy arrays keyed by band role
# (red, nir, green, blue, swir) and returns the index array. Division-by-zero
# is handled by the caller (NaN propagates through float division on masked
# arrays).


def _ndvi(b: dict[str, np.ndarray]) -> np.ndarray:
    denom = b["nir"] + b["red"]
    return np.where(denom != 0, (b["nir"] - b["red"]) / denom, 0.0)


def _ndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    denom = b["green"] + b["nir"]
    return np.where(denom != 0, (b["green"] - b["nir"]) / denom, 0.0)


def _ndsi(b: dict[str, np.ndarray]) -> np.ndarray:
    denom = b["green"] + b["swir"]
    return np.where(denom != 0, (b["green"] - b["swir"]) / denom, 0.0)


def _evi(b: dict[str, np.ndarray]) -> np.ndarray:
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    return 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1)


FORMULAS: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
    "ndvi": _ndvi,
    "ndwi": _ndwi,
    "ndsi": _ndsi,
    "evi": _evi,
}

# Value range of each formula; results are clipped to it.
RANGES: dict[str, tuple[float, float]] = {
    "ndvi": (-1.0, 1.0),
    "ndwi": (-1.0, 1.0),
    "ndsi": (-1.0, 1.0),
    "evi": (-1.0, 1.0),
}


# --- Colormaps: name → list of RGB tuples ------------------------------------
# Reused from the legacy per-index files; LinearSegmentedColormap interpolates
# between them. Add a new entry to support a new index's visualization.

COLORMAPS: dict[str, list[tuple[float, float, float]]] = {
    "ndvi": [
        (0.6, 0.3, 0.1),  # Brown - bare soil/water
        (0.8, 0.6, 0.2),  # Tan - sparse vegetation
        (1.0, 1.0, 0.4),  # Yellow - moderate vegetation
        (0.6, 0.8, 0.2),  # Yellow-green
        (0.2, 0.6, 0.2),  # Green - healthy vegetation
        (0.0, 0.4, 0.0),  # Dark green - dense vegetation
    ],
    "ndwi": [
        (0.6, 0.4, 0.2),  # Brown - dry land/vegetation
        (0.8, 0.7, 0.5),  # Tan
        (0.95, 0.95, 0.95),  # Near white - transition
        (0.6, 0.8, 1.0),  # Light blue - shallow/mixed
        (0.2, 0.5, 0.9),  # Medium blue - water
        (0.0, 0.2, 0.6),  # Dark blue - deep water
    ],
    "ndsi": [
        (0.4, 0.3, 0.2),  # Dark brown - no snow / vegetation
        (0.7, 0.6, 0.4),  # Tan - bare soil
        (1.0, 0.9, 0.7),  # Cream - mixed
        (0.8, 0.9, 1.0),  # Pale blue - thin snow
        (0.4, 0.7, 1.0),  # Blue - moderate snow
        (1.0, 1.0, 1.0),  # White - dense snow / ice
    ],
}


# --- Public API --------------------------------------------------------------


def run(raster_path: str, index_config: dict[str, Any]) -> dict[str, Any]:
    """Calculate a spectral index and write it as GeoTIFF + PNG (see :func:`calculate_index`).

    Returns:
        The :func:`calculate_index` result with the array in
        ``rasters["<formula>"]``, ``visualizations["<formula>_map"]`` and its
        map-only ``visualizations["<formula>_map_bare"]`` as artifact
        references (``{kind, path, url, stats, size_bytes}``).
    """
    result, index_array, (crs, transform) = calculate_index(raster_path, index_config)
    key = result["index_type"]
    return artifact_files.save(
        result,
        {key: index_array},
        result["visualizations"],
        crs=crs,
        transform=transform,
        styles={key: display_style(index_config)},
        bare={f"{key}_map": key},
    )


def display_style(index_config: dict[str, Any]) -> dict[str, Any]:
    """The colours and value range of an index's PNG, as an :func:`dta.dti.artifact_files.index_style`."""
    formula_name = str(index_config.get("formula") or "ndvi").lower()
    colormap_name = str(index_config.get("colormap") or formula_name)
    colours = COLORMAPS.get(colormap_name) or COLORMAPS["ndvi"]
    return artifact_files.index_style(
        colours, float(index_config.get("vmin", -1.0)), float(index_config.get("vmax", 1.0))
    )


def calculate_index(
    raster_path: str, index_config: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, tuple[Any, Any]]:
    """Calculate a spectral index from a raster + config, in memory.

    Band values are scaled to 0-1 reflectance first, by the source dtype
    (:func:`dta.dti.raster.reflectance_divisor`). Pixels that are nodata or
    non-finite in any required band are NaN in the output; values are clipped
    to the formula's range (``RANGES``); statistics cover valid pixels only.

    Args:
        raster_path: Path to a multi-band GeoTIFF.
        index_config: Dict with keys:
            - ``formula``: name in ``FORMULAS`` (e.g. "ndvi", "ndwi").
            - ``required_bands``: list of band-role names the formula needs.
            - ``colormap``: name in ``COLORMAPS`` (optional; falls back to "ndvi").
            - ``vmin`` / ``vmax``: visualization value range (optional).

    Returns:
        ``(result, index_array, (crs, transform))`` where ``index_array`` is
        NaN-filled, ``(crs, transform)`` are the input's, and ``result`` is::

            {
                "metadata": {crs, transform, bounds, width, height, count},
                "statistics": {min, max, mean, std, median, valid_pixels, total_pixels},
                "visualizations": {"<formula>_map": <png bytes> | None},
                "path": "<input path>",
                "index_type": "<formula>",
            }

    Raises:
        FileNotFoundError: If raster file doesn't exist.
        ValueError: If the formula is unknown or the raster lacks required bands.
    """
    formula_name = str(index_config.get("formula") or "ndvi").lower()
    required = list(index_config.get("required_bands") or [])
    colormap_name = str(index_config.get("colormap") or formula_name)
    vmin = float(index_config.get("vmin", -1.0))
    vmax = float(index_config.get("vmax", 1.0))

    if formula_name not in FORMULAS:
        raise ValueError(f"Unknown spectral index formula: {formula_name!r}")

    raster_path_obj = Path(raster_path)
    if not raster_path_obj.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    formula = FORMULAS[formula_name]

    with rasterio.open(raster_path) as src:
        roles = band_indices(src)
        bands: dict[str, np.ndarray] = {}
        dtypes: list[str] = []
        for role in required:
            idx = roles.get(role)
            if idx is None:
                raise ValueError(
                    f"{formula_name.upper()} requires band '{role}', but the raster "
                    f"({src.count} bands) doesn't expose it."
                )
            dtypes.append(src.dtypes[idx - 1])
            data = src.read(idx, masked=True).astype(float)
            if hasattr(data, "filled"):
                data = data.filled(np.nan)  # nodata -> NaN
            bands[role] = data

        # A pixel is valid only if every required band is finite (not NaN/inf/nodata).
        input_valid = np.logical_and.reduce([np.isfinite(b) for b in bands.values()])
        # Stored values -> 0-1 reflectance, decided by the source dtype.
        divisor = reflectance_divisor(np.result_type(*dtypes), np.stack([b[input_valid] for b in bands.values()]))
        if divisor != 1.0:
            bands = {role: b / divisor for role, b in bands.items()}

        with np.errstate(divide="ignore", invalid="ignore"):
            index_array = formula(bands)
        # Mask invalids that the divide-by-zero protection didn't catch (NaN inputs).
        if np.ma.isMaskedArray(index_array):
            index_array = index_array.filled(np.nan)
        index_array = np.asarray(index_array, dtype=np.float64)
        index_array[~input_valid | ~np.isfinite(index_array)] = np.nan
        lo, hi = RANGES.get(formula_name, (-np.inf, np.inf))
        index_array = np.clip(index_array, lo, hi)

        valid_mask = np.isfinite(index_array)
        valid = index_array[valid_mask]
        stats: dict[str, float | int] = {}
        if valid.size > 0:
            stats = {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "median": float(np.median(valid)),
                "valid_pixels": int(valid.size),
                "total_pixels": int(index_array.size),
            }

        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
            "count": src.count,
        }
        geo = (src.crs, src.transform)

    visualizations: dict[str, bytes | None] = {}
    if HAS_MATPLOTLIB:
        visualizations[f"{formula_name}_map"] = _render_index_png(
            index_array,
            colormap_name=colormap_name,
            vmin=vmin,
            vmax=vmax,
            title=f"{formula_name.upper()} - {raster_path_obj.stem}",
        )

    result = {
        "metadata": metadata,
        "statistics": stats,
        "visualizations": visualizations,
        "path": str(raster_path),
        "index_type": formula_name,
    }
    return result, index_array, geo


def run_change(
    before_path: str,
    after_path: str,
    index_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute the temporal difference of a spectral index between two rasters.

    Calls :func:`calculate_index` on each path and computes ``after - before``;
    the change is written as GeoTIFF and referenced from ``rasters["change"]``.
    """
    before, arr_before, _ = calculate_index(before_path, index_config)
    after, arr_after, (crs, transform) = calculate_index(after_path, index_config)
    formula_name = before["index_type"]
    if arr_before.shape != arr_after.shape:
        raise ValueError(
            f"{formula_name.upper()} arrays must have same shape; got {arr_before.shape} and {arr_after.shape}."
        )

    change = arr_after - arr_before
    valid = change[np.isfinite(change)]
    stats: dict[str, float | int] = {}
    if valid.size > 0:
        stats = {
            "min_change": float(np.min(valid)),
            "max_change": float(np.max(valid)),
            "mean_change": float(np.mean(valid)),
            "std_change": float(np.std(valid)),
            "median_change": float(np.median(valid)),
            "increase_pixels": int(np.sum(valid > 0.1)),
            "decrease_pixels": int(np.sum(valid < -0.1)),
            "stable_pixels": int(np.sum(np.abs(valid) <= 0.1)),
            "total_valid_pixels": int(valid.size),
        }

    result = {
        "statistics": stats,
        "metadata": after["metadata"],
        "before_path": before.get("path"),
        "after_path": after.get("path"),
        "index_type": formula_name,
    }
    return artifact_files.save(
        result,
        {"change": change},
        None,
        crs=crs,
        transform=transform,
        styles={"change": artifact_files.change_style(0.5)},
    )


# --- Internals ---------------------------------------------------------------


def _render_index_png(
    index_array: np.ndarray,
    *,
    colormap_name: str,
    vmin: float,
    vmax: float,
    title: str,
    label: str | None = None,
) -> bytes | None:
    """Render an index array to PNG bytes using a registered colormap (colorbar ``label``, default its name)."""
    if not HAS_MATPLOTLIB:
        return None

    colors = COLORMAPS.get(colormap_name) or COLORMAPS["ndvi"]
    cmap = LinearSegmentedColormap.from_list(colormap_name, colors, N=256)

    fig = Figure(figsize=(10, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(label or colormap_name.upper(), fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    return buf.getvalue()
