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

import base64
from collections.abc import Callable
import io
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from dta.dti.raster import band_indices

# Try to import matplotlib for visualization; degrade gracefully if absent.
try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt

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
    """Calculate a spectral index from a raster + config.

    Args:
        raster_path: Path to a multi-band GeoTIFF.
        index_config: Dict with keys:
            - ``formula``: name in ``FORMULAS`` (e.g. "ndvi", "ndwi").
            - ``required_bands``: list of band-role names the formula needs.
            - ``colormap``: name in ``COLORMAPS`` (optional; falls back to "ndvi").
            - ``vmin`` / ``vmax``: visualization value range (optional).

    Returns:
        Dict with::

            {
                "<formula>_array": list[list[float]] (NaN-filled),
                "metadata": {crs, transform, bounds, width, height, count},
                "statistics": {min, max, mean, std, median, valid_pixels, total_pixels},
                "visualizations": {"<formula>_map": <base64 png> | None},
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
        for role in required:
            idx = roles.get(role)
            if idx is None:
                raise ValueError(
                    f"{formula_name.upper()} requires band '{role}', but the raster "
                    f"({src.count} bands) doesn't expose it."
                )
            data = src.read(idx, masked=True).astype(float)
            if hasattr(data, "filled"):
                data = data.filled(np.nan)
            bands[role] = data

        index_array = formula(bands)
        # Mask invalids that the divide-by-zero protection didn't catch (NaN inputs).
        if np.ma.isMaskedArray(index_array):
            index_array = index_array.filled(np.nan)
        index_array = np.asarray(index_array, dtype=np.float64)

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

    visualizations: dict[str, str | None] = {}
    if HAS_MATPLOTLIB:
        visualizations[f"{formula_name}_map"] = _render_index_png(
            index_array,
            colormap_name=colormap_name,
            vmin=vmin,
            vmax=vmax,
            title=f"{formula_name.upper()} - {raster_path_obj.stem}",
        )

    return {
        f"{formula_name}_array": index_array.tolist(),
        "metadata": metadata,
        "statistics": stats,
        "visualizations": visualizations,
        "path": str(raster_path),
        "index_type": formula_name,
    }


def run_change(
    before_path: str,
    after_path: str,
    index_config: dict[str, Any],
) -> dict[str, Any]:
    """Compute the temporal difference of a spectral index between two rasters.

    Convenience wrapper that calls :func:`run` on each path and computes
    ``after - before``. The legacy per-index ``<index>_change()`` helpers
    in `ndvi.py` etc. are thin shims around this.
    """
    before = run(before_path, index_config)
    after = run(after_path, index_config)
    formula_name = before["index_type"]
    array_key = f"{formula_name}_array"

    arr_before = np.asarray(before[array_key], dtype=np.float64)
    arr_after = np.asarray(after[array_key], dtype=np.float64)
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

    return {
        "change_array": change,
        "statistics": stats,
        "metadata": after["metadata"],
        "before_path": before.get("path"),
        "after_path": after.get("path"),
        "index_type": formula_name,
    }


# --- Internals ---------------------------------------------------------------


def _render_index_png(
    index_array: np.ndarray,
    *,
    colormap_name: str,
    vmin: float,
    vmax: float,
    title: str,
) -> str | None:
    """Render an index array to a base64 PNG using a registered colormap."""
    if not HAS_MATPLOTLIB:
        return None

    colors = COLORMAPS.get(colormap_name) or COLORMAPS["ndvi"]
    cmap = LinearSegmentedColormap.from_list(colormap_name, colors, N=256)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label(colormap_name.upper(), fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
