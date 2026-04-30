"""NDVI (Normalized Difference Vegetation Index) calculation.

Thin adapter over :mod:`dta.dti.algorithms.spectral_index`. The formula,
required bands, colormap, and value range are loaded from this algorithm's
``config`` block in ``registry.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dta.dti.algorithms.spectral_index import run as _run_index
from dta.dti.registry import get_item, load_registry


def _config() -> dict[str, Any]:
    return get_item(load_registry(), "algorithms/ndvi").config or {}


def calculate_ndvi(raster_path: str) -> dict[str, Any]:
    """Calculate NDVI from a multispectral raster.

    Returns a dict with ``ndvi_array``, ``metadata``, ``statistics``,
    ``visualizations``, and ``path`` (compatible with the legacy API).
    """
    return _run_index(raster_path, _config())


def ndvi_change(
    ndvi_before: dict[str, Any],
    ndvi_after: dict[str, Any],
) -> dict[str, Any]:
    """Compute NDVI change between two pre-computed NDVI results."""
    arr_before = np.asarray(ndvi_before["ndvi_array"], dtype=float)
    arr_after = np.asarray(ndvi_after["ndvi_array"], dtype=float)
    if arr_before.shape != arr_after.shape:
        raise ValueError(f"NDVI arrays must have same shape; got {arr_before.shape} and {arr_after.shape}.")

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
        "metadata": ndvi_after["metadata"],
        "before_path": ndvi_before.get("path"),
        "after_path": ndvi_after.get("path"),
    }


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803  # CamelCase mirrors registry input key
    """Registry-compatible NDVI calculation."""
    return calculate_ndvi(RasterPath)
