"""NDSI (Normalized Difference Snow Index) calculation.

Thin adapter over :mod:`dta.dti.algorithms.spectral_index`. The formula,
required bands (green, swir), colormap, and value range come from this
algorithm's ``config`` block in ``registry.yaml``. Adds NDSI-specific snow
coverage statistics to the standard result dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dta.dti.algorithms.spectral_index import run as _run_index
from dta.dti.registry import get_item, load_registry

# Standard snow threshold; NDSI ≥ 0.42 typically indicates snow/ice.
SNOW_THRESHOLD = 0.42


def _config() -> dict[str, Any]:
    return get_item(load_registry(), "algorithms/ndsi").config or {}


def calculate_ndsi(raster_path: str) -> dict[str, Any]:
    """Calculate NDSI from a multispectral raster.

    Returns a dict with ``ndsi_array``, ``snow_mask``, ``metadata``,
    ``statistics`` (extended with ``snow_threshold``, ``snow_pixels``,
    ``non_snow_pixels``, ``snow_coverage_percent``), ``visualizations``, and
    ``path``.
    """
    result = _run_index(raster_path, _config())
    arr = np.asarray(result["ndsi_array"], dtype=float)
    finite = np.isfinite(arr)
    snow_mask = (finite & (arr >= SNOW_THRESHOLD)).astype(np.uint8)
    result["snow_mask"] = snow_mask
    valid = arr[finite]
    if valid.size > 0:
        snow_pixels = int(np.sum(valid >= SNOW_THRESHOLD))
        result["statistics"].update(
            {
                "snow_threshold": SNOW_THRESHOLD,
                "snow_pixels": snow_pixels,
                "non_snow_pixels": int(valid.size - snow_pixels),
                "snow_coverage_percent": float(snow_pixels / valid.size * 100),
            }
        )
    return result


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803  # CamelCase mirrors registry input key
    """Registry-compatible NDSI calculation."""
    return calculate_ndsi(RasterPath)
