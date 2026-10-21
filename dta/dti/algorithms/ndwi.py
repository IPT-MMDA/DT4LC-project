"""NDWI (Normalized Difference Water Index) calculation.

Thin adapter over :mod:`dta.dti.algorithms.spectral_index`. The formula,
required bands (green, nir), colormap, and value range come from this
algorithm's ``config`` block in ``registry.yaml``. Adds NDWI-specific water
coverage statistics to the standard result dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dta.dti import artifact_files
from dta.dti.algorithms.spectral_index import calculate_index
from dta.dti.registry import get_item, load_registry

# Standard water threshold; NDWI > 0.3 typically indicates open water.
WATER_THRESHOLD = 0.3


def _config() -> dict[str, Any]:
    return get_item(load_registry(), "algorithms/ndwi").config or {}


def calculate_ndwi(raster_path: str) -> dict[str, Any]:
    """Calculate NDWI from a multispectral raster.

    Returns a dict with ``rasters`` (``ndwi``), ``metadata``, ``statistics`` (extended
    with ``water_threshold``, ``water_pixels``, ``water_coverage_percent``),
    ``visualizations``, and ``path``.
    """
    result, arr, (crs, transform) = calculate_index(raster_path, _config())
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
        water_pixels = int(np.sum(valid > WATER_THRESHOLD))
        result["statistics"].update(
            {
                "water_threshold": WATER_THRESHOLD,
                "water_pixels": water_pixels,
                "water_coverage_percent": float(water_pixels / valid.size * 100),
            }
        )
    return artifact_files.save(result, {"ndwi": arr}, result["visualizations"], crs=crs, transform=transform)


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803  # CamelCase mirrors registry input key
    """Registry-compatible NDWI calculation."""
    return calculate_ndwi(RasterPath)
