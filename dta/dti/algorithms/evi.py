"""EVI (Enhanced Vegetation Index) calculation.

Thin adapter over :mod:`dta.dti.algorithms.spectral_index`. The formula,
required bands (red, nir, blue), colormap, and value range come from this
algorithm's ``config`` block in ``registry.yaml``.
"""

from __future__ import annotations

from typing import Any

from dta.dti.algorithms.spectral_index import run as _run_index
from dta.dti.registry import get_item, load_registry


def _config() -> dict[str, Any]:
    return get_item(load_registry(), "algorithms/evi").config or {}


def calculate_evi(raster_path: str) -> dict[str, Any]:
    """Calculate EVI from a multispectral raster.

    Returns a dict with ``evi_array``, ``metadata``, ``statistics``,
    ``visualizations``, and ``path``.
    """
    return _run_index(raster_path, _config())


def run(RasterPath: str) -> dict[str, Any]:  # noqa: N803  # CamelCase mirrors registry input key
    """Registry-compatible EVI calculation."""
    return calculate_evi(RasterPath)
