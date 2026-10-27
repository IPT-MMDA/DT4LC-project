"""The area of a region sent from the map, as a raster, for analyses that fetch their own data.

Runner of ``input/region-footprint``. A model that builds its inputs itself from the area (registry ``config:
region_input: footprint``, e.g. ``models/lulc-change-hazard``) needs only the region's bounds, not an imagery
composite: :func:`dta.dti.data_sources.gee_region.inject_region_fetch` plans this step instead of an Earth Engine
fetch when every analysis step of the plan says so. The file is a 2 x 2 pixel GeoTIFF in EPSG:4326 covering the
bounds, all no data; nothing here calls Earth Engine.
"""

from __future__ import annotations

from typing import Any
import uuid

import numpy as np

CONFIG_KEYS = frozenset({"name", "bbox"})


def check_config(config: dict[str, Any]) -> None:
    """ValueError unless ``config`` is ``{name, bbox}`` with bbox [west, south, east, north] in degrees."""
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Region footprint takes no {', '.join(unknown)}.")
    bbox = config.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int | float) for v in bbox)):
        raise ValueError("Region footprint needs bbox [west, south, east, north].")
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError(f"Region footprint bbox {bbox} is not [west, south, east, north] in degrees.")


def run(config: dict[str, Any]) -> str:
    """Write the footprint GeoTIFF of ``config["bbox"]``; returns its path (under the Earth Engine export folder)."""
    import rasterio
    from rasterio.transform import from_bounds

    from .gee_export import EXPORT_DIR

    check_config(config)
    west, south, east, north = (float(v) for v in config["bbox"])
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"region_footprint_{uuid.uuid4().hex[:8]}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, 2, 2),
    ) as dst:
        dst.write(np.zeros((1, 2, 2), np.uint8))
        dst.set_band_description(1, "footprint")
    return str(path)
