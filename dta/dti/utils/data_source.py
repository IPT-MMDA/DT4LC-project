"""Data source detection utilities.

Auto-detect satellite data source from raster metadata and resolution.
Used to configure algorithm parameters for different satellite platforms.
"""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio

logger = logging.getLogger(__name__)

# Resolution thresholds in meters for satellite detection
# Based on typical native resolutions:
# - Sentinel-2: 10m (visible/NIR), 20m (red edge/SWIR), 60m (cirrus)
# - Planet (PlanetScope): 3-4m
# - Maxar (WorldView): 0.3-0.5m (panchromatic), 1.2-2m (multispectral)
SENTINEL_MIN_RES = 8.0  # >= 8m is Sentinel-like
PLANET_MIN_RES = 2.0  # 2-8m is Planet-like
# < 2m is high-resolution (Maxar-like)


def detect_data_source(raster_path: str | Path) -> str:
    """Detect satellite data source from raster resolution and metadata.

    Uses pixel resolution as primary indicator:
    - >= 8m: Sentinel-2 or similar medium resolution
    - 2-8m: Planet or similar high resolution
    - < 2m: Maxar or similar very high resolution

    Args:
        raster_path: Path to GeoTIFF raster file

    Returns:
        Data source identifier: "sentinel", "planet", or "maxar"
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        logger.warning(f"Raster not found: {raster_path}, defaulting to 'sentinel'")
        return "sentinel"

    try:
        with rasterio.open(raster_path) as src:
            # Get pixel resolution (meters per pixel)
            # transform[0] is pixel width, transform[4] is pixel height (negative)
            res_x = abs(src.transform[0])
            res_y = abs(src.transform[4])
            resolution = (res_x + res_y) / 2  # Average resolution

            # Check CRS to ensure we're working in meters
            if src.crs and not src.crs.is_projected:
                # Geographic CRS (degrees) - estimate meters at equator
                # 1 degree ≈ 111,320 meters at equator
                resolution = resolution * 111320
                logger.debug(f"Geographic CRS detected, estimated resolution: {resolution:.2f}m")

            logger.debug(f"Detected resolution: {resolution:.2f}m for {raster_path.name}")

            # Classify by resolution
            if resolution >= SENTINEL_MIN_RES:
                source = "sentinel"
            elif resolution >= PLANET_MIN_RES:
                source = "planet"
            else:
                source = "maxar"

            logger.info(f"Detected data source '{source}' (resolution: {resolution:.2f}m) for {raster_path.name}")
            return source

    except Exception as e:
        logger.warning(f"Failed to detect data source for {raster_path}: {e}, defaulting to 'sentinel'")
        return "sentinel"


def get_filtering_thresholds(data_source: str) -> dict[str, int]:
    """Get filtering thresholds for a data source.

    Returns appropriate minimum_area and minimum_hole values for
    the Delineate-Anything model based on data source.

    Args:
        data_source: One of "sentinel", "planet", "maxar"

    Returns:
        Dict with minimum_area_m2 and minimum_hole_area_m2
    """
    thresholds = {
        "sentinel": {"minimum_area_m2": 2500, "minimum_hole_area_m2": 2500},
        "planet": {"minimum_area_m2": 1000, "minimum_hole_area_m2": 1000},
        "maxar": {"minimum_area_m2": 500, "minimum_hole_area_m2": 500},
    }
    return thresholds.get(data_source, thresholds["sentinel"])
