"""Google Earth Engine integration for MODIS data fetching."""

import logging
from typing import Any

import ee

from .gee_common import (
    calculate_index,
    create_geometry,
    format_error_response,
    format_tile_response,
    get_index_vis_params,
    initialize_gee,
)

logger = logging.getLogger(__name__)

# MODIS band mapping for indices
MODIS_BAND_MAPPING = {
    "red": "sur_refl_b01",  # Band 1: 620-670nm
    "nir": "sur_refl_b02",  # Band 2: 841-876nm
    "blue": "sur_refl_b03",  # Band 3: 459-479nm
    "green": "sur_refl_b04",  # Band 4: 545-565nm
}


def fetch_modis_composite(
    bbox: list[float],
    start_date: str,
    end_date: str,
    bands: list[str] | None = None,
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch MODIS composite imagery for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        bands: List of band names to include (default: RGB bands)
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        Dictionary with tile URL, metadata, and image info
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    try:
        # Default to RGB bands if not specified
        if bands is None:
            bands = ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"]  # R, G, B

        # Create geometry from bbox
        geometry = create_geometry(bbox)

        # Load MODIS Terra Surface Reflectance 8-Day composite
        # Note: MODIS 8-day composites may not align with exact dates
        # Expand date range by 16 days (2 composites) to ensure coverage
        from datetime import datetime, timedelta

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Expand by 8 days before and after to capture at least one composite
        expanded_start = (start_dt - timedelta(days=8)).strftime("%Y-%m-%d")
        expanded_end = (end_dt + timedelta(days=8)).strftime("%Y-%m-%d")

        logger.info(f"MODIS date range expanded: {expanded_start} to {expanded_end}")

        collection = (
            ee.ImageCollection("MODIS/006/MOD09A1").filterBounds(geometry).filterDate(expanded_start, expanded_end)
            # MODIS uses different cloud property - state_1km for QA
            # For simplicity, we'll just use basic filtering
        )

        # Get collection size
        count = collection.size().getInfo()
        logger.info(f"Found {count} MODIS images matching criteria")

        if count == 0:
            return format_error_response("No images found matching the specified criteria")

        # Create median composite
        composite = collection.median().clip(geometry)

        # Select requested bands
        selected = composite.select(bands)

        # Scale MODIS reflectance values (0-10000 to 0-1000 for visualization)
        selected = selected.multiply(0.1)

        # Generate visualization parameters
        # For RGB: bands order should be Red, Green, Blue
        if len(bands) >= 3:
            vis_bands = bands[:3]
            vis_params = {"min": 0, "max": 1000, "bands": vis_bands}
        else:
            vis_params = {
                "min": 0,
                "max": 1000,
            }

        # Get tile URL for MapLibre
        map_id = selected.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format

        result = format_tile_response(
            tile_url,
            {
                "image_count": count,
                "bbox": bbox,
                "start_date": start_date,
                "end_date": end_date,
                "cloud_cover_max": cloud_cover_max,
                "bands": bands,
                "vis_params": vis_params,
                "data_type": "composite",
                "dataset": "MODIS Terra/Aqua",
            },
        )
        if return_image:
            result["image"] = selected
        return result

    except Exception as e:
        logger.error(f"Error fetching MODIS data: {e}")
        return format_error_response(str(e))


def fetch_modis_indices(
    bbox: list[float],
    start_date: str,
    end_date: str,
    index_type: str = "ndvi",
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch MODIS spectral index (NDVI, NDWI) for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        index_type: Spectral index type ('ndvi', 'ndwi')
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        Dictionary with tile URL, metadata, and index info
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    # MODIS doesn't have SWIR in the 8-day composite used for NDSI
    if index_type not in ["ndvi", "ndwi"]:
        return format_error_response(f"MODIS only supports 'ndvi' and 'ndwi' indices (got '{index_type}')")

    try:
        # Create geometry from bbox
        geometry = create_geometry(bbox)

        # Load MODIS collection with expanded date range
        from datetime import datetime, timedelta

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Expand by 8 days before and after to capture at least one composite
        expanded_start = (start_dt - timedelta(days=8)).strftime("%Y-%m-%d")
        expanded_end = (end_dt + timedelta(days=8)).strftime("%Y-%m-%d")

        logger.info(f"MODIS index date range expanded: {expanded_start} to {expanded_end}")

        collection = (
            ee.ImageCollection("MODIS/006/MOD09A1").filterBounds(geometry).filterDate(expanded_start, expanded_end)
        )

        count = collection.size().getInfo()
        if count == 0:
            return format_error_response("No images found matching the specified criteria")

        # Calculate spectral index
        def calculate_modis_index(image: Any) -> Any:
            return calculate_index(image, index_type, MODIS_BAND_MAPPING)

        # Apply index calculation and create median composite
        index_collection = collection.map(calculate_modis_index)
        index_composite = index_collection.median().clip(geometry)

        # Get visualization parameters
        vis_params = get_index_vis_params(index_type)

        # Get tile URL
        map_id = index_composite.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format

        result = format_tile_response(
            tile_url,
            {
                "index_type": index_type,
                "image_count": count,
                "bbox": bbox,
                "start_date": start_date,
                "end_date": end_date,
                "cloud_cover_max": cloud_cover_max,
                "vis_params": vis_params,
                "data_type": "index",
                "dataset": "MODIS Terra/Aqua",
            },
        )
        if return_image:
            result["image"] = index_composite
        return result

    except Exception as e:
        logger.error(f"Error fetching MODIS index: {e}")
        return format_error_response(str(e))
