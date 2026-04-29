"""Google Earth Engine integration for Landsat 8/9 data fetching."""

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

# Landsat 8/9 band mapping for indices
LANDSAT_BAND_MAPPING = {
    "blue": "SR_B2",  # Band 2: 450-515nm
    "green": "SR_B3",  # Band 3: 525-600nm
    "red": "SR_B4",  # Band 4: 630-680nm
    "nir": "SR_B5",  # Band 5: 845-885nm
    "swir": "SR_B6",  # Band 6: 1560-1660nm (SWIR1)
}


def fetch_landsat_composite(
    bbox: list[float],
    start_date: str,
    end_date: str,
    bands: list[str] | None = None,
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch Landsat 8/9 composite imagery for a bounding box.

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
            bands = ["SR_B4", "SR_B3", "SR_B2"]  # R, G, B

        # Create geometry from bbox
        geometry = create_geometry(bbox)

        # Load Landsat 8 Collection 2 Tier 1 Level-2 (Surface Reflectance)
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))
        )

        # Get collection size
        count = collection.size().getInfo()
        logger.info(f"Found {count} Landsat 8/9 images matching criteria")

        if count == 0:
            return format_error_response("No images found matching the specified criteria")

        # Apply scaling factors for Landsat Collection 2 Surface Reflectance
        # Scale factor: 0.0000275, Offset: -0.2
        def apply_landsat_scaling(image: Any) -> Any:
            optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
            return image.addBands(optical_bands, None, True)

        # Apply scaling and create median composite
        collection = collection.map(apply_landsat_scaling)
        composite = collection.median().clip(geometry)

        # Select requested bands
        selected = composite.select(bands)

        # Scale for visualization (0-0.3 range to 0-3000)
        selected = selected.multiply(10000)

        # Generate visualization parameters
        if len(bands) >= 3:
            vis_bands = bands[:3]
            vis_params = {"min": 0, "max": 3000, "bands": vis_bands}
        else:
            vis_params = {
                "min": 0,
                "max": 3000,
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
                "dataset": "Landsat 8/9",
            },
        )
        if return_image:
            result["image"] = selected
        return result

    except Exception as e:
        logger.error(f"Error fetching Landsat data: {e}")
        return format_error_response(str(e))


def fetch_landsat_indices(
    bbox: list[float],
    start_date: str,
    end_date: str,
    index_type: str = "ndvi",
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch Landsat 8/9 spectral index (NDVI, NDWI, NDSI) for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        index_type: Spectral index type ('ndvi', 'ndwi', 'ndsi')
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        Dictionary with tile URL, metadata, and index info
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    try:
        # Create geometry from bbox
        geometry = create_geometry(bbox)

        # Load Landsat collection
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))
        )

        count = collection.size().getInfo()
        if count == 0:
            return format_error_response("No images found matching the specified criteria")

        # Apply scaling factors
        def apply_landsat_scaling(image: Any) -> Any:
            optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
            return image.addBands(optical_bands, None, True)

        collection = collection.map(apply_landsat_scaling)

        # Calculate spectral index
        def calculate_landsat_index(image: Any) -> Any:
            return calculate_index(image, index_type, LANDSAT_BAND_MAPPING)

        # Apply index calculation and create median composite
        index_collection = collection.map(calculate_landsat_index)
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
                "dataset": "Landsat 8/9",
            },
        )
        if return_image:
            result["image"] = index_composite
        return result

    except Exception as e:
        logger.error(f"Error fetching Landsat index: {e}")
        return format_error_response(str(e))
