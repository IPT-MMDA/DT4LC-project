"""Google Earth Engine integration for Sentinel-2 data fetching."""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any

import ee

logger = logging.getLogger(__name__)

# GEE initialization state
_ee_initialized = False


def initialize_gee() -> bool:
    """Initialize Google Earth Engine API.

    Attempts to authenticate using:
    1. Service account credentials from GEE_SERVICE_ACCOUNT_KEY env var
    2. Default credentials (if already authenticated via gcloud)

    Requires GEE_PROJECT_ID environment variable to be set.

    Returns:
        True if initialization successful, False otherwise
    """
    global _ee_initialized

    if _ee_initialized:
        return True

    try:
        # Get project ID from environment (required)
        project_id = os.getenv("GEE_PROJECT_ID")
        if not project_id:
            logger.error(
                "GEE_PROJECT_ID environment variable not set. "
                "Please set it to your Google Cloud Project ID. "
                "See: https://developers.google.com/earth-engine/guides/python_install"
            )
            return False

        # Try service account authentication first
        service_account_key = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
        if service_account_key and Path(service_account_key).exists():
            credentials = ee.ServiceAccountCredentials(
                email=None,  # Will be read from key file
                key_file=service_account_key,
            )
            ee.Initialize(credentials, project=project_id)
            logger.info(f"GEE initialized with service account (project: {project_id})")
            _ee_initialized = True
            return True

        # Fall back to default authentication
        ee.Initialize(project=project_id)
        logger.info(f"GEE initialized with default credentials (project: {project_id})")
        _ee_initialized = True
        return True

    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        return False


def fetch_sentinel2_composite(
    bbox: list[float],
    start_date: str,
    end_date: str,
    bands: list[str] | None = None,
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch Sentinel-2 composite imagery for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        bands: List of band IDs to include (default: B4, B3, B2 for RGB)
        cloud_cover_max: Maximum cloud cover percentage (0-100)
        return_image: If True, include ee.Image object in response for export

    Returns:
        Dictionary with tile URL, metadata, and optionally image object
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    try:
        # Create geometry from bbox
        geometry = ee.Geometry.Rectangle(bbox)

        # Load Sentinel-2 Surface Reflectance collection
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        )

        # Get collection size
        count = collection.size().getInfo()
        logger.info(f"Found {count} Sentinel-2 images matching criteria")

        if count == 0:
            return {"ok": False, "error": "No images found matching the specified criteria", "image_count": 0}

        # Create median composite
        composite = collection.median().clip(geometry)

        # Select bands (default to RGB if not specified)
        selected_bands = bands if bands else ["B4", "B3", "B2"]
        band_image = composite.select(selected_bands)

        # Generate visualization parameters (use first 3 bands for RGB visualization)
        vis_bands = selected_bands[:3] if len(selected_bands) >= 3 else selected_bands
        vis_params = {"min": 0, "max": 3000, "bands": vis_bands}

        # Get tile URL for MapLibre
        map_id = band_image.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format

        # Get image properties
        first_image = ee.Image(collection.first())
        properties = first_image.toDictionary().getInfo()

        result = {
            "ok": True,
            "tile_url": tile_url,
            "image_count": count,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_max": cloud_cover_max,
            "bands": selected_bands,
            "vis_params": vis_params,
            "properties": {
                "system:time_start": properties.get("system:time_start"),
                "CLOUDY_PIXEL_PERCENTAGE": properties.get("CLOUDY_PIXEL_PERCENTAGE"),
                "SPACECRAFT_NAME": properties.get("SPACECRAFT_NAME"),
            },
        }

        # Include image object if requested (for export)
        if return_image:
            result["image"] = band_image

        return result

    except Exception as e:
        logger.error(f"Error fetching Sentinel-2 data: {e}")
        return {"ok": False, "error": str(e)}


def fetch_sentinel2_indices(
    bbox: list[float],
    start_date: str,
    end_date: str,
    index_type: str = "ndvi",
    cloud_cover_max: float = 20.0,
    return_image: bool = False,
) -> dict[str, Any]:
    """Fetch Sentinel-2 spectral index (NDVI, NDWI, NDSI) for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        index_type: Spectral index type ('ndvi', 'ndwi', 'ndsi')
        cloud_cover_max: Maximum cloud cover percentage (0-100)
        return_image: If True, include ee.Image object in response for export

    Returns:
        Dictionary with tile URL, metadata, index info, and optionally image object
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    try:
        # Create geometry from bbox
        geometry = ee.Geometry.Rectangle(bbox)

        # Load Sentinel-2 collection
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        )

        count = collection.size().getInfo()
        if count == 0:
            return {"ok": False, "error": "No images found matching the specified criteria", "image_count": 0}

        # Calculate spectral index based on type
        def calculate_index(image: Any) -> Any:
            if index_type == "ndvi":
                # NDVI = (NIR - Red) / (NIR + Red)
                return image.normalizedDifference(["B8", "B4"]).rename("index")
            elif index_type == "ndwi":
                # NDWI = (Green - NIR) / (Green + NIR)
                return image.normalizedDifference(["B3", "B8"]).rename("index")
            elif index_type == "ndsi":
                # NDSI = (Green - SWIR) / (Green + SWIR)
                return image.normalizedDifference(["B3", "B11"]).rename("index")
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        # Apply index calculation and create median composite
        index_collection = collection.map(calculate_index)
        index_composite = index_collection.median().clip(geometry)

        # Visualization parameters based on index type
        if index_type == "ndvi":
            vis_params = {"min": -0.2, "max": 0.8, "palette": ["#d73027", "#fee08b", "#d9ef8b", "#66bd63", "#1a9850"]}
        elif index_type == "ndwi":
            vis_params = {"min": -0.3, "max": 0.3, "palette": ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]}
        elif index_type == "ndsi":
            vis_params = {
                "min": -0.2,
                "max": 0.6,
                "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"],
            }
        else:
            vis_params = {"min": -1, "max": 1}

        # Get tile URL
        map_id = index_composite.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format

        result = {
            "ok": True,
            "tile_url": tile_url,
            "index_type": index_type,
            "image_count": count,
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
            "cloud_cover_max": cloud_cover_max,
            "vis_params": vis_params,
        }

        # Include image object if requested (for export)
        if return_image:
            result["image"] = index_composite

        return result

    except Exception as e:
        logger.error(f"Error fetching Sentinel-2 index: {e}")
        return {"ok": False, "error": str(e)}


def get_available_dates(
    bbox: list[float], start_date: str, end_date: str, cloud_cover_max: float = 20.0
) -> dict[str, Any]:
    """Get available Sentinel-2 acquisition dates for a bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cloud_cover_max: Maximum cloud cover percentage (0-100)

    Returns:
        Dictionary with available dates and image info
    """
    if not initialize_gee():
        raise RuntimeError("Failed to initialize Google Earth Engine")

    try:
        # Create geometry from bbox
        geometry = ee.Geometry.Rectangle(bbox)

        # Load Sentinel-2 collection
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        )

        # Get acquisition dates
        dates_list = collection.aggregate_array("system:time_start").getInfo()

        # Convert timestamps to dates
        dates = [datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") for timestamp in dates_list]

        return {
            "ok": True,
            "dates": dates,
            "count": len(dates),
            "bbox": bbox,
            "start_date": start_date,
            "end_date": end_date,
        }

    except Exception as e:
        logger.error(f"Error getting available dates: {e}")
        return {"ok": False, "error": str(e)}
