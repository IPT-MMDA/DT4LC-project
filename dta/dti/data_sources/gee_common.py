"""Common utilities for Google Earth Engine data fetching."""
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
        project_id = os.getenv('GEE_PROJECT_ID')
        if not project_id:
            logger.error(
                "GEE_PROJECT_ID environment variable not set. "
                "Please set it to your Google Cloud Project ID. "
                "See: https://developers.google.com/earth-engine/guides/python_install"
            )
            return False

        # Try service account authentication first
        service_account_key = os.getenv('GEE_SERVICE_ACCOUNT_KEY')
        if service_account_key and Path(service_account_key).exists():
            credentials = ee.ServiceAccountCredentials(
                email=None,  # Will be read from key file
                key_file=service_account_key
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


def create_geometry(bbox: list[float]) -> ee.Geometry:
    """Create GEE geometry from bounding box.

    Args:
        bbox: Bounding box as [minX, minY, maxX, maxY] (WGS84)

    Returns:
        ee.Geometry.Rectangle
    """
    return ee.Geometry.Rectangle(bbox)


def apply_cloud_filter(
    collection: ee.ImageCollection,
    cloud_property: str,
    max_cloud: float
) -> ee.ImageCollection:
    """Apply cloud cover filter to image collection.

    Args:
        collection: Image collection to filter
        cloud_property: Property name for cloud cover (e.g., 'CLOUDY_PIXEL_PERCENTAGE')
        max_cloud: Maximum cloud cover percentage (0-100)

    Returns:
        Filtered image collection
    """
    return collection.filter(ee.Filter.lt(cloud_property, max_cloud))


def calculate_index(
    image: ee.Image,
    index_type: str,
    band_mapping: dict[str, str]
) -> ee.Image:
    """Calculate spectral index from image.

    Args:
        image: Input image
        index_type: Index type ('ndvi', 'ndwi', 'ndsi')
        band_mapping: Mapping of {band_role: band_name}
                      e.g., {'nir': 'B8', 'red': 'B4'} for Sentinel-2

    Returns:
        Calculated index as single-band image named 'index'
    """
    if index_type == 'ndvi':
        # NDVI = (NIR - Red) / (NIR + Red)
        return image.normalizedDifference([
            band_mapping['nir'],
            band_mapping['red']
        ]).rename('index')

    elif index_type == 'ndwi':
        # NDWI = (Green - NIR) / (Green + NIR)
        return image.normalizedDifference([
            band_mapping['green'],
            band_mapping['nir']
        ]).rename('index')

    elif index_type == 'ndsi':
        # NDSI = (Green - SWIR) / (Green + SWIR)
        return image.normalizedDifference([
            band_mapping['green'],
            band_mapping['swir']
        ]).rename('index')

    else:
        raise ValueError(f"Unknown index type: {index_type}")


def get_index_vis_params(index_type: str) -> dict[str, Any]:
    """Get visualization parameters for spectral index.

    Args:
        index_type: Index type ('ndvi', 'ndwi', 'ndsi')

    Returns:
        Dictionary with visualization parameters (min, max, palette)
    """
    if index_type == 'ndvi':
        return {
            'min': -0.2,
            'max': 0.8,
            'palette': ['#d73027', '#fee08b', '#d9ef8b', '#66bd63', '#1a9850']
        }
    elif index_type == 'ndwi':
        return {
            'min': -0.3,
            'max': 0.3,
            'palette': ['#f7f7f7', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
        }
    elif index_type == 'ndsi':
        return {
            'min': -0.2,
            'max': 0.6,
            'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
        }
    else:
        return {'min': -1, 'max': 1}


def format_tile_response(
    tile_url: str,
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """Format standardized tile response.

    Args:
        tile_url: Tile URL from GEE
        metadata: Additional metadata to include

    Returns:
        Standardized response dictionary
    """
    return {
        'ok': True,
        'tile_url': tile_url,
        **metadata
    }


def format_error_response(error: str) -> dict[str, Any]:
    """Format standardized error response.

    Args:
        error: Error message

    Returns:
        Error response dictionary
    """
    return {
        'ok': False,
        'error': error
    }
