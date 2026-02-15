"""Bulk fetching logic for multiple datasets across pre/post periods."""
import logging
from typing import Any
from datetime import datetime

from .gee_sentinel2 import fetch_sentinel2_composite, fetch_sentinel2_indices
from .gee_modis import fetch_modis_composite, fetch_modis_indices
from .gee_landsat import fetch_landsat_composite, fetch_landsat_indices

logger = logging.getLogger(__name__)


def generate_layer_name(
    dataset: str,
    bands: list[str] | None,
    index_type: str | None,
    start_date: str,
    end_date: str,
    period: str
) -> str:
    """Generate descriptive layer name for bulk import.

    Args:
        dataset: Dataset name (e.g., 'Sentinel-2', 'MODIS', 'Landsat 8/9')
        bands: List of band IDs (None for indices)
        index_type: Index type (None for bands)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        period: Period label ('Pre' or 'Post')

    Returns:
        Formatted layer name
    """
    if index_type:
        # For indices: "Sentinel-2 - NDVI (2024-01-01 to 2024-01-31) - Pre"
        return f"{dataset} - {index_type.upper()} ({start_date} to {end_date}) - {period}"

    elif bands and len(bands) > 0:
        # For bands: "Sentinel-2 - B4,B3,B2 (2024-01-01 to 2024-01-31) - Pre"
        band_str = ','.join(bands) if len(bands) <= 4 else f"{len(bands)} bands"
        return f"{dataset} - {band_str} ({start_date} to {end_date}) - {period}"

    else:
        return f"{dataset} ({start_date} to {end_date}) - {period}"


def generate_layer_id(
    dataset_id: str,
    bands: list[str] | None,
    index_type: str | None,
    period: str
) -> str:
    """Generate unique layer ID.

    Args:
        dataset_id: Dataset ID (e.g., 'sentinel-2', 'modis', 'landsat-8')
        bands: List of band IDs
        index_type: Index type
        period: 'pre' or 'post'

    Returns:
        Unique layer ID
    """
    timestamp = int(datetime.now().timestamp() * 1000)

    if index_type:
        return f"gee-{dataset_id}-{index_type}-{period}-{timestamp}"
    elif bands:
        band_hash = '-'.join(sorted(bands))
        return f"gee-{dataset_id}-{band_hash}-{period}-{timestamp}"
    else:
        return f"gee-{dataset_id}-{period}-{timestamp}"


def bulk_fetch_data(
    dataset_id: str,
    bbox: list[float],
    bands: list[str],
    indices: list[str],
    pre_period: dict[str, str],
    post_period: dict[str, str],
    cloud_cover_max: float
) -> dict[str, Any]:
    """Fetch multiple bands and indices for both pre/post periods.

    Args:
        dataset_id: Dataset ID ('sentinel-2', 'modis', 'landsat-8')
        bbox: Bounding box [minX, minY, maxX, maxY]
        bands: List of band IDs to fetch
        indices: List of indices to fetch ('ndvi', 'ndwi', 'ndsi')
        pre_period: {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        post_period: {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        cloud_cover_max: Maximum cloud cover percentage

    Returns:
        Dictionary with list of layer metadata for bulk import
    """
    layers = []
    errors = []

    # Dataset name mapping for display
    dataset_names = {
        'sentinel-2': 'Sentinel-2',
        'modis': 'MODIS Terra/Aqua',
        'landsat-8': 'Landsat 8/9'
    }
    dataset_name = dataset_names.get(dataset_id, dataset_id)

    # Function mapping for each dataset
    fetch_functions = {
        'sentinel-2': {
            'composite': fetch_sentinel2_composite,
            'index': fetch_sentinel2_indices
        },
        'modis': {
            'composite': fetch_modis_composite,
            'index': fetch_modis_indices
        },
        'landsat-8': {
            'composite': fetch_landsat_composite,
            'index': fetch_landsat_indices
        }
    }

    if dataset_id not in fetch_functions:
        return {
            'ok': False,
            'error': f"Unknown dataset: {dataset_id}",
            'layers': []
        }

    funcs = fetch_functions[dataset_id]

    # Process both periods
    for period_name, period_dates in [('Pre', pre_period), ('Post', post_period)]:
        period_lower = period_name.lower()

        # Fetch band composite if bands are selected
        if bands and len(bands) > 0:
            try:
                logger.info(f"Fetching {dataset_name} composite for {period_name} period: {bands}")

                result = funcs['composite'](
                    bbox=bbox,
                    start_date=period_dates['start'],
                    end_date=period_dates['end'],
                    bands=bands,
                    cloud_cover_max=cloud_cover_max
                )

                if result.get('ok'):
                    layer_name = generate_layer_name(
                        dataset_name,
                        bands,
                        None,
                        period_dates['start'],
                        period_dates['end'],
                        period_name
                    )

                    layer_id = generate_layer_id(
                        dataset_id,
                        bands,
                        None,
                        period_lower
                    )

                    layers.append({
                        'tile_url': result['tile_url'],
                        'layer_name': layer_name,
                        'layer_id': layer_id,
                        'period': period_lower,
                        'data_type': 'bands',
                        'bands': bands,
                        'image_count': result.get('image_count', 0),
                        'vis_params': result.get('vis_params', {})
                    })
                else:
                    error_msg = f"{dataset_name} composite ({period_name}): {result.get('error', 'Unknown error')}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            except Exception as e:
                error_msg = f"{dataset_name} composite ({period_name}): {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Fetch each spectral index separately
        for index_type in indices:
            try:
                logger.info(f"Fetching {dataset_name} {index_type.upper()} for {period_name} period")

                result = funcs['index'](
                    bbox=bbox,
                    start_date=period_dates['start'],
                    end_date=period_dates['end'],
                    index_type=index_type,
                    cloud_cover_max=cloud_cover_max
                )

                if result.get('ok'):
                    layer_name = generate_layer_name(
                        dataset_name,
                        None,
                        index_type,
                        period_dates['start'],
                        period_dates['end'],
                        period_name
                    )

                    layer_id = generate_layer_id(
                        dataset_id,
                        None,
                        index_type,
                        period_lower
                    )

                    layers.append({
                        'tile_url': result['tile_url'],
                        'layer_name': layer_name,
                        'layer_id': layer_id,
                        'period': period_lower,
                        'data_type': index_type,
                        'index_type': index_type,
                        'image_count': result.get('image_count', 0),
                        'vis_params': result.get('vis_params', {})
                    })
                else:
                    error_msg = f"{dataset_name} {index_type.upper()} ({period_name}): {result.get('error', 'Unknown error')}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            except Exception as e:
                error_msg = f"{dataset_name} {index_type.upper()} ({period_name}): {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

    # Return results
    if len(layers) == 0:
        return {
            'ok': False,
            'error': 'No layers could be fetched. ' + '; '.join(errors),
            'layers': [],
            'errors': errors
        }

    return {
        'ok': True,
        'layers': layers,
        'total_layers': len(layers),
        'errors': errors if errors else None
    }
