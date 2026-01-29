"""Microsoft Planetary Computer data source for satellite imagery.

This module provides access to Sentinel-2 and other satellite data
from Microsoft Planetary Computer with no size limits.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds

from dta.config import CACHE_PATH

logger = logging.getLogger(__name__)

# Export directory for GeoTIFF files
EXPORT_DIR = CACHE_PATH / "gee_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Planetary Computer STAC API
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Band name mapping: GEE -> Planetary Computer
# GEE uses B1-B12, Planetary Computer uses B01-B12 with leading zeros
BAND_NAME_MAPPING = {
    'B1': 'B01',
    'B2': 'B02',
    'B3': 'B03',
    'B4': 'B04',
    'B5': 'B05',
    'B6': 'B06',
    'B7': 'B07',
    'B8': 'B08',
    'B8A': 'B8A',
    'B9': 'B09',
    'B10': 'B10',
    'B11': 'B11',
    'B12': 'B12',
}


def normalize_band_name(band_name: str) -> str:
    """Convert GEE band name to Planetary Computer format.

    Args:
        band_name: Band name from GEE (e.g., 'B4', 'B3', 'B2')

    Returns:
        Planetary Computer band name (e.g., 'B04', 'B03', 'B02')
    """
    return BAND_NAME_MAPPING.get(band_name, band_name)


def export_sentinel2_from_mpc(
    bbox: list[float],
    start_date: str,
    end_date: str,
    bands: list[str],
    scale: int = 10,
    cloud_cover_max: float = 20.0,
    filename: str | None = None,
) -> dict[str, Any]:
    """Export Sentinel-2 data from Microsoft Planetary Computer.

    Args:
        bbox: Bounding box [minX, minY, maxX, maxY] in EPSG:4326
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        bands: List of band names ['B04', 'B03', 'B02']
        scale: Resolution in meters (10, 20, 60)
        cloud_cover_max: Maximum cloud cover percentage
        filename: Output filename without extension

    Returns:
        Dictionary with status and file path:
        {
            'status': 'completed' | 'failed',
            'file_path': str,
            'size_bytes': int,
            'error': str (if failed),
            'source': 'planetary-computer',
            'metadata': dict (scene info)
        }
    """
    try:
        logger.info(
            f"Fetching Sentinel-2 from Planetary Computer: "
            f"bbox={bbox}, dates={start_date}/{end_date}, bands={bands}, scale={scale}m"
        )

        # 1. Connect to Planetary Computer catalog
        catalog = pystac_client.Client.open(
            STAC_API_URL,
            modifier=pc.sign_inplace,  # Automatically sign URLs
        )

        # 2. Search for Sentinel-2 scenes
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        )

        items = list(search.items())
        if not items:
            error_msg = (
                f"No Sentinel-2 images found for criteria: "
                f"bbox={bbox}, dates={start_date}/{end_date}, cloud_cover<{cloud_cover_max}%"
            )
            logger.warning(error_msg)
            return {"status": "failed", "error": error_msg}

        # 3. Use most recent scene with least cloud cover
        items_sorted = sorted(items, key=lambda x: (
            -x.datetime.timestamp(),  # Most recent first
            x.properties.get("eo:cloud_cover", 100)  # Least cloudy
        ))
        item = items_sorted[0]

        scene_date = item.datetime.strftime("%Y-%m-%d")
        cloud_cover = item.properties.get("eo:cloud_cover", 0)
        scene_id = item.id

        logger.info(
            f"Selected scene: {scene_id}, date={scene_date}, "
            f"cloud_cover={cloud_cover:.1f}%"
        )

        # 4. Normalize band names (GEE -> Planetary Computer format)
        normalized_bands = [normalize_band_name(b) for b in bands]
        logger.info(f"Normalized bands: {bands} -> {normalized_bands}")

        # 5. Read and stack bands
        band_data_list = []
        output_profile = None
        output_transform = None
        output_crs = None

        for band_name in normalized_bands:
            if band_name not in item.assets:
                logger.warning(f"Band {band_name} not available, skipping")
                continue

            # Get signed URL
            asset = item.assets[band_name]
            band_url = asset.href

            logger.info(f"Reading band {band_name} from Planetary Computer")

            # Read band data
            with rasterio.open(band_url) as src:
                # Transform bbox from EPSG:4326 to the source CRS
                # bbox is [minX, minY, maxX, maxY] in WGS84
                bbox_in_src_crs = transform_bounds(
                    'EPSG:4326',
                    src.crs,
                    bbox[0], bbox[1], bbox[2], bbox[3]
                )

                # Calculate window for the transformed bbox
                window = from_bounds(*bbox_in_src_crs, transform=src.transform)

                # Ensure window has valid dimensions
                window = window.round_lengths()

                if window.width <= 0 or window.height <= 0:
                    error_msg = f"Invalid window dimensions for {band_name}: {window.width}x{window.height}. Bbox: {bbox}, Source CRS: {src.crs}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"Window for {band_name}: {window.width}x{window.height} at offset ({window.col_off}, {window.row_off})")

                # Read data for this window
                data = src.read(1, window=window)

                # Get transform for this window
                window_transform = src.window_transform(window)
                window_crs = src.crs

                # Handle resampling if needed
                if scale != src.res[0]:
                    # Calculate new dimensions based on target scale
                    height, width = data.shape
                    scale_factor = src.res[0] / scale
                    new_height = int(height * scale_factor)
                    new_width = int(width * scale_factor)

                    logger.info(f"Resampling {band_name} from {src.res[0]}m to {scale}m: {height}x{width} -> {new_height}x{new_width}")

                    # Create temporary dataset for resampling
                    data_resampled = np.empty((new_height, new_width), dtype=data.dtype)

                    # Calculate destination transform (same origin, different pixel size)
                    dst_transform = window_transform * window_transform.scale(
                        (width / new_width),
                        (height / new_height)
                    )

                    # Resample
                    reproject(
                        source=data,
                        destination=data_resampled,
                        src_transform=window_transform,
                        src_crs=window_crs,
                        dst_transform=dst_transform,
                        dst_crs=window_crs,
                        resampling=Resampling.bilinear
                    )

                    data = data_resampled
                    window_transform = dst_transform

                band_data_list.append(data)

                # Store metadata from first band
                if output_profile is None:
                    output_crs = window_crs
                    output_transform = window_transform

                    # Create output profile
                    output_profile = {
                        'driver': 'GTiff',
                        'dtype': data.dtype,
                        'width': data.shape[1],
                        'height': data.shape[0],
                        'count': len(normalized_bands),
                        'crs': output_crs,
                        'transform': output_transform,
                        'compress': 'lzw',
                        'tiled': True,
                        'blockxsize': 256,
                        'blockysize': 256,
                    }

                    logger.info(f"Output dimensions: {data.shape[0]}x{data.shape[1]}, bands: {len(normalized_bands)}")

        if not band_data_list:
            return {"status": "failed", "error": "No bands could be read"}

        # 6. Generate filename if not provided
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"mpc_s2_{scene_date}_{timestamp}"

        # 7. Stack bands and write output
        output_path = EXPORT_DIR / f"{filename}.tif"

        with rasterio.open(output_path, 'w', **output_profile) as dst:
            for i, data in enumerate(band_data_list, 1):
                dst.write(data, i)

        # 8. Get file size
        size_bytes = output_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        logger.info(
            f"Successfully exported from Planetary Computer: "
            f"{output_path} ({size_mb:.2f} MB)"
        )

        return {
            "status": "completed",
            "file_path": str(output_path),
            "size_bytes": size_bytes,
            "source": "planetary-computer",
            "metadata": {
                "scene_id": scene_id,
                "scene_date": scene_date,
                "cloud_cover": cloud_cover,
                "bands": bands,
                "resolution_m": scale,
            }
        }

    except Exception as e:
        error_msg = f"Planetary Computer export failed: {str(e)}"
        logger.exception(error_msg)
        return {"status": "failed", "error": error_msg}


def test_planetary_computer_access() -> bool:
    """Test if Planetary Computer is accessible.

    Returns:
        True if accessible, False otherwise
    """
    try:
        catalog = pystac_client.Client.open(STAC_API_URL)
        # Try a simple search
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[8.0, 46.7, 8.1, 46.8],
            datetime="2024-01-01/2024-01-02",
            max_items=1
        )
        list(search.items())
        logger.info("Planetary Computer access test: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Planetary Computer access test: FAILED - {e}")
        return False
