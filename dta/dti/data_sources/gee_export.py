"""Google Earth Engine export utilities for AI analysis.

This module provides functions to export GEE images as GeoTIFF files
that can be used by AI agents for land cover analysis.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import ee
import requests

from dta.config import CACHE_PATH

logger = logging.getLogger(__name__)

# Export directory for GeoTIFF files
EXPORT_DIR = CACHE_PATH / "gee_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum size for direct download (32MB limit from GEE)
MAX_DIRECT_DOWNLOAD_SIZE = 32 * 1024 * 1024


def export_gee_image_to_geotiff(
    image: ee.Image,
    bbox: list[float],
    bands: list[str] | None,
    scale: int,
    filename: str | None = None,
) -> dict[str, Any]:
    """Export GEE image to local GeoTIFF file using direct download.

    This method works for small regions (<32MB). For larger regions,
    use export_large_gee_image() instead.

    Args:
        image: Earth Engine Image to export
        bbox: Bounding box [minX, minY, maxX, maxY] in EPSG:4326
        bands: List of band IDs to export (None = all bands)
        scale: Resolution in meters (e.g., 10 for Sentinel-2)
        filename: Output filename without extension (auto-generated if None)

    Returns:
        Dictionary with status and file path:
        {
            'status': 'completed' | 'failed',
            'file_path': str,
            'size_bytes': int,
            'error': str (if failed)
        }
    """
    try:
        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"gee_export_{timestamp}"

        # Create region geometry
        region = ee.Geometry.Rectangle(bbox)

        # Select specific bands if provided
        export_image = image.select(bands) if bands else image

        logger.info(
            f"Exporting GEE image to GeoTIFF: {filename}, "
            f"bands={bands}, scale={scale}m"
        )

        # Get download URL
        url = export_image.getDownloadURL(
            {
                "region": region,
                "scale": scale,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
        )

        # Download the file
        logger.info(f"Downloading from GEE: {url[:100]}...")
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        # Check size
        size_bytes = len(response.content)
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"Downloaded {size_mb:.2f} MB")

        if size_bytes > MAX_DIRECT_DOWNLOAD_SIZE:
            logger.warning(
                f"File size ({size_mb:.2f} MB) exceeds direct download limit. "
                "Consider using export_large_gee_image() for async export."
            )

        # Save to local storage
        output_path = EXPORT_DIR / f"{filename}.tif"
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Exported GEE image to {output_path}")

        return {
            "status": "completed",
            "file_path": str(output_path),
            "size_bytes": size_bytes,
        }

    except requests.exceptions.Timeout:
        error_msg = "Download timed out (>5 minutes). Try a smaller region or use async export."
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

    except requests.exceptions.RequestException as e:
        error_msg = f"Download failed: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

    except Exception as e:
        error_msg = f"Export failed: {str(e)}"
        logger.exception(error_msg)
        return {"status": "failed", "error": error_msg}


def export_large_gee_image(
    image: ee.Image,
    bbox: list[float],
    bands: list[str] | None,
    scale: int,
    filename: str | None = None,
) -> dict[str, Any]:
    """Export large GEE image using batch task (async export to Google Drive).

    Use this for regions larger than 32MB. The export runs asynchronously
    in Google's servers and must be monitored using check_export_task_status().

    Args:
        image: Earth Engine Image to export
        bbox: Bounding box [minX, minY, maxX, maxY]
        bands: List of band IDs to export
        scale: Resolution in meters
        filename: Output filename without extension

    Returns:
        Dictionary with task information:
        {
            'status': 'running' | 'failed',
            'task_id': str,
            'description': str,
            'error': str (if failed)
        }
    """
    try:
        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"gee_export_{timestamp}"

        # Create region geometry
        region = ee.Geometry.Rectangle(bbox)

        # Select specific bands if provided
        export_image = image.select(bands) if bands else image

        logger.info(
            f"Starting async GEE export task: {filename}, "
            f"bands={bands}, scale={scale}m"
        )

        # Start batch export task to Google Drive
        task = ee.batch.Export.image.toDrive(
            image=export_image,
            description=filename,
            folder="DT4LC_Exports",
            fileNamePrefix=filename,
            scale=scale,
            region=region,
            maxPixels=1e9,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )

        task.start()

        logger.info(f"Started export task: {task.id}")

        return {
            "status": "running",
            "task_id": task.id,
            "description": filename,
        }

    except Exception as e:
        error_msg = f"Failed to start export task: {str(e)}"
        logger.exception(error_msg)
        return {"status": "failed", "error": error_msg}


def check_export_task_status(task_id: str) -> dict[str, Any]:
    """Check status of async GEE export task.

    Args:
        task_id: Task ID returned from export_large_gee_image()

    Returns:
        Dictionary with current status:
        {
            'status': 'running' | 'completed' | 'failed' | 'not_found',
            'progress': float (0-1),
            'error_message': str (if failed),
            'output_url': str (if completed)
        }
    """
    try:
        # Get all tasks and find the matching one
        task_list = ee.batch.Task.list()
        task = next((t for t in task_list if t.id == task_id), None)

        if not task:
            logger.warning(f"Task not found: {task_id}")
            return {"status": "not_found"}

        # Get task status
        status = task.status()
        state = status.get("state", "UNKNOWN").lower()

        result: dict[str, Any] = {
            "status": state,
            "progress": status.get("progress", 0) / 100.0,  # Convert to 0-1
        }

        # Add error message if failed
        if state in ["failed", "cancel_requested", "cancelled"]:
            result["error_message"] = status.get("error_message", "Unknown error")

        # Add output URL if completed
        if state == "completed":
            # Note: For Drive exports, users must manually download from their Drive
            result["output_url"] = status.get(
                "destination_uris", ["Manual download from Google Drive"]
            )[0]

        logger.info(f"Task {task_id} status: {state} ({result.get('progress', 0):.1%})")

        return result

    except Exception as e:
        error_msg = f"Failed to check task status: {str(e)}"
        logger.exception(error_msg)
        return {"status": "error", "error": error_msg}


def estimate_export_size(
    bbox: list[float], scale: int, num_bands: int = 3
) -> dict[str, Any]:
    """Estimate the export file size to determine if direct download is feasible.

    Args:
        bbox: Bounding box [minX, minY, maxX, maxY] in degrees
        scale: Resolution in meters
        num_bands: Number of bands to export

    Returns:
        Dictionary with size estimate:
        {
            'width_pixels': int,
            'height_pixels': int,
            'total_pixels': int,
            'estimated_size_mb': float,
            'can_use_direct_download': bool
        }
    """
    # Calculate approximate dimensions
    # Rough conversion: 1 degree ≈ 111km at equator
    width_deg = bbox[2] - bbox[0]
    height_deg = bbox[3] - bbox[1]

    width_m = width_deg * 111000  # meters
    height_m = height_deg * 111000

    width_pixels = int(width_m / scale)
    height_pixels = int(height_m / scale)
    total_pixels = width_pixels * height_pixels

    # Estimate size (4 bytes per pixel for Float32, multiply by bands)
    estimated_bytes = total_pixels * num_bands * 4
    estimated_mb = estimated_bytes / (1024 * 1024)

    can_use_direct = estimated_mb < 30  # Leave some margin

    logger.info(
        f"Export size estimate: {width_pixels}x{height_pixels} pixels "
        f"({estimated_mb:.2f} MB)"
    )

    return {
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
        "total_pixels": total_pixels,
        "estimated_size_mb": estimated_mb,
        "can_use_direct_download": can_use_direct,
    }


def create_attachment_from_export(
    file_path: str,
    layer_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create attachment object from exported GeoTIFF for chat/AI analysis.

    Args:
        file_path: Path to the exported GeoTIFF file
        layer_name: Original layer name from map
        metadata: Additional metadata (dataset, bands, dates, etc.)

    Returns:
        Attachment dictionary compatible with chat/job system
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"Export file not found: {file_path}")

    attachment_id = str(uuid.uuid4())

    attachment = {
        "id": attachment_id,
        "filename": f"{layer_name}.tif",
        "path": str(file_path),
        "mime_type": "image/tiff",
        "size_bytes": file_path_obj.stat().st_size,
        "source": "gee-export",
        "metadata": metadata or {},
    }

    logger.info(f"Created attachment for export: {attachment_id}")

    return attachment
