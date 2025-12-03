"""Raster IO utilities for DT4LC.

Shared utilities for loading and processing raster data (GeoTIFF, etc.).
Used by models and algorithms to ensure consistent data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RasterData:
    """Container for loaded raster data with metadata."""

    image: np.ndarray  # HxWxC uint8 RGB image
    crs: Any  # Coordinate reference system
    transform: Any  # Affine transform
    meta: dict[str, Any]  # Full rasterio metadata
    original_shape: tuple[int, ...]  # Original shape before processing
    original_dtype: np.dtype  # Original data type


def load_raster_as_rgb(
    raster_path: str | Path,
    bands: list[int] | None = None,
    nodata_fill: float = 0.0,
) -> RasterData:
    """Load a raster file and convert to RGB uint8 image.

    Handles:
    - Multi-band GeoTIFFs (selects RGB bands)
    - NaN/nodata values (replaces with fill value)
    - High bit-depth normalization (16-bit to 8-bit)
    - Various CRS and transforms

    Args:
        raster_path: Path to GeoTIFF or other rasterio-supported format
        bands: Band indices to use as RGB (1-indexed). Default: [3, 2, 1] for Sentinel-2
        nodata_fill: Value to use for nodata pixels. Default: 0.0

    Returns:
        RasterData with uint8 RGB image and metadata

    Raises:
        FileNotFoundError: If raster file doesn't exist
        ValueError: If raster has insufficient bands
    """
    import rasterio

    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    # Default to Sentinel-2 style RGB bands (1-indexed: B4=Red, B3=Green, B2=Blue)
    if bands is None:
        bands = [3, 2, 1]

    with rasterio.open(raster_path) as src:
        # Validate band count
        if src.count < max(bands):
            raise ValueError(
                f"Raster has {src.count} bands, but band {max(bands)} was requested. Available bands: 1-{src.count}"
            )

        # Read bands
        band_arrays = [src.read(b) for b in bands]
        original_dtype = band_arrays[0].dtype
        original_shape = (src.count, src.height, src.width)

        # Stack to HxWxC
        image = np.stack(band_arrays, axis=-1).astype(np.float32)

        # Handle NaN/nodata values
        image = np.nan_to_num(image, nan=nodata_fill, posinf=nodata_fill, neginf=nodata_fill)

        # Handle rasterio nodata value if set
        if src.nodata is not None:
            image[image == src.nodata] = nodata_fill

        # Normalize to 0-255 range
        max_val = image.max()
        if max_val > 255:
            # High bit-depth (16-bit, etc.) - scale to 8-bit
            if max_val > 0:
                image = ((image / max_val) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        elif max_val > 0:
            # Already 8-bit range but might be float
            image = image.astype(np.uint8)
        else:
            # All zeros
            image = np.zeros_like(image, dtype=np.uint8)

        return RasterData(
            image=image,
            crs=src.crs,
            transform=src.transform,
            meta=src.meta.copy(),
            original_shape=original_shape,
            original_dtype=original_dtype,
        )


def validate_geotiff(raster_path: str | Path) -> tuple[bool, str]:
    """Validate that a file is a readable GeoTIFF.

    Args:
        raster_path: Path to file

    Returns:
        Tuple of (is_valid, error_message)
    """
    import rasterio

    path = Path(raster_path)

    if not path.exists():
        return False, f"File not found: {raster_path}"

    suffix = path.suffix.lower()
    if suffix not in (".tif", ".tiff", ".geotiff"):
        return False, f"Unsupported format: {suffix}. Expected GeoTIFF (.tif/.tiff)"

    try:
        with rasterio.open(raster_path) as src:
            if src.count < 1:
                return False, "Raster has no bands"
            if src.crs is None:
                logger.warning("Raster has no CRS - results may have incorrect georeferencing")
        return True, ""
    except Exception as e:
        return False, f"Cannot read raster: {e}"
