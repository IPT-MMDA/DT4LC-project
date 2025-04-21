"""
Visualization utilities for the Cognitive Digital Twin Framework

This module provides functions for loading, processing, and enhancing
satellite imagery and other raster data for visualization.
"""

import io
import rasterio
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import cast
from matplotlib.figure import Figure
from rasterio.transform import Affine

from cognitive_ui.config import VIZ_NO_DATA, VIZ_NO_DATA_FLOAT, VIZ_PERCENTILES


def load_raster(file_path: Path, crop: tuple[int, int] | None = None) -> NDArray[np.float32]:
    """
    Load raster data from a file

    Args:
        file_path: Path to the raster file
        crop: Optional tuple specifying crop dimensions (height, width)

    Returns:
        Numpy array containing the raster data (bands, height, width)
    """
    try:
        with rasterio.open(file_path) as src:
            # Read all bands
            raster = src.read().astype(np.float32)

            # Replace no-data values with NaN
            if src.nodata is not None:
                raster[raster == src.nodata] = VIZ_NO_DATA

            # Apply crop if specified
            if crop:
                raster = raster[:, -crop[0] :, -crop[1] :]

            return cast(NDArray[np.float32], raster)
    except Exception:
        # Fallback to a simpler method if rasterio fails
        try:
            from PIL import Image

            img = Image.open(file_path)
            raster = np.array(img, dtype=np.float32).transpose(2, 0, 1)

            if crop:
                raster = raster[:, -crop[0] :, -crop[1] :]

            return cast(NDArray[np.float32], raster)
        except Exception as e:
            print(f"Error loading raster: {e}")
            return cast(NDArray[np.float32], np.zeros((3, 10, 10), dtype=np.float32))


def enhance_raster_for_visualization(
    raster: NDArray[np.float32],
    rgb_bands: tuple[int, int, int] = (2, 1, 0),
    ref_img: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """
    Enhance a satellite imagery raster for visualization by combining RGB bands
    with appropriate contrast enhancement.

    Args:
        raster: Satellite imagery raster data (bands, height, width)
        rgb_bands: Tuple of (red, green, blue) band indices to use for RGB visualization
        ref_img: Optional reference image for normalization

    Returns:
        Visualization-ready RGB image array (height, width, 3)
    """
    if raster is None or raster.size == 0:
        return cast(NDArray[np.float32], np.zeros((300, 300, 3), dtype=np.float32))

    # Use reference image for normalization if provided, otherwise use input raster
    if ref_img is None:
        ref_img = raster

    # Get the specified RGB bands
    if raster.ndim == 3 and raster.shape[0] >= 3:
        r_idx, g_idx, b_idx = rgb_bands
        if max(r_idx, g_idx, b_idx) < raster.shape[0]:
            r_band = raster[r_idx]
            g_band = raster[g_idx]
            b_band = raster[b_idx]
        else:
            # Fallback to first three bands if specified indices are out of range
            r_band, g_band, b_band = raster[:3]
    else:
        # Create grayscale image if fewer than 3 bands
        gray = raster[0] if raster.ndim == 3 else raster
        r_band = g_band = b_band = gray

    # Stack into RGB
    rgb = np.stack([r_band, g_band, b_band], axis=-1)

    # Handle potential NaN or infinite values
    rgb = np.nan_to_num(rgb, nan=VIZ_NO_DATA_FLOAT, posinf=1.0, neginf=0.0)

    # Identify no-data regions
    no_data_mask = np.isclose(rgb, VIZ_NO_DATA) | np.isclose(rgb, VIZ_NO_DATA_FLOAT)
    no_data_mask = np.any(no_data_mask, axis=-1)

    # Apply percentile-based contrast enhancement
    for i in range(3):
        # Skip if the band is only no-data values
        if np.all(no_data_mask):
            continue

        # Get values for contrast calculation (excluding no-data values)
        valid_values = rgb[~no_data_mask, i]
        if valid_values.size == 0:
            continue

        min_val, max_val = np.percentile(valid_values, VIZ_PERCENTILES)

        # Avoid division by zero
        if max_val > min_val:
            # Apply contrast enhancement
            rgb[:, :, i] = np.clip((rgb[:, :, i] - min_val) / (max_val - min_val), 0, 1)
        else:
            rgb[:, :, i] = 0

    # Set no-data areas to black
    if np.any(no_data_mask):
        rgb[no_data_mask] = 0

    return cast(NDArray[np.float32], rgb)


def plot_to_image(fig: Figure) -> io.BytesIO:
    """Convert a matplotlib figure to an image buffer.

    Args:
        fig: The matplotlib figure to convert

    Returns:
        BytesIO object containing the image data
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return buf


# HACK: This is a hack to save a numpy array as a GeoTIFF file.
def save_array_as_geotiff(array: NDArray[np.float32], output_path: str) -> None:
    """Save a numpy array as a GeoTIFF file.

    Args:
        array: Numpy array to save
        output_path: Path to save the GeoTIFF file
    """
    # Create a simple affine transform (identity)
    transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # Get array dimensions
    if len(array.shape) == 3:
        # Multi-band image (bands, height, width)
        bands, height, width = array.shape
    else:
        # Single-band image (height, width)
        bands = 1
        height, width = array.shape

    # Create the new file
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=array.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        if bands == 1:
            dst.write(array, 1)
        else:
            for i in range(bands):
                dst.write(array[i], i + 1)
