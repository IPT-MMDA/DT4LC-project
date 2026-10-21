"""Scale raster data to HLS (Harmonized Landsat Sentinel) format.

This preprocessor converts various input formats to HLS-compatible scale:
- 0-1 surface reflectance -> 0-10000 HLS scale
- Already HLS-scale data -> passthrough
- Raw DN values -> passthrough with int16 conversion

Used by models trained on HLS data (e.g., Prithvi foundation model).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# Prithvi-EO's HLS band set (config.json pretrained_cfg.bands):
# B02 blue, B03 green, B04 red, B05 NIR, B06 SWIR1, B07 SWIR2.
HLS_BAND_COUNT = 6


def run(input_path: str, output_dir: str) -> str:
    """Scale raster to HLS format.

    Args:
        input_path: Path to input raster file
        output_dir: Directory to write preprocessed file

    Returns:
        Path to preprocessed raster (or original if already compatible)

    Raises:
        ValueError: If the raster does not have HLS_BAND_COUNT bands.
    """
    try:
        with rasterio.open(input_path) as src:
            if src.count != HLS_BAND_COUNT:
                raise ValueError(
                    f"HLS input: expected {HLS_BAND_COUNT} bands (B02, B03, B04, B05, B06, B07), "
                    f"got {src.count} in {input_path}"
                )
            data = src.read()
            meta = src.meta.copy()

            # Check if data needs scaling (0-1 range -> HLS 0-10000 range)
            data_max = np.nanmax(data)
            data_min = np.nanmin(data)

            if data_max <= 1.0:
                # Data is 0-1 reflectance (small negatives are valid surface
                # reflectance, not HLS scale), scale to HLS
                logger.info(f"Scaling input from 0-1 to HLS scale (range: {data_min:.4f}-{data_max:.4f})")
                scaled_data = (data * 10000).astype(np.int16)
                meta.update(dtype="int16")
            elif data_max > 10000:
                # Large values - might be raw DN, just convert to int16
                logger.info(f"Input has large values (max={data_max:.0f}), converting to int16")
                scaled_data = np.clip(data, -32768, 32767).astype(np.int16)
                meta.update(dtype="int16")
            else:
                # Already in HLS-like range (0-10000)
                logger.info(f"Input appears to be in HLS scale (range: {data_min:.0f}-{data_max:.0f})")
                scaled_data = data.astype(np.int16)
                meta.update(dtype="int16")

            # Write preprocessed file
            output_path = Path(output_dir) / "hls_scaled_input.tif"
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(scaled_data)

            logger.info(f"HLS-scaled input saved to {output_path}")
            return str(output_path)

    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"HLS scaling failed, using original file: {e}")
        return input_path
