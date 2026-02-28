"""XYZ tile serving endpoint for GeoTIFF files."""

import io
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import mercantile
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from ..utils import apply_colormap, detect_data_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/tiles", tags=["tiles"])


@router.get("/{z}/{x}/{y}")  # type: ignore[misc]
async def get_tile(
    z: int, x: int, y: int, path: str = Query(..., description="Path to GeoTIFF file")
) -> StreamingResponse:
    """Serve GeoTIFF as map tiles in XYZ format.

    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
        path: Path to the GeoTIFF file

    Returns:
        PNG image tile
    """
    try:
        # Get tile bounds in Web Mercator (EPSG:3857)
        tile = mercantile.Tile(x, y, z)
        tile_bounds_wgs84 = mercantile.bounds(tile)

        # Open GeoTIFF
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"GeoTIFF file not found: {path}")

        # Detect data type from filename
        data_type = detect_data_type(file_path.name)

        with rasterio.open(str(file_path)) as src:
            # Transform tile bounds from WGS84 to source CRS
            src_bounds = transform_bounds(
                "EPSG:4326",
                src.crs,
                tile_bounds_wgs84.west,
                tile_bounds_wgs84.south,
                tile_bounds_wgs84.east,
                tile_bounds_wgs84.north,
            )

            # Read window from GeoTIFF
            window = from_bounds(*src_bounds, src.transform)

            # Check if window is valid (intersects with raster)
            if window.width <= 0 or window.height <= 0:
                # Return transparent tile for areas outside the raster
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                return StreamingResponse(img_bytes, media_type="image/png")

            # Read data based on band count
            if src.count >= 3:
                # RGB data - read first 3 bands
                data = src.read([1, 2, 3], window=window, out_shape=(3, 256, 256))
                data = np.transpose(data, (1, 2, 0))  # CHW to HWC

                # Handle nodata
                if src.nodata is not None:
                    mask = data[:, :, 0] == src.nodata
                    data[mask] = 0

                # Normalize to 0-255
                if data.max() > 255 or data.dtype != np.uint8:
                    data_finite = data[np.isfinite(data)]
                    if len(data_finite) > 0:
                        p2, p98 = np.percentile(data_finite, [2, 98])
                        data = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                    else:
                        data = np.zeros((256, 256, 3), dtype=np.uint8)

                img = Image.fromarray(data, mode="RGB")
            else:
                # Single band data - read first band
                data = src.read(1, window=window, out_shape=(256, 256))

                # Handle nodata values
                if src.nodata is not None:
                    nodata_mask = data == src.nodata
                    data = np.where(nodata_mask, np.nan, data)

                # Apply data-type-specific normalization and colormapping
                data_finite = data[np.isfinite(data)]

                if len(data_finite) == 0:
                    # All nodata - return transparent tile
                    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)
                    return StreamingResponse(img_bytes, media_type="image/png")

                # Normalize based on data type
                if data_type in ["ndvi", "ndwi", "ndsi"]:
                    # Spectral indices typically range from -1 to 1
                    # Clip to expected range
                    data_clipped = np.clip(data, -1, 1)
                    # Scale to 0-255
                    data_normalized = ((data_clipped + 1) / 2 * 255).astype(np.uint8)

                    # Apply colormap
                    if data_type == "ndvi":
                        # Green-yellow-red colormap for vegetation
                        rgb_data = apply_colormap(data_normalized, "RdYlGn")
                    elif data_type == "ndwi":
                        # Blue colormap for water
                        rgb_data = apply_colormap(data_normalized, "Blues")
                    elif data_type == "ndsi":
                        # Cool colormap for snow/ice
                        rgb_data = apply_colormap(data_normalized, "cool")
                    else:
                        rgb_data = apply_colormap(data_normalized, "viridis")

                    # Handle nodata in RGB
                    if src.nodata is not None:
                        rgb_data[nodata_mask] = [0, 0, 0]

                    img = Image.fromarray(rgb_data, mode="RGB")
                else:
                    # Generic single-band data - use percentile stretch
                    p2, p98 = np.percentile(data_finite, [2, 98])
                    data_normalized = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

                    # Apply viridis colormap for better visualization
                    rgb_data = apply_colormap(data_normalized, "viridis")

                    # Handle nodata
                    if src.nodata is not None:
                        rgb_data[nodata_mask] = [0, 0, 0]

                    img = Image.fromarray(rgb_data, mode="RGB")

            # Save as PNG
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            return StreamingResponse(img_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving tile {z}/{x}/{y} for {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tile: {str(e)}") from e
