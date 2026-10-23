"""XYZ tile serving endpoint for GeoTIFF files."""

import io
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import mercantile
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from dta.dti import artifact_files

from ..paths import ensure_served_owner, resolve_served_path
from ..sessions import CurrentSession
from ..utils import apply_colormap, detect_data_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/tiles", tags=["tiles"])

TILE_SIZE = 256


def _png(rgba: np.ndarray) -> StreamingResponse:
    """An RGBA uint8 array as a PNG response."""
    img_bytes = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/png")


def _untagged_rgb(data: np.ndarray, valid: np.ndarray, data_type: str) -> np.ndarray:
    """RGB of an untagged single-band tile, by the data type its file name suggests (NaN where not ``valid``)."""
    if data_type in ["ndvi", "ndwi", "ndsi"]:
        # Spectral indices range from -1 to 1: clip, scale to 0-255 and apply the index's colormap
        data_normalized = ((np.clip(np.nan_to_num(data), -1, 1) + 1) / 2 * 255).astype(np.uint8)
        colormap = {"ndvi": "RdYlGn", "ndwi": "Blues", "ndsi": "cool"}[data_type]
        return apply_colormap(data_normalized, colormap)
    # Generic single-band data: percentile stretch in viridis
    p2, p98 = np.percentile(data[valid], [2, 98])
    stretched = np.clip((np.nan_to_num(data, nan=p2) - p2) / ((p98 - p2) or 1.0) * 255, 0, 255)
    return apply_colormap(stretched.astype(np.uint8), "viridis")


@router.get(  # type: ignore[misc]
    "/{z}/{x}/{y}",
    summary="GeoTIFF map tile",
    response_description="256×256 RGBA PNG tile; transparent off the raster and on nodata",
)
async def get_tile(
    z: int,
    x: int,
    y: int,
    session: CurrentSession,
    path: str = Query(..., description="Server path to the GeoTIFF file"),
) -> StreamingResponse:
    """Serve a GeoTIFF as an XYZ map tile (PNG); another owner's job artifact or upload is 404."""
    try:
        # Get tile bounds in Web Mercator (EPSG:3857)
        tile = mercantile.Tile(x, y, z)
        tile_bounds_wgs84 = mercantile.bounds(tile)

        # Open GeoTIFF
        file_path = resolve_served_path(path)
        await ensure_served_owner(file_path, session)

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

            # Tile pixels whose centre lies on the raster; the rest stays transparent.
            cols = window.col_off + (np.arange(TILE_SIZE) + 0.5) * window.width / TILE_SIZE
            rows = window.row_off + (np.arange(TILE_SIZE) + 0.5) * window.height / TILE_SIZE
            inside = ((rows >= 0) & (rows < src.height))[:, None] & ((cols >= 0) & (cols < src.width))[None, :]
            if not inside.any():
                return _png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))

            # Boundless: the part of the window off the raster is filled, not stretched over the tile.
            bands = [1, 2, 3] if src.count >= 3 else [1]
            read = src.read(
                bands, window=window, out_shape=(len(bands), TILE_SIZE, TILE_SIZE), boundless=True, masked=True
            )
            data = np.ma.filled(read.astype(np.float64), np.nan)
            valid = inside & ~np.ma.getmaskarray(read).any(axis=0) & np.isfinite(data).all(axis=0)
            if not valid.any():
                return _png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))

            rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
            rgba[..., 3] = np.where(valid, 255, 0)
            style = artifact_files.read_style(src.tags())
            if src.count >= 3:
                # RGB data: the first 3 bands, 2-98 % stretch unless already 8-bit
                rgb = np.transpose(data, (1, 2, 0))  # CHW to HWC
                if src.dtypes[0] != "uint8":
                    p2, p98 = np.percentile(rgb[valid], [2, 98])
                    rgb = np.clip((rgb - p2) / ((p98 - p2) or 1.0) * 255, 0, 255)
                rgba[..., :3] = np.where(valid[..., None], rgb, 0).astype(np.uint8)
            elif style is not None:
                # Tagged job artifact: the colours of its bare map PNG
                rgba = artifact_files.colourize(np.where(valid, data[0], np.nan), style)
            else:
                rgba[..., :3] = _untagged_rgb(np.where(valid, data[0], np.nan), valid, data_type)

            return _png(rgba)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving tile {z}/{x}/{y} for {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tile: {str(e)}") from e
