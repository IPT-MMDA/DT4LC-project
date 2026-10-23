"""XYZ tile serving endpoint for GeoTIFF files."""

from functools import lru_cache
import io
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import mercantile
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from dta.dti import artifact_files
from dta.dti.raster import band_indices

from ..paths import ensure_served_owner, resolve_served_path
from ..sessions import CurrentSession
from ..utils import apply_colormap, detect_data_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/tiles", tags=["tiles"])

TILE_SIZE = 256
WEB_MERCATOR = "EPSG:3857"
# The true-colour stretch is computed on a read of at most this many pixels on the long side.
STRETCH_MAX_SIDE = 1024


def _png(rgba: np.ndarray) -> Response:
    """An RGBA uint8 array as a PNG response.

    One body, not a stream: streaming a ``BytesIO`` sends it line by line, ~1800 chunks a tile.
    Compression level 1 encodes in a quarter of the default's time for ~13 % more bytes.
    """
    img_bytes = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(img_bytes, format="PNG", compress_level=1)
    return Response(img_bytes.getvalue(), media_type="image/png")


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


@lru_cache(maxsize=64)
def true_colour_stretch(file: str, mtime_ns: int) -> tuple[list[int], np.ndarray] | None:
    """``(red, green, blue band indices, (3, 2) array of 2nd/98th percentiles)`` of a raster, or ``None``.

    ``None`` if the raster has fewer than 3 bands or :func:`band_indices` finds no red, green or
    blue band. The percentiles are per band, over the pixels valid in all three, as in the sample's
    preview PNG; a large raster is read decimated to :data:`STRETCH_MAX_SIDE`. Cached per path and
    modification time, so each raster is read once and a rewritten file is read again.
    """
    with rasterio.open(file) as src:
        if src.count < 3:
            return None
        roles = band_indices(src)
        bands = [roles["red"], roles["green"], roles["blue"]]
        if any(b is None for b in bands):
            return None
        scale = min(1.0, STRETCH_MAX_SIDE / max(src.width, src.height))
        shape = (3, max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        read = src.read(bands, out_shape=shape, masked=True)
    data = np.ma.filled(read.astype(np.float64), np.nan)
    valid = np.isfinite(data).all(axis=0)
    limits = np.array([np.percentile(band[valid], [2, 98]) if valid.any() else (0.0, 1.0) for band in data])
    return [int(b) for b in bands if b is not None], limits


def _stretch(data: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """``(3, H, W)`` bands -> ``(H, W, 3)`` uint8, each band's ``[lo, hi]`` stretched to 0-255."""
    lo, hi = limits[:, 0, None, None], limits[:, 1, None, None]
    scaled = np.divide(data - lo, hi - lo, out=np.zeros_like(data), where=hi > lo)
    return np.transpose(np.clip(np.nan_to_num(scaled), 0.0, 1.0) * 255.0 + 0.5, (1, 2, 0)).astype(np.uint8)


def _warped_tile(
    src: rasterio.DatasetReader, tile: mercantile.Tile, bands: list[int], resampling: Resampling
) -> np.ndarray:
    """``bands`` of ``src`` reprojected onto ``tile``'s 256×256 Web Mercator grid, as float32; NaN off the raster
    and where a band is nodata (each band's own nodata, so one band's nodata never blends into its neighbours)."""
    bounds = mercantile.xy_bounds(tile)
    with WarpedVRT(
        src,
        crs=WEB_MERCATOR,
        transform=from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, TILE_SIZE, TILE_SIZE),
        width=TILE_SIZE,
        height=TILE_SIZE,
        resampling=resampling,
        dtype="float32",
        nodata=np.nan,
        UNIFIED_SRC_NODATA="NO",
    ) as vrt:
        return np.asarray(vrt.read(bands))


def _overlaps(src: rasterio.DatasetReader, tile: mercantile.Tile) -> bool:
    """Whether ``tile`` meets the raster's Web Mercator bounding box."""
    west, south, east, north = transform_bounds(src.crs, WEB_MERCATOR, *src.bounds, densify_pts=21)
    b = mercantile.xy_bounds(tile)
    return bool(b.left < east and b.right > west and b.bottom < north and b.top > south)


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
    style: Literal["truecolor"] | None = Query(
        None,
        description=(
            "'truecolor': red, green and blue bands, each stretched to its 2nd-98th percentile over the raster "
            "(422 if the raster has no red, green or blue band). Default: the style in the GeoTIFF tags; "
            "untagged rasters with resolvable red, green and blue bands render in true colour too."
        ),
    ),
) -> Response:
    """Serve a GeoTIFF as an XYZ map tile (PNG), reprojected to Web Mercator; another owner's job artifact
    or upload is 404."""
    try:
        tile = mercantile.Tile(x, y, z)

        # Open GeoTIFF
        file_path = resolve_served_path(path)
        await ensure_served_owner(file_path, session)

        # Detect data type from filename
        data_type = detect_data_type(file_path.name)

        with rasterio.open(str(file_path)) as src:
            tagged = artifact_files.read_style(src.tags())
            true_colour = None
            if style == "truecolor" or (tagged is None and src.count >= 3):
                true_colour = true_colour_stretch(str(file_path), file_path.stat().st_mtime_ns)
                if style == "truecolor" and true_colour is None:
                    raise HTTPException(status_code=422, detail="The raster has no red, green and blue bands")

            if not _overlaps(src, tile):
                return _png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))

            if true_colour is not None:
                bands = true_colour[0]
            else:
                bands = [1, 2, 3] if src.count >= 3 else [1]
            # Class values must not blend into in-between values; everything else is continuous.
            is_class = tagged is not None and tagged["kind"] == "class"
            resampling = Resampling.nearest if is_class else Resampling.bilinear
            data = _warped_tile(src, tile, bands, resampling).astype(np.float64)
            valid = np.isfinite(data).all(axis=0)
            if not valid.any():
                return _png(np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8))

            rgba = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
            rgba[..., 3] = np.where(valid, 255, 0)
            if true_colour is not None:
                rgba[..., :3] = np.where(valid[..., None], _stretch(data, true_colour[1]), 0)
            elif src.count >= 3:
                # RGB data: the first 3 bands, 2-98 % stretch unless already 8-bit
                rgb = np.transpose(data, (1, 2, 0))  # CHW to HWC
                if src.dtypes[0] != "uint8":
                    p2, p98 = np.percentile(rgb[valid], [2, 98])
                    rgb = np.clip((rgb - p2) / ((p98 - p2) or 1.0) * 255, 0, 255)
                rgba[..., :3] = np.where(valid[..., None], rgb, 0).astype(np.uint8)
            elif tagged is not None:
                # Tagged job artifact: the colours of its bare map PNG
                rgba = artifact_files.colourize(np.where(valid, data[0], np.nan), tagged)
            else:
                rgba[..., :3] = _untagged_rgb(np.where(valid, data[0], np.nan), valid, data_type)

            return _png(rgba)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving tile {z}/{x}/{y} for {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate tile: {str(e)}") from e
