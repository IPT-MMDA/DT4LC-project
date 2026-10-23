"""File upload, listing, and download endpoints."""

import base64
import io
import logging
from typing import Annotated
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from dta.dti import artifact_files
from dta.dti.raster import describe_raster, format_size
from dta.dti.schemas import RasterMetadata

from ..paths import resolve_served_path
from ..schemas import FileInfo, FileListResponse, UploadResponse
from ..utils import MAX_UPLOAD_SIZE, UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["files"])


@router.post(  # type: ignore[misc]
    "/upload",
    response_model=UploadResponse,
    summary="Upload GeoTIFF",
)
async def upload_geotiff(
    file: Annotated[UploadFile, File(description="GeoTIFF raster (.tif or .tiff)")],
) -> UploadResponse:
    """Upload a GeoTIFF, store it on the server, and return metadata plus a PNG preview.

    The preview is written next to the upload and returned as ``preview_url``;
    ``preview_png_base64`` is kept for compatibility (clients must not persist it).
    """
    # Basic checks
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Please upload a .tif/.tiff GeoTIFF.")

    # Check Content-Length header first (fast reject before reading body)
    if file.size is not None and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    # Read in chunks to enforce size limit without trusting Content-Length
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):  # 1 MB chunks
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
            )
        chunks.append(chunk)
    raw = b"".join(chunks)

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Save file to temp directory
    file_id = str(uuid.uuid4())[:8]
    saved_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    saved_path.write_bytes(raw)
    preview_path = saved_path.with_suffix(".png")

    # Read in-memory with rasterio for preview
    try:
        with MemoryFile(raw) as mem, mem.open() as src:
            # Read first band as masked array
            band1 = src.read(1, masked=True)  # (H, W) masked ndarray
            h, w = band1.shape
            bounds = src.bounds  # left, bottom, right, top
            crs = src.crs.to_string() if src.crs else None

            # Simple percentile stretch to 8-bit for preview
            # Convert to float FIRST, then fill with NaN (can't fill uint8 with NaN)
            data = band1.astype("float64").filled(np.nan)
            finite = np.isfinite(data)
            if not finite.any():
                raise HTTPException(status_code=400, detail="All pixels are nodata.")

            # percentiles on finite pixels only
            p2, p98 = np.percentile(data[finite], [2, 98])
            if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                p2, p98 = float(np.nanmin(data[finite])), float(np.nanmax(data[finite]))

            # Handle edge case where p2 == p98
            if p98 - p2 < 1e-10:
                scaled = np.zeros_like(data)
            else:
                scaled = (data - p2) / (p98 - p2)

            # Replace NaN/inf with 0 BEFORE converting to uint8
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
            scaled = np.clip(scaled, 0.0, 1.0)
            scaled = (scaled * 255.0 + 0.5).astype("uint8")

            # Convert to PNG (grayscale, mode "L")
            img = Image.fromarray(scaled, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            preview_url = artifact_files.write_png(preview_path, buf.getvalue())["url"]
            metadata = _raster_metadata(src, file.filename, len(raw))

        return UploadResponse(
            id=file_id,
            filename=file.filename,
            path=str(saved_path),
            size=[int(w), int(h)],
            crs=crs,
            bounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
            preview_png_base64=b64,
            preview_url=preview_url,
            metadata=metadata,
        )
    except Exception as e:
        # Clean up saved file on error
        if saved_path.exists():
            saved_path.unlink()
        preview_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF: {e}") from e


def _raster_metadata(src: object, filename: str, size_bytes: int) -> RasterMetadata:
    """``describe_raster``, or just the size if it fails: metadata never fails an upload or a listing."""
    try:
        return describe_raster(src, filename, size_bytes)
    except Exception as exc:
        logger.warning("Raster metadata extraction failed for %s: %s", filename, exc)
        return RasterMetadata(size_bytes=size_bytes, size_label=format_size(size_bytes))


@router.get(  # type: ignore[misc]
    "/files",
    response_model=FileListResponse,
    summary="List GeoTIFF files",
)
async def list_files() -> FileListResponse:
    """List uploaded and GEE-exported GeoTIFF files on the server."""
    try:
        import rasterio

        from dta.config import CACHE_PATH

        files: list[FileInfo] = []

        # Scan uploads directory
        upload_files = list(UPLOAD_DIR.glob("*.tif")) + list(UPLOAD_DIR.glob("*.tiff"))
        for file_path in upload_files:
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string() if src.crs else None
                    width, height = src.width, src.height

                    files.append(
                        FileInfo(
                            id=file_path.stem,
                            filename=file_path.name,
                            path=str(file_path),
                            size=[width, height],
                            crs=crs,
                            bounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                            size_bytes=file_path.stat().st_size,
                            source="upload",
                            modified=file_path.stat().st_mtime,
                            # Uploads are stored as <id>_<original name>
                            metadata=_raster_metadata(src, file_path.name.split("_", 1)[-1], file_path.stat().st_size),
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

        # Scan exports directory
        export_dir = CACHE_PATH / "gee_exports"
        if export_dir.exists():
            export_files = list(export_dir.glob("*.tif")) + list(export_dir.glob("*.tiff"))
            for file_path in export_files:
                try:
                    with rasterio.open(file_path) as src:
                        bounds = src.bounds
                        crs = src.crs.to_string() if src.crs else None
                        width, height = src.width, src.height

                        files.append(
                            FileInfo(
                                id=file_path.stem,
                                filename=file_path.name,
                                path=str(file_path),
                                size=[width, height],
                                crs=crs,
                                bounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                                size_bytes=file_path.stat().st_size,
                                source="export",
                                modified=file_path.stat().st_mtime,
                                metadata=_raster_metadata(src, file_path.name, file_path.stat().st_size),
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.modified, reverse=True)

        num_uploads = len([f for f in files if f.source == "upload"])
        num_exports = len([f for f in files if f.source == "export"])
        logger.info(f"Listed {len(files)} files ({num_uploads} uploads, {num_exports} exports)")

        return FileListResponse(files=files, count=len(files))

    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(  # type: ignore[misc]
    "/download",
    summary="Download file by path",
    response_description="Binary file (GeoTIFF, GeoPackage, PNG, etc.)",
)
async def download_file(
    path: str = Query(..., description="Absolute or allowed-relative path to the file on the server"),
) -> FileResponse:
    """Download a generated or uploaded file (GeoPackage, GeoTIFF, PNG, JSON)."""
    file_path = resolve_served_path(path)

    # Determine content type based on extension
    suffix = file_path.suffix.lower()
    content_types = {
        ".gpkg": "application/geopackage+sqlite3",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".png": "image/png",
        ".json": "application/json",
    }
    media_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
