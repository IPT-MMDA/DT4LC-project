"""File upload, listing, and download endpoints."""

import base64
import io
import logging
from pathlib import Path
import tempfile
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from ..utils import MAX_UPLOAD_SIZE, UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["files"])


@router.post("/upload")  # type: ignore[misc]
async def upload_geotiff(file: UploadFile = File) -> JSONResponse:
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

        return JSONResponse(
            {
                "id": file_id,
                "filename": file.filename,
                "path": str(saved_path),  # File path for use in execution
                "size": [int(w), int(h)],
                "crs": crs,
                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                "preview_png_base64": b64,  # data:image/png;base64,<this>
            }
        )
    except Exception as e:
        # Clean up saved file on error
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF: {e}") from e


@router.get("/files")  # type: ignore[misc]
async def list_files() -> JSONResponse:
    """List all available GeoTIFF files (uploaded and exported).

    Returns:
        JSON with list of files from both uploads and exports directories
    """
    try:
        import rasterio

        from dta.config import CACHE_PATH

        files = []

        # Scan uploads directory
        upload_files = list(UPLOAD_DIR.glob("*.tif")) + list(UPLOAD_DIR.glob("*.tiff"))
        for file_path in upload_files:
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string() if src.crs else None
                    width, height = src.width, src.height

                    files.append(
                        {
                            "id": file_path.stem,
                            "filename": file_path.name,
                            "path": str(file_path),
                            "size": [width, height],
                            "crs": crs,
                            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                            "size_bytes": file_path.stat().st_size,
                            "source": "upload",
                            "modified": file_path.stat().st_mtime,
                        }
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
                            {
                                "id": file_path.stem,
                                "filename": file_path.name,
                                "path": str(file_path),
                                "size": [width, height],
                                "crs": crs,
                                "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                                "size_bytes": file_path.stat().st_size,
                                "source": "export",
                                "modified": file_path.stat().st_mtime,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue

        # Sort by modification time (newest first)
        files.sort(key=lambda x: float(x["modified"]), reverse=True)  # type: ignore[arg-type]

        num_uploads = len([f for f in files if f["source"] == "upload"])
        num_exports = len([f for f in files if f["source"] == "export"])
        logger.info(f"Listed {len(files)} files ({num_uploads} uploads, {num_exports} exports)")

        return JSONResponse({"ok": True, "files": files, "count": len(files)})

    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/download")  # type: ignore[misc]
async def download_file(path: str) -> FileResponse:
    """Download a file from the server.

    Used for downloading generated outputs like GeoPackage files.

    Args:
        path: Path to the file to download

    Returns:
        File response with appropriate content type
    """
    file_path = Path(path)

    # Security: Only allow downloads from specific directories
    allowed_dirs = [
        Path(tempfile.gettempdir()) / "dt4lc_delineate",
        UPLOAD_DIR,
        Path(tempfile.gettempdir()),
    ]

    # Check if path is under an allowed directory
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            file_path.resolve().relative_to(allowed_dir.resolve())
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied: path not in allowed directories")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

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
