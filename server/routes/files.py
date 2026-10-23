"""File upload, listing, and download endpoints."""

import asyncio
import base64
from datetime import datetime
import io
import logging
from pathlib import Path
from typing import Annotated
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from dta.dti import artifact_files
from dta.dti.raster import describe_raster, format_size
from dta.dti.schemas import RasterMetadata

from ..paths import ensure_served_owner, resolve_served_path
from ..sample import SAMPLE_DIR, SAMPLE_NAME, SampleNotBundledError, attach_sample, is_sample_copy, pair_line
from ..schemas import (
    ErrorDetail,
    ErrorResponse,
    FileDeleteResponse,
    FileGroup,
    FileInfo,
    FileListResponse,
    SampleResponse,
    UploadResponse,
)
from ..sessions import CurrentSession
from ..uploads import delete_upload, owner_file, read_created_at, read_owner, write_owner
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
    session: CurrentSession,
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
    write_owner(saved_path, session)
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
            owner=session.owner,
        )
    except Exception as e:
        # Clean up saved file on error
        if saved_path.exists():
            saved_path.unlink()
        preview_path.unlink(missing_ok=True)
        owner_file(saved_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF: {e}") from e


@router.post(  # type: ignore[misc]
    "/sample",
    response_model=SampleResponse,
    summary="Attach the bundled Kakhovka sample",
    responses={404: {"model": ErrorResponse, "description": "SAMPLE_NOT_BUNDLED"}},
)
async def attach_bundled_sample(session: CurrentSession) -> SampleResponse | JSONResponse:
    """Add the bundled Kakhovka pair (2023-06-01, 2023-06-09) to the caller's uploads.

    Each file is an upload of the caller (owner-scoped, with metadata and a true-colour
    ``preview_url``) and expires with a guest session like any upload; ``line`` is the pair's
    chip line. A second call in the same session returns the same files. 404
    ``SAMPLE_NOT_BUNDLED`` if the server has no sample.
    """
    try:
        line, attachments = await asyncio.to_thread(attach_sample, UPLOAD_DIR, session)
    except SampleNotBundledError as e:
        error = ErrorDetail(code="SAMPLE_NOT_BUNDLED", message=str(e))
        return JSONResponse(ErrorResponse(error=error).model_dump(mode="json"), status_code=404)
    return SampleResponse(name=SAMPLE_NAME, line=line, attachments=attachments)


def _raster_metadata(src: object, filename: str, size_bytes: int) -> RasterMetadata:
    """``describe_raster``, or just the size if it fails: metadata never fails an upload or a listing."""
    try:
        return describe_raster(src, filename, size_bytes)
    except Exception as exc:
        logger.warning("Raster metadata extraction failed for %s: %s", filename, exc)
        return RasterMetadata(size_bytes=size_bytes, size_label=format_size(size_bytes))


def _group_samples(files: list[FileInfo]) -> None:
    """Give each owner's sample files one ``group``: the sample name and the pair line with the CRS."""
    pairs: dict[str | None, list[FileInfo]] = {}
    for f in files:
        if f.kind == "sample":
            pairs.setdefault(f.upload_id, []).append(f)
    for pair in pairs.values():
        pair.sort(key=lambda f: f.filename)
        first, last = pair[0].metadata, pair[-1].metadata
        if first is None or last is None:
            continue
        line = " · ".join(p for p in (pair_line(first, last), pair[0].crs) if p)  # one date left: that date
        group = FileGroup(name=f"{SAMPLE_NAME}.tif", line=line)
        for f in pair:
            f.group = group


@router.get(  # type: ignore[misc]
    "/files",
    response_model=FileListResponse,
    summary="List GeoTIFF files",
)
async def list_files(session: CurrentSession) -> FileListResponse:
    """List the caller's uploads and the GEE-exported GeoTIFF files on the server.

    Operators see every upload, including those from before v0.6.0 that have no owner. ``kind`` is
    ``upload``, ``sample`` (the caller's copy of the bundled sample, both dates carrying the same
    ``group`` so the Library shows them as one row) or ``export``.
    """
    try:
        import rasterio

        from dta.config import CACHE_PATH

        files: list[FileInfo] = []

        # Scan uploads directory
        upload_files = list(UPLOAD_DIR.glob("*.tif")) + list(UPLOAD_DIR.glob("*.tiff"))
        for file_path in upload_files:
            owner = read_owner(file_path)
            if not session.can_see(owner):
                continue
            try:
                with rasterio.open(file_path) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string() if src.crs else None
                    width, height = src.width, src.height

                    preview = file_path.with_suffix(".png")
                    modified = file_path.stat().st_mtime
                    files.append(
                        FileInfo(
                            id=file_path.stem,
                            upload_id=file_path.name.split("_", 1)[0],
                            filename=file_path.name,
                            path=str(file_path),
                            size=[width, height],
                            crs=crs,
                            bounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                            size_bytes=file_path.stat().st_size,
                            source="upload",
                            modified=modified,
                            # Uploads are stored as <id>_<original name>
                            metadata=_raster_metadata(src, file_path.name.split("_", 1)[-1], file_path.stat().st_size),
                            owner=owner,
                            kind="sample" if is_sample_copy(file_path, owner) else "upload",
                            preview_url=artifact_files.ref("png", preview)["url"] if preview.is_file() else None,
                            created_at=read_created_at(file_path) or datetime.fromtimestamp(modified).isoformat(),
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
                                kind="export",
                                created_at=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
                    continue

        _group_samples(files)

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.modified, reverse=True)

        num_uploads = len([f for f in files if f.source == "upload"])
        num_exports = len([f for f in files if f.source == "export"])
        logger.info(f"Listed {len(files)} files ({num_uploads} uploads, {num_exports} exports)")

        return FileListResponse(files=files, count=len(files))

    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete(  # type: ignore[misc]
    "/files/{file_id}",
    response_model=FileDeleteResponse,
    summary="Delete an upload",
)
async def delete_file(file_id: str, session: CurrentSession) -> FileDeleteResponse:
    """Delete the caller's upload with its preview and owner record.

    ``file_id`` is an ``upload_id`` (what ``POST /v1/upload`` or ``POST /v1/sample`` returned) or
    a file's ``id`` from ``GET /v1/files``. The two sample dates share one ``upload_id``, so
    deleting by it deletes the pair; ``POST /v1/sample`` attaches a fresh copy. Only copies in the
    upload directory are deleted, never the bundled sample. 404 for an unknown id or another
    session's upload (operators may delete any).
    """

    def matches(p: Path) -> bool:
        return p.suffix.lower() in (".tif", ".tiff") and (p.stem == file_id or p.name.startswith(f"{file_id}_"))

    upload_root = UPLOAD_DIR.resolve()
    bundled = SAMPLE_DIR.resolve()
    targets = [
        p
        for p in sorted(UPLOAD_DIR.iterdir() if UPLOAD_DIR.is_dir() else [])
        if p.is_file()
        and matches(p)
        and p.resolve().parent == upload_root
        and not p.resolve().is_relative_to(bundled)
        and session.can_see(read_owner(p))
    ]
    if not targets:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    for p in targets:
        await asyncio.to_thread(delete_upload, p)
    logger.info(f"Deleted {len(targets)} upload(s) for id {file_id}")
    return FileDeleteResponse(deleted=[p.name for p in targets])


@router.get(  # type: ignore[misc]
    "/download",
    summary="Download file by path",
    response_description="Binary file (GeoTIFF, GeoPackage, PNG, etc.)",
)
async def download_file(
    session: CurrentSession,
    path: str = Query(..., description="Absolute or allowed-relative path to the file on the server"),
) -> FileResponse:
    """Download a generated or uploaded file (GeoPackage, GeoTIFF, PNG, JSON).

    Another owner's job artifact or upload is 404.
    """
    file_path = resolve_served_path(path)
    await ensure_served_owner(file_path, session)

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
