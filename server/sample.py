"""The bundled Kakhovka sample as the caller's uploads (``POST /v1/sample``).

The two dates in ``resources/kahovka_data/`` are linked (or copied, across filesystems)
into the upload directory under an id derived from the session, with the owner sidecar
and a true-colour ``<stem>.png`` preview, so they behave like any upload: owner-scoped,
served by ``/v1/download`` and the tiles, removed by the guest cleanup. Asking again in
the same session returns the files already there.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import rasterio

from dta.config import RESOURCES_PATH
from dta.dti import artifact_files
from dta.dti.data_sources.gee_region import date_range
from dta.dti.raster import _band_summary, band_indices, describe_raster, metadata_line
from dta.dti.schemas import RasterMetadata

from .schemas import UploadResponse
from .sessions import Session
from .uploads import owner_file, read_adopted_from, read_owner, write_owner

SAMPLE_NAME = "kakhovka_hls_l30"
SAMPLE_DIR = RESOURCES_PATH / "kahovka_data"
SAMPLE_FILES = ("hlsl_20230601.tif", "hlsl_20230609.tif")


class SampleNotBundledError(Exception):
    """The sample files are not on this server."""


def _sample_id(session_id: str) -> str:
    return hashlib.sha256(f"sample:{session_id}".encode()).hexdigest()[:8]


def _file_id(session: Session) -> str:
    """8 hex characters, like an upload id, fixed per owner (the session id itself stays secret)."""
    return _sample_id(session.owner)


def is_sample_copy(path: Path, owner: str | None) -> bool:
    """Whether upload ``path`` is ``owner``'s copy of a bundled sample file (from :func:`attach_sample`),
    including one it adopted from its guest session on sign-in."""
    file_id, _, name = path.name.partition("_")
    if owner is None or name not in SAMPLE_FILES:
        return False
    adopted_from = read_adopted_from(path)
    return file_id == _sample_id(owner) or (adopted_from is not None and file_id == _sample_id(adopted_from))


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _true_colour_png(src: rasterio.DatasetReader) -> bytes:
    """Red, green and blue, each stretched to its 2nd-98th percentile; nodata is black."""
    roles = band_indices(src)
    channels = []
    for role in ("red", "green", "blue"):
        band = src.read(roles[role], masked=True).astype("float64").filled(np.nan)
        finite = band[np.isfinite(band)]
        lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
        scaled = (band - lo) / (hi - lo) if hi > lo else np.zeros_like(band)
        channels.append((np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0) * 255.0 + 0.5).astype("uint8"))
    buf = io.BytesIO()
    Image.fromarray(np.dstack(channels), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _attachment(file_id: str, filename: str, saved: Path, session: Session) -> UploadResponse:
    preview = saved.with_suffix(".png")
    with rasterio.open(saved) as src:
        if not preview.is_file():
            artifact_files.write_png(preview, _true_colour_png(src))
        bounds = src.bounds
        return UploadResponse(
            id=file_id,
            filename=filename,
            path=str(saved),
            size=[src.width, src.height],
            crs=src.crs.to_string() if src.crs else None,
            bounds=[bounds.left, bounds.bottom, bounds.right, bounds.top],
            preview_png_base64="",  # not inlined: use preview_url
            preview_url=artifact_files.ref("png", preview)["url"],
            metadata=describe_raster(src, filename, saved.stat().st_size),
            owner=session.owner,
        )


def pair_line(first: RasterMetadata, second: RasterMetadata) -> str:
    """``"HLS L30 · B2–B6 · 30 m · 4.0 MB · 2023-06-01 → 06-09"``: the chip line of two dates
    (sensor, bands and resolution of the first file, the size of the larger file, the date range)."""
    base = metadata_line(first.sensor, _band_summary([b.code for b in first.bands]), first.resolution_m, None)
    larger = max((first, second), key=lambda m: m.size_bytes or 0)
    dates = date_range(first.date, second.date) if first.date and second.date else None
    return " · ".join(p for p in (base, larger.size_label, dates) if p)


def attach_sample(upload_dir: Path, session: Session) -> tuple[str, list[UploadResponse]]:
    """``(pair line, attachments)`` of the sample in ``upload_dir`` for ``session``; blocking.

    Raises:
        SampleNotBundledError: a sample file is missing.
    """
    sources = [SAMPLE_DIR / name for name in SAMPLE_FILES]
    if not all(p.is_file() for p in sources):
        raise SampleNotBundledError(f"The Kakhovka sample is not bundled on this server ({SAMPLE_DIR.name}/).")
    file_id = _file_id(session)
    attachments = []
    for src_path in sources:
        saved = upload_dir / f"{file_id}_{src_path.name}"
        if not (saved.is_file() and read_owner(saved) == session.owner):
            for stale in (saved, saved.with_suffix(".png"), owner_file(saved)):
                stale.unlink(missing_ok=True)
            _link_or_copy(src_path, saved)
            write_owner(saved, session)
        attachments.append(_attachment(file_id, src_path.name, saved, session))
    return pair_line(attachments[0].metadata, attachments[1].metadata), attachments
