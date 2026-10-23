"""Raster IO utilities for DT4LC.

Shared utilities for loading and processing raster data (GeoTIFF, etc.).
Used by models and algorithms to ensure consistent data handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from dta.dti.schemas import BandInfo, RasterMetadata

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


# Semantic band descriptions (lower-cased) -> role.
_SEMANTIC_ROLES = {
    "blue": "blue",
    "b": "blue",
    "green": "green",
    "g": "green",
    "red": "red",
    "r": "red",
    "nir": "nir",
    "nir08": "nir",
    "swir": "swir",
    "swir1": "swir",
    "swir16": "swir",
}
# Band codes per sensor family -> role.
_SENTINEL2_CODES = {"blue": "2", "green": "3", "red": "4", "nir": "8", "swir": "11"}
_LANDSAT_CODES = {"blue": "2", "green": "3", "red": "4", "nir": "5", "swir": "6"}
_BAND_CODE = re.compile(r"^(?:sr_|st_)?b0*(\d+a?)$")


def _roles_from_count(n: int) -> dict[str, int | None]:
    """Guess band roles from the band count alone (see :func:`band_indices`)."""
    if n >= 7:
        # Landsat 8/9: B1 coastal, B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1
        return {"red": 4, "nir": 5, "green": 3, "blue": 2, "swir": 6}
    if n >= 5:
        # Sentinel-2 subset / generic: B, G, R, NIR, SWIR (+ optional SWIR2)
        return {"red": 3, "nir": 4, "green": 2, "blue": 1, "swir": 5}
    if n == 4:
        # RGBN format: R, G, B, NIR
        return {"red": 1, "nir": 4, "green": 2, "blue": 3, "swir": None}
    # 2-3 bands: minimal — assume red and NIR are present.
    return {"red": 1, "nir": 2, "green": None, "blue": None, "swir": None}


def _roles_from_descriptions(descriptions: Any, count: int) -> dict[str, int | None] | None:
    """Resolve band roles from band descriptions, or ``None`` if they don't identify any."""
    names = [(d or "").strip().lower() for d in (descriptions or ())]

    semantic: dict[str, int] = {}
    for i, name in enumerate(names, start=1):
        if name in _SEMANTIC_ROLES:
            semantic.setdefault(_SEMANTIC_ROLES[name], i)
    if semantic:
        # Roles not named: take the count heuristic's band, unless another role names it.
        guessed = {
            role: idx
            for role, idx in _roles_from_count(count).items()
            if role not in semantic and idx is not None and idx not in semantic.values()
        }
        if guessed:
            logger.warning(
                "Band descriptions %r name only %s; guessing %s from the band count (%d)",
                descriptions,
                sorted(semantic),
                guessed,
                count,
            )
        semantic.update(guessed)
        return {role: semantic.get(role) for role in ("red", "nir", "green", "blue", "swir")}

    codes = {m.group(1): i for i, n in enumerate(names, start=1) if (m := _BAND_CODE.match(n))}
    family = _code_family(codes)
    if family == "sentinel2":
        scheme = dict(_SENTINEL2_CODES)
        if "8" not in codes:
            scheme["nir"] = "8a"
    elif family == "landsat":
        scheme = _LANDSAT_CODES
    else:
        return None
    return {role: codes.get(scheme[role]) for role in ("red", "nir", "green", "blue", "swir")}


def _code_family(codes: Any) -> str | None:
    """``"sentinel2"``, ``"landsat"`` or ``None`` for a set of lower-case band codes (``"8a"``, ``"11"``)."""
    if not codes:
        return None
    # Decided by the whole code set: Landsat 8/9 B8 is panchromatic, B5-B7 are NIR/SWIR and
    # B10 is thermal (TIRS1); Sentinel-2 B5-B7 are red edge and L2A has no B10. B8 with B5-B7
    # is Landsat only if B10 is present, so a Landsat stack without B10 (e.g. B1..B9) still
    # reads as Sentinel-2 (known limitation).
    landsat_full = any(c in codes for c in ("5", "6", "7")) and "10" in codes
    if "8a" in codes or "12" in codes or ("8" in codes and not landsat_full):
        return "sentinel2"
    if "8" in codes or all(c.isdigit() and (1 <= int(c) <= 7 or 9 <= int(c) <= 11) for c in codes):
        return "landsat"
    return None


def band_indices(src: Any) -> dict[str, int | None]:
    """Mapping from band-role names to 1-based band indices.

    Single source of truth for "which band is red?" across spectral indices
    and change detection. Returns ``None`` for roles the source doesn't carry
    (e.g. SWIR on a 4-band RGBN image).

    Band descriptions (``src.descriptions``) are read first:

    * Semantic names, case-insensitive: ``blue`` / ``B``, ``green`` / ``G``,
      ``red`` / ``R``, ``nir`` / ``nir08``, ``swir`` / ``swir1`` / ``swir16``
      (SWIR1). Roles not named take the count heuristic's band below,
      with a warning, unless that band is named as another role; else ``None``.
    * ``Bn`` band codes (optionally ``SR_Bn`` / ``ST_Bn``), resolved by
      sensor family from the whole code set. Sentinel-2 (B2 blue, B3 green,
      B4 red, B8 NIR (B8A if B8 is absent), B11 SWIR1) if ``B8A`` or ``B12``
      is present, or ``B8`` without both B10 and any of B5-B7. Landsat 8/9 /
      HLS L30 (B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1) if ``B8`` is
      present with B10 and any of B5-B7, or if there is no ``B8`` and all
      codes are within B1-B7 or B9-B11. Any other code set identifies no band.

    If the descriptions identify no band, a warning is logged and the layout
    is guessed from ``src.count``:

    * 7+ bands: Landsat 8/9 surface-reflectance order
      (B1 coastal, B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1, B7 SWIR2).
    * 5–6 bands: Sentinel-2 subset / generic ``(B, G, R, NIR, SWIR[, ...])``.
    * 4 bands: assume ``RGBN`` ordering (R, G, B, NIR). No SWIR.
    * 2–3 bands: assume ``red, NIR[, …]`` minimum. Most roles unavailable.

    Args:
        src: An open ``rasterio.DatasetReader``.

    Returns:
        Dict with keys ``red``, ``nir``, ``green``, ``blue``, ``swir`` and
        1-based band-index values (or ``None`` if absent).
    """
    roles = _roles_from_descriptions(getattr(src, "descriptions", None), src.count)
    if roles is not None:
        return roles

    n = src.count
    logger.warning(
        "Band roles not identifiable from band descriptions %r; guessing the layout from the band count (%d)",
        getattr(src, "descriptions", None),
        n,
    )
    return _roles_from_count(n)


def reflectance_divisor(dtype: Any, data: np.ndarray) -> float:
    """Divisor that maps stored band values to 0-1 reflectance.

    Decided by the source raster dtype (``src.dtypes[...]``), not by the
    values, so dark integer scenes are scaled like bright ones:

    * 1-byte integers (``uint8``): 255.
    * Integers of 2+ bytes (``int16``, ``uint16``, ...): 10000.
    * Floats: 10000 if the finite maximum of ``data`` is above 1000 (legacy
      float-stored DN), else 1 (already reflectance).

    Args:
        dtype: Source band dtype, e.g. ``src.dtypes[0]``.
        data: Band values (masked entries and non-finite values are ignored);
            only read for float dtypes.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return 255.0 if dt.itemsize == 1 else 10000.0
    values = np.ma.filled(np.ma.asarray(data, dtype=np.float64), np.nan)
    finite = values[np.isfinite(values)]
    return 10000.0 if finite.size and float(finite.max()) > 1000 else 1.0


def pixel_area_km2(crs: Any, transform: Any) -> float | None:
    """Area of one pixel in km² for a projected ``crs``, else ``None``."""
    if crs is None or not crs.is_projected:
        return None
    return float(abs(transform.a * transform.e) * crs.linear_units_factor[1] ** 2 / 1e6)


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


# --- Upload metadata ----------------------------------------------------------

# Dataset tags that hold an acquisition date, compared case-insensitively.
_DATE_TAGS = ("ACQUISITION_DATE", "ACQUISITIONDATE", "DATE_ACQUIRED", "SENSING_DATE", "SENSING_TIME", "DATETIME")
_TAG_DATE = re.compile(r"^\s*(\d{4})[-:/]?(\d{2})[-:/]?(\d{2})")
# 20230601, 2023-06-01 or 2023_06_01 in a file name, not inside a longer digit run.
_NAME_DATE = re.compile(r"(?<!\d)(\d{4})([-_]?)(\d{2})\2(\d{2})(?!\d)")
_METRES_PER_DEGREE = 111_320.0


def _valid_date(y: str, m: str, d: str) -> str | None:
    try:
        day = datetime.date(int(y), int(m), int(d))
    except ValueError:
        return None
    return day.isoformat() if 1970 <= day.year <= 2100 else None


def _acquisition_date(tags: dict[str, str], filename: str) -> str | None:
    """Date from an acquisition tag, else from the file name, as ``YYYY-MM-DD``."""
    by_key = {k.upper(): v for k, v in tags.items()}
    for key in _DATE_TAGS:
        if (m := _TAG_DATE.match(by_key.get(key, ""))) and (day := _valid_date(*m.groups())):
            return day
    for m in _NAME_DATE.finditer(filename):
        if day := _valid_date(m.group(1), m.group(3), m.group(4)):
            return day
    return None


def _resolution_m(src: Any) -> float | None:
    """Pixel size in metres; approximate (from the y pixel size) for geographic CRSs."""
    if src.crs is None:
        return None
    if src.crs.is_geographic:
        metres = abs(src.res[1]) * _METRES_PER_DEGREE
    else:
        metres = abs(src.res[0]) * src.crs.linear_units_factor[1]
    return round(float(metres), 1)


def _guess_sensor(filename: str, codes: dict[str, int], prefixed: bool, resolution_m: float | None) -> str | None:
    """Sensor from the file name (HLS ``hlsl``/``hlss``), else from the band-code family and pixel size."""
    name = filename.lower()
    if "hlsl" in name or "hls.l30" in name:
        return "HLS L30"
    if "hlss" in name or "hls.s30" in name:
        return "HLS S30"
    family = _code_family(codes)
    if family == "sentinel2":
        return "Sentinel-2"
    if family == "landsat":
        # SR_Bn is the Landsat Collection 2 naming; plain Bn at 30 m is how HLS L30 names its bands.
        # Bn within B1-B7 at 10/20 m is a Sentinel-2 subset (e.g. B2-B4) that reads as Landsat codes.
        if prefixed:
            return "Landsat 8/9"
        if resolution_m is not None and abs(resolution_m - 30) <= 1:
            return "HLS L30"
        if resolution_m is not None and any(abs(resolution_m - r) <= 1 for r in (10, 20)):
            return "Sentinel-2"
    return None


def _band_summary(codes: list[str | None]) -> str:
    """``B2–B6`` for a run of three or more consecutive codes, ``B2–B4, B8, B11, B12`` otherwise."""
    known = [c for c in codes if c]
    if not known or len(known) != len(codes):
        return f"{len(codes)} band" + ("" if len(codes) == 1 else "s")
    runs: list[list[str]] = []
    for code in known:
        last = runs[-1][-1][1:] if runs else ""
        if last.isdigit() and code[1:].isdigit() and int(code[1:]) == int(last) + 1:
            runs[-1].append(code)
        else:
            runs.append([code])
    return ", ".join(f"{r[0]}–{r[-1]}" if len(r) >= 3 else ", ".join(r) for r in runs)


def format_size(size_bytes: int) -> str:
    """``4180159`` -> ``"4.0 MB"`` (binary units, one decimal)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    size = size_bytes / 1024
    for unit in ("KB", "MB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def metadata_line(sensor: str | None, band_summary: str | None, resolution_m: float | None, date: str | None) -> str:
    """``"HLS L30 · B2–B6 · 30 m · 2023-06-01"``, skipping the parts that are unknown."""
    res = f"{resolution_m:g} m" if resolution_m is not None else None
    return " · ".join(p for p in (sensor, band_summary, res, date) if p)


def describe_raster(src: Any, filename: str, size_bytes: int | None = None) -> RasterMetadata:
    """Best-guess metadata of an open raster, for the upload response and the attachment chip.

    Each part is extracted on its own, so an odd file yields a partial result instead of an error.

    Args:
        src: An open ``rasterio`` dataset.
        filename: The file name the user gave (the date and the HLS hint are read from it).
        size_bytes: File size, if known.
    """
    meta = RasterMetadata(size_bytes=size_bytes)
    if size_bytes is not None:
        meta.size_label = format_size(size_bytes)

    codes: dict[str, int] = {}
    prefixed = False
    for i, raw in enumerate(src.descriptions, start=1):
        desc = (raw or "").strip()
        m = _BAND_CODE.match(desc.lower())
        if m:
            codes.setdefault(m.group(1), i)
            prefixed = prefixed or desc.lower().startswith(("sr_", "st_"))
        meta.bands.append(BandInfo(index=i, code=f"B{m.group(1).upper()}" if m else None, description=desc or None))

    try:
        meta.dtype = str(src.dtypes[0])
    except Exception as exc:
        logger.warning("Upload metadata: no dtype for %s: %s", filename, exc)
    try:
        meta.crs = src.crs.to_string() if src.crs else None
        meta.resolution_m = _resolution_m(src)
    except Exception as exc:
        logger.warning("Upload metadata: no CRS/resolution for %s: %s", filename, exc)
    try:
        if src.crs is not None:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            meta.bounds_wgs84 = [round(float(v), 6) for v in (west, south, east, north)]
    except Exception as exc:
        logger.warning("Upload metadata: no WGS84 bounds for %s: %s", filename, exc)
    try:
        meta.date = _acquisition_date(src.tags(), filename)
    except Exception as exc:
        logger.warning("Upload metadata: no date for %s: %s", filename, exc)

    meta.sensor = _guess_sensor(filename, codes, prefixed, meta.resolution_m)
    meta.line = metadata_line(meta.sensor, _band_summary([b.code for b in meta.bands]), meta.resolution_m, meta.date)
    return meta
