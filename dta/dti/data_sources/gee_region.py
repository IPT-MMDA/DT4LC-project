"""Earth Engine fetch for a region sent from the map (``POST /v1/jobs`` with ``region``).

The registry items ``input/earth-engine`` (one composite, ``RasterPath``) and
``input/earth-engine-before`` / ``input/earth-engine-after`` (two composites,
``RasterPathBefore`` / ``RasterPathAfter``) run :func:`run`. No prompt plans them:
when a job has a region and no files, :func:`inject_region_fetch` swaps the plan's
file inputs for them.

Dates. A single-date analysis fetches one median composite over ``start_date`` to
``end_date``. A change analysis (a plan with before/after inputs) fetches two, each
over a window of ±N days around ``start_date`` and around ``end_date``, with
N = :data:`CHANGE_WINDOW_DAYS` (8, one HLS L30 revisit), cut down so the windows stay
apart: N = min(8, (days between the dates − 1) // 2). For 2023-06-01 → 2023-06-09 that
is ±3 days: 2023-05-29 → 06-04 and 2023-06-06 → 06-12. Dates are inclusive.

Datasets. The names the map offers map to the collections the existing fetchers use
(:mod:`gee_landsat`, :mod:`gee_sentinel2`, :mod:`gee_modis`); see :data:`DATASETS`.
The fetch covers the region's bounding box (a polygon's bounds).

Nothing here calls Earth Engine at import time; :func:`run` checks
:func:`gee_common.is_configured` before ``ee.Initialize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from typing import Any
import uuid

from dta.dti.schemas import ExecutionPlan

logger = logging.getLogger(__name__)

EE_NOT_CONFIGURED_MESSAGE = "Earth Engine isn't configured on this server, so I can't fetch imagery for this region."
CHANGE_WINDOW_DAYS = 8

FETCH_SINGLE = "input/earth-engine"
FETCH_BEFORE = "input/earth-engine-before"
FETCH_AFTER = "input/earth-engine-after"
# File input -> the fetch that replaces it for a region job.
FETCH_FOR = {
    "input/file": FETCH_SINGLE,
    "input/file-before": FETCH_BEFORE,
    "input/file-after": FETCH_AFTER,
    FETCH_SINGLE: FETCH_SINGLE,
    FETCH_BEFORE: FETCH_BEFORE,
    FETCH_AFTER: FETCH_AFTER,
}
CONFIG_KEYS = frozenset({"name", "dataset", "bbox", "start_date", "end_date", "bands", "cloud_max"})


class EarthEngineNotConfiguredError(RuntimeError):
    """A region job on a server without Earth Engine credentials (``failed`` code ``EE_NOT_CONFIGURED``)."""

    def __init__(self, message: str = EE_NOT_CONFIGURED_MESSAGE) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Dataset:
    source: str  # fetcher: "landsat-8", "sentinel-2" or "modis" (server/routes/gee.py dataset ids)
    scale_m: int  # export resolution
    bands: tuple[str, ...]  # band codes a request may name
    default_bands: tuple[str, ...]
    ee_prefix: str = ""  # Earth Engine band name = prefix + code (Landsat: SR_B5)


_LANDSAT = Dataset(
    "landsat-8",
    30,
    ("B1", "B2", "B3", "B4", "B5", "B6", "B7"),
    ("B2", "B3", "B4", "B5", "B6", "B7"),
    "SR_",
)
_S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12")
_S2_DEFAULT = ("B2", "B3", "B4", "B8", "B11", "B12")
_MODIS_BANDS = tuple(f"sur_refl_b0{i}" for i in range(1, 8))

# Map dataset names -> existing fetchers. HLS L30 is read from Landsat 8 Collection 2 L2 and
# HLS S30 from Sentinel-2 SR (at 30 m): the existing fetchers have no HLS collection.
DATASETS: dict[str, Dataset] = {
    "HLS L30": _LANDSAT,
    "Landsat 8/9": _LANDSAT,
    "HLS S30": Dataset("sentinel-2", 30, _S2_BANDS, _S2_DEFAULT),
    "Sentinel-2": Dataset("sentinel-2", 10, _S2_BANDS, _S2_DEFAULT),
    "MODIS": Dataset("modis", 500, _MODIS_BANDS, _MODIS_BANDS),
}


def band_code(dataset: str, band: str) -> str | None:
    """``band`` as one of the dataset's codes (``SR_B5`` -> ``B5`` for Landsat), or ``None`` if it has no such band."""
    spec = DATASETS[dataset]
    code = band.strip()
    if spec.ee_prefix and code.upper().startswith(spec.ee_prefix):
        code = code[len(spec.ee_prefix) :]
    by_upper = {b.upper(): b for b in spec.bands}
    return by_upper.get(code.upper())


def date_range(start: str, end: str) -> str:
    """``"2023-06-01 → 06-09"``; the year is repeated only when it changes, one date when they are equal."""
    if start == end:
        return start
    return f"{start} → {end[5:] if start[:4] == end[:4] else end}"


def change_windows(start: str, end: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Before and after windows, ±N days around ``start`` and ``end`` (see the module docstring)."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    half = timedelta(days=min(CHANGE_WINDOW_DAYS, max((e - s).days - 1, 0) // 2))
    return ((s - half).isoformat(), (s + half).isoformat()), ((e - half).isoformat(), (e + half).isoformat())


def fetch_config(region: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    """Step config of one fetch: the region's bounds, dataset, bands and cloud limit over ``start``..``end``."""
    return {
        "name": f"Fetch {region['dataset']} · {date_range(start, end)}",
        "dataset": region["dataset"],
        "bbox": list(region["bounds"]),
        "start_date": start,
        "end_date": end,
        "bands": list(region["bands"]),
        "cloud_max": region["cloud_max"],
    }


def inject_region_fetch(plan: ExecutionPlan, region: dict[str, Any]) -> ExecutionPlan:
    """Replace ``plan``'s file inputs with Earth Engine fetches of ``region`` (the job's region echo).

    A plan with before/after inputs gets two fetches (:func:`change_windows`); a plan with one
    input gets one over the whole date range. Existing fetch steps get their config rewritten,
    so an edited plan cannot fetch anything but the job's region.
    """
    change = any(FETCH_FOR.get(s.uses) in (FETCH_BEFORE, FETCH_AFTER) for s in plan.steps)
    start, end = region["start_date"], region["end_date"]
    before, after = change_windows(start, end) if change else ((start, end), (start, end))
    for step in plan.steps:
        fetch = FETCH_FOR.get(step.uses)
        if fetch is None:
            continue
        window = before if fetch == FETCH_BEFORE else after if fetch == FETCH_AFTER else (start, end)
        step.uses = fetch
        step.binds = {}
        step.config = fetch_config(region, *window)
    return plan


def check_config(config: dict[str, Any]) -> None:
    """ValueError unless ``config`` is a fetch step's config as :func:`fetch_config` writes it."""
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Earth Engine fetch takes no {', '.join(unknown)}.")
    if config.get("dataset") not in DATASETS:
        raise ValueError(f"Unknown dataset {config.get('dataset')!r}; one of {', '.join(DATASETS)}.")
    bbox = config.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int | float) for v in bbox)):
        raise ValueError("Earth Engine fetch needs bbox [west, south, east, north].")
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError(f"Earth Engine fetch bbox {bbox} is not [west, south, east, north] in degrees.")
    try:
        if date.fromisoformat(config["start_date"]) > date.fromisoformat(config["end_date"]):
            raise ValueError("Earth Engine fetch start_date is after end_date.")
    except (KeyError, TypeError) as e:
        raise ValueError("Earth Engine fetch needs start_date and end_date (YYYY-MM-DD).") from e
    bands = config.get("bands") or []
    if not isinstance(bands, list) or any(band_code(config["dataset"], str(b)) is None for b in bands):
        raise ValueError(f"Earth Engine fetch bands {bands} are not all {config['dataset']} bands.")


def _fetcher(source: str) -> Any:
    if source == "landsat-8":
        from .gee_landsat import fetch_landsat_composite

        return fetch_landsat_composite
    if source == "sentinel-2":
        from .gee_sentinel2 import fetch_sentinel2_composite

        return fetch_sentinel2_composite
    from .gee_modis import fetch_modis_composite

    return fetch_modis_composite


def fetch(config: dict[str, Any]) -> str:
    """Download the median composite ``config`` describes as a GeoTIFF; returns its path.

    Uses the existing fetchers and :func:`gee_export.export_gee_image_to_geotiff` (direct
    download, about 30 MB at most). Band descriptions are set to the band codes (``B5``).
    """
    import rasterio

    from .gee_export import estimate_export_size, export_gee_image_to_geotiff

    spec = DATASETS[config["dataset"]]
    codes = [str(band_code(config["dataset"], b)) for b in config["bands"]]
    ee_bands = [spec.ee_prefix + c for c in codes]
    bbox = [float(v) for v in config["bbox"]]
    size = estimate_export_size(bbox, spec.scale_m, len(ee_bands))
    if not size["can_use_direct_download"]:
        raise RuntimeError(
            f"This region is too large for a direct Earth Engine download at {spec.scale_m} m "
            f"(about {size['estimated_size_mb']:.0f} MB; the limit is 30 MB). "
            "Draw a smaller region or pick fewer bands."
        )
    start, end = config["start_date"], config["end_date"]
    end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()  # Earth Engine's filterDate end
    result = _fetcher(spec.source)(bbox, start, end_exclusive, ee_bands, float(config["cloud_max"]), return_image=True)
    if not result.get("ok") or result.get("image") is None:
        error = str(result.get("error") or "no image")
        if error.startswith("No images found"):
            raise RuntimeError(
                f"Earth Engine has no {config['dataset']} image of this region between {start} and {end} "
                f"with ≤{config['cloud_max']} % cloud."
            )
        raise RuntimeError(f"Earth Engine fetch failed: {error}")
    filename = f"region_{spec.source}_{start}_{end}_{uuid.uuid4().hex[:8]}"
    export = export_gee_image_to_geotiff(result["image"], bbox, ee_bands, spec.scale_m, filename)
    if export.get("status") != "completed":
        raise RuntimeError(f"Earth Engine export failed: {export.get('error', 'unknown error')}")
    path = str(export["file_path"])
    with rasterio.open(path, "r+") as dst:
        for i, code in enumerate(codes[: dst.count], start=1):
            dst.set_band_description(i, code)
    return path


def run(config: dict[str, Any]) -> str:
    """Runner of the ``input/earth-engine*`` items: the fetched GeoTIFF's path.

    Raises:
        EarthEngineNotConfiguredError: No credentials (checked before ``ee.Initialize``), or they were refused.
    """
    from . import gee_common

    if not gee_common.is_configured():
        raise EarthEngineNotConfiguredError()
    if not gee_common.initialize_gee():
        raise EarthEngineNotConfiguredError(
            "Earth Engine refused this server's credentials, so I can't fetch imagery for this region."
        )
    logger.info(f"Fetching {config['dataset']} {config['start_date']}..{config['end_date']} for {config['bbox']}")
    return fetch(config)
