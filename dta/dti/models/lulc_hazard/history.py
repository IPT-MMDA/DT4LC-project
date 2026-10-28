"""Runner of ``models/lulc-history``: how the land cover of an area changed since 2017.

No model: the map of every year of a label source (:data:`.features.SOURCES`: Esri by default, or KPI-Ukraine; 10 m,
3 x 3 mode to 30 m) for the footprint of the input raster. Esri changes its mind from year to year (about four in ten
class changes return), so the main comparison is
between the majority class of the first three years and that of the last three; the single-year pair is reported
beside it. Writes the two majority maps and the changed cells, area per class per year, the transition matrix, and
sentences built from these numbers only.
"""

from __future__ import annotations

from datetime import date
import logging
from typing import Any

import numpy as np

from dta.dti import artifact_files

from . import features, stats
from .inference import MAX_AREA_KM2, _ee, footprint_wgs84

logger = logging.getLogger(__name__)

PERIOD = 3  # years in each majority period


def label_years(
    ee: Any, grid: features.Grid, years: list[int], source: features.LabelSource = features.ESRI
) -> np.ndarray:
    """``(years, ny, nx)`` class index of ``source`` on the core of ``grid``."""
    return np.stack(
        [features.build_features(grid, y, [], ee=ee, check_years=False, source=source).start_class for y in years]
    )


def summarise(cls: np.ndarray, years: list[int], names: tuple[str, ...] = features.ESRI_CLASSES) -> dict[str, Any]:
    """Statistics of a ``(years, ny, nx)`` class stack: the result's ``statistics``."""
    k = len(names)
    first, last = years[:PERIOD], years[-PERIOD:]
    start, end = stats.majority(cls[:PERIOD], k), stats.majority(cls[-PERIOD:], k)
    m = stats.matrix(start, end, k)
    m1 = stats.matrix(cls[0], cls[-1], k)
    valid = max(int(m.sum()), 1)
    lead = f"Between {first[0]}–{first[-1]} and {last[0]}–{last[-1]}:"
    return {
        "years": years,
        "area_km2": {
            n: [stats.km2(c) for c in (cls == i + 1).reshape(len(years), -1).sum(1)]
            for i, n in enumerate(names)
            if (cls == i + 1).any()
        },
        "periods": {"start": [first[0], first[-1]], "end": [last[0], last[-1]]},
        "matrix": stats.matrix_table(m, names),
        "classes": stats.class_table(m.sum(1), m.sum(0), names),
        "changed_percent": round(100.0 * float(m.sum() - np.trace(m)) / valid, 1),
        "changed_percent_single_years": round(100.0 * float(m1.sum() - np.trace(m1)) / max(int(m1.sum()), 1), 1),
        "single_year_pair": [years[0], years[-1]],
        "summary_lead": lead,
        "summary": stats.sentences(m, "{past}", names),
        "transitions": stats.top_transitions(m, names=names),
        "changed_km2": stats.changed_km2(m),
    }


def run(RasterPath: str, last_year: int | None = None, source: str = "esri") -> dict[str, Any]:  # noqa: N803
    """Land-cover history of the footprint of ``RasterPath`` from 2017 to the source's latest year."""
    if source not in features.SOURCES:
        raise ValueError(f"Unknown land-cover source {source!r}; one of {', '.join(features.SOURCES)}.")
    src = features.SOURCES[source]
    bbox = footprint_wgs84(RasterPath)
    grid = features.grid_for_bbox(bbox)
    if grid.area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"This area is {grid.area_km2:.0f} km²; the land-cover history runs on at most {MAX_AREA_KM2:.0f} km². "
            "Draw a smaller region."
        )
    ee = _ee()
    last = last_year or date.today().year
    have = features.available_years(ee, grid, range(features.T0_FIRST, last + 1), src)["labels"]
    years = sorted(have)
    if len(years) < 2 * PERIOD:
        raise features.FeatureUnavailableError(f"{src.title} has {len(years)} years for this area; need {2 * PERIOD}.")
    cls = label_years(ee, grid, years, src)
    return write_result(cls, years, grid, bbox, src)


def write_result(
    cls: np.ndarray,
    years: list[int],
    grid: features.Grid,
    bbox: list[float],
    source: features.LabelSource = features.ESRI,
) -> dict[str, Any]:
    names, k = source.classes, len(source.classes)
    statistics = summarise(cls, years, names)
    start, end = stats.majority(cls[:PERIOD], k), stats.majority(cls[-PERIOD:], k)
    style = artifact_files.class_style({i + 1: source.colours[n] for i, n in enumerate(names)})
    rasters = {
        "class_start": np.where(start > 0, start, np.nan).astype(np.float32),
        "class_end": np.where(end > 0, end, np.nan).astype(np.float32),
        "changed": np.where((start > 0) & (end > 0) & (start != end), end, np.nan).astype(np.float32),
    }
    result: dict[str, Any] = {
        "index_type": "land_cover_history",
        "statistics": statistics,
        "metadata": {
            "source": f"{source.title}, 3 x 3 majority to 30 m",
            "legend": list(names),
            "bbox_wgs84": [round(v, 6) for v in bbox],
            "grid": {"crs": grid.crs, "resolution_m": features.RES, "width": grid.nx, "height": grid.ny},
            "note": "Annual maps flicker between years; the periods' majority class damps it.",
        },
    }
    return artifact_files.save(
        result,
        rasters,
        {},
        crs=grid.crs,
        transform=grid.transform(),
        styles=dict.fromkeys(rasters, style),
        bare={f"{k}_map": k for k in rasters},
    )
