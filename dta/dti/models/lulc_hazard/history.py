"""Runner of ``models/lulc-history``: how the land cover of an area changed since 2017.

No model: the map of every year of a label source (:data:`.features.SOURCES`: Esri by default, or KPI-Ukraine; 10 m,
3 x 3 mode to 30 m) for the footprint of the input raster. Esri changes its mind from year to year (about four in ten
class changes return), so the main comparison is
between the majority class of the first three years and that of the last three; the single-year pair is reported
beside it. Writes the two majority maps and the changed cells, area per class per year, the transition matrix, and
sentences built from these numbers only.

:func:`run_prewar` (``algorithms/lulc-since-prewar``) is a fixed comparison: the KPI-Ukraine map of 2021, the last
pre-war year, against 2024, the latest year of one consistent KPI series, as single years; plus the area the map
marks as not cultivated (raw class 20, Crop in C7x) per year. It describes the maps; it does not attribute change
to the war.
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
PREWAR, LATEST = 2021, 2024  # run_prewar: the last pre-war KPI year, the latest of one consistent KPI series
NOT_CULTIVATED = 20  # KPI raw code "Not cultivated" (abandoned cropland); Crop in the C7x legend
GARDENS, ARTIFICIAL = 18, 1  # KPI raw codes "Gardens, parks" (Tree in C7x) and "Artificial" (Built)
FINE_KM2 = features.FINE * features.FINE / 1e6
PREWAR_NOTE = (
    "KPI-Ukraine maps of 2021 and 2024 compared as single years: a description of the maps, not the effect of the "
    "war. Known map artefacts in this period: new forest is recorded only in the 2024 map (the 2021-2023 maps add "
    "none), 2024 relabels forest and gardens (both Tree here), and land the map marks as not cultivated counts as "
    "Crop in this legend; its area is shown separately."
)


def label_years(
    ee: Any, grid: features.Grid, years: list[int], source: features.LabelSource = features.ESRI
) -> np.ndarray:
    """``(years, ny, nx)`` class index of ``source`` on the core of ``grid``."""
    return np.stack(
        [features.build_features(grid, y, [], ee=ee, check_years=False, source=source).start_class for y in years]
    )


def compared(cls: np.ndarray, years: list[int], k: int, pair: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """The two maps compared: the years of ``pair``, else the majorities of the first and last :data:`PERIOD` years."""
    if pair is not None:
        return cls[years.index(pair[0])], cls[years.index(pair[1])]
    return stats.majority(cls[:PERIOD], k), stats.majority(cls[-PERIOD:], k)


def summarise(
    cls: np.ndarray,
    years: list[int],
    names: tuple[str, ...] = features.ESRI_CLASSES,
    pair: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Statistics of a ``(years, ny, nx)`` class stack: the result's ``statistics``. ``pair``: compare those two
    single years instead of the first and last periods."""
    k = len(names)
    first, last = years[:PERIOD], years[-PERIOD:]
    start, end = compared(cls, years, k, pair)
    m = stats.matrix(start, end, k)
    single = pair or (years[0], years[-1])
    m1 = stats.matrix(cls[years.index(single[0])], cls[years.index(single[1])], k)
    valid = max(int(m.sum()), 1)
    if pair is None:
        lead = f"Between {first[0]}–{first[-1]} and {last[0]}–{last[-1]}:"
        periods = {"start": [first[0], first[-1]], "end": [last[0], last[-1]]}
    else:
        lead = f"Between {pair[0]} and {pair[1]}:"
        periods = {"start": [pair[0]], "end": [pair[1]]}
    return {
        "years": years,
        "area_km2": {
            n: [stats.km2(c) for c in (cls == i + 1).reshape(len(years), -1).sum(1)]
            for i, n in enumerate(names)
            if (cls == i + 1).any()
        },
        "periods": periods,
        "matrix": stats.matrix_table(m, names),
        "classes": stats.class_table(m.sum(1), m.sum(0), names),
        "changed_percent": round(100.0 * float(m.sum() - np.trace(m)) / valid, 1),
        "changed_percent_single_years": round(100.0 * float(m1.sum() - np.trace(m1)) / max(int(m1.sum()), 1), 1),
        "single_year_pair": list(single),
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


def run_prewar(RasterPath: str) -> dict[str, Any]:  # noqa: N803
    """KPI-Ukraine land cover of the footprint of ``RasterPath``: 2021 against 2024, and the area marked not
    cultivated per year from 2017 (module docstring)."""
    src = features.KPI
    bbox = footprint_wgs84(RasterPath)
    grid = features.grid_for_bbox(bbox)
    if grid.area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"This area is {grid.area_km2:.0f} km²; the land-cover comparison runs on at most {MAX_AREA_KM2:.0f} km². "
            "Draw a smaller region."
        )
    ee = _ee()
    years = list(range(features.T0_FIRST, LATEST + 1))
    have = features.available_years(ee, grid, years, src)["labels"]
    missing = [y for y in years if y not in have]
    if missing:
        raise features.FeatureUnavailableError(
            f"{src.title} has no map of this area for {missing}; it covers Ukraine."
        )
    cls, fallow, raw = prewar_maps(ee, grid, years)
    extra: dict[str, Any] = {
        "not_cultivated_km2": {"years": years, "km2": [stats.km2(v / 9) for v in fallow]},
        "checked": PREWAR_NOTE,
    }
    a, b = fallow[years.index(PREWAR)] * FINE_KM2, fallow[years.index(LATEST)] * FINE_KM2
    extra["summary_extra"] = [
        f"Land the map marks as not cultivated (counted as Crop above): {stats.km2_text(a)} in {PREWAR}, "
        f"{stats.km2_text(b)} in {LATEST}."
    ]
    tree = [c for c in range(256) if src.lut[c] == src.classes.index("Tree") + 1]
    built = np.isin(raw[PREWAR], tree) & (raw[LATEST] == ARTIFICIAL)
    if built.any():
        gardens = float((raw[PREWAR][built] == GARDENS).mean())
        extra["summary_extra"].append(
            f"Of the Tree that became Built ({stats.km2_text(built.sum() * FINE_KM2)}), {100 * gardens:.0f} % was "
            f"gardens and parks in {PREWAR}, the rest forest."
        )
    return write_result(cls, years, grid, bbox, src, pair=(PREWAR, LATEST), extra=extra)


def prewar_maps(ee: Any, grid: features.Grid, years: list[int]) -> tuple[np.ndarray, list[int], dict[int, np.ndarray]]:
    """Per year, the KPI C7x class on the core of ``grid`` and the count of 10 m core pixels with raw class 20, from
    one pull of the raw codes; and the raw 10 m core codes of :data:`PREWAR` and :data:`LATEST`."""
    src = features.KPI
    core10 = tuple(slice(sl.start * 3, sl.stop * 3) for sl in grid.core)
    cls, fallow, raw = [], [], {}
    for y in years:
        codes = features.pull(
            ee, features.label_image(ee, grid, y, src), ["b1"], grid, features.FINE, features.TILE["labels"], nodata=0
        )[0]
        codes = np.nan_to_num(codes, nan=0).astype(np.uint8)
        cls.append(features.mode3(src.lut[codes], len(src.classes))[grid.core])
        fallow.append(int((codes[core10] == NOT_CULTIVATED).sum()))
        if y in (PREWAR, LATEST):
            raw[y] = codes[core10]
    return np.stack(cls), fallow, raw


def write_result(
    cls: np.ndarray,
    years: list[int],
    grid: features.Grid,
    bbox: list[float],
    source: features.LabelSource = features.ESRI,
    pair: tuple[int, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names, k = source.classes, len(source.classes)
    statistics = summarise(cls, years, names, pair)
    if extra:
        statistics["summary"] += extra.pop("summary_extra", [])
        statistics.update(extra)
    start, end = compared(cls, years, k, pair)
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
            "note": PREWAR_NOTE
            if pair
            else "Annual maps flicker between years; the periods' majority class damps it.",
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
