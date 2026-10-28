"""Runners of ``models/lulc-change-hazard`` and ``models/lulc-forecast``: where and how land cover may change.

The steps take the input raster only for its footprint (an upload, or the Earth Engine composite of a region drawn
on the map; its dates play no part), build the model's features for that footprint from Earth Engine
(:mod:`.features`) at the origin year ``t0`` = the latest year with both a label map of the bundle's ``target``
(Esri or KPI-Ukraine, :data:`.features.SOURCES`) and an AlphaEarth embedding of the area, and apply a trained bundle.

:func:`run` (hazards) writes, on a 30 m grid, per hazard group of the bundle at that group's longest checked horizon
``h`` (change between ``t0`` and ``t0 + h``):

- ``change_rank`` (the ``any_change`` group) and ``rank_<group>``: percentile (0-100) of the group's probability
  among the training cells that could make that change (national rank);
- ``<key>_local``: the same score ranked among the area's own cells (where in this area first);
- ``start_class``: the class at ``t0`` (raster value i = the bundle's ``legend[i - 1]``).

:func:`run_forecast` writes the map at ``t0 + h``: the amount of each change follows the trend of the latest
``h``-year pair in a 12 km square around the area (Markov), and the model places it, highest probability first
(:mod:`.stats`). The pair ends at ``t0``, or at the bundle's ``trend_last`` year if that is earlier (a start map the
model may read but whose change from the year before is not a trend). The bundle states up to which horizon this
placement and these amounts were checked. Results carry the legend in ``metadata.legend``.

The values are ranks, not probabilities: the probability level of this model does not carry over to a new period,
only its ordering of cells does. A percentile of 90 means "more likely to change than 90 % of the training cells".

Bundle format 2 (``<model dir>/lulc_hazard.json``)::

    {"format": "dt4lc-lulc-hazard/2", "model_id", "version", "target": "esri" | "kpi", "legend", "columns",
     "t0_min", "dem", "grid", "domain",                     # as in format 1, below
     "trend_last": int,                                     # optional: the forecast's trend pair ends by this year
     "groups": {"<group>": {"title": str, "from": [classes], "to": [classes] | null}},   # null: any other class
     "horizons": {"<h>": {"pair": [y0, y1], "from_classes": {...as format 1...},
                          "reference_quantiles": {"<group>": [sorted floats]}}},
     "served": {"hazards": {"<group>": h}, "map": h, "amounts_expected": {"<h>": bool}},   # 0: not served
     "skill": {"<h>": {...}}}                               # optional, copied into the result

Format 1 is one horizon (``interval_years``) whose groups are ``any_change`` and the listed transitions; it serves
hazards only.

Bundle format 1 (one file, so the model manager downloads it as one)::

    {"format": "dt4lc-lulc-hazard/1", "model_id": str, "version": str, "target": "esri",
     "legend": [8 Esri class names], "columns": [feature names, in the boosters' order],
     "interval_years": float, "t0_min": int,   # optional, default 2017: earliest origin year it accepts
     "dem": "glo30_2024_1" | "glo30",          # optional; a bundle without it was built on dataset v1.0: "glo30"
     "grid": {"crs": "EPSG:32636" | "auto", "origin": [x, y]},
     "domain": {"bbox": [west, south, east, north], "name": str},
     "from_classes": {"<class>": {"to": [class names], "prior_correction": [float per "to"],
                                  "booster": <XGBoost JSON model> | null}},
     "transitions": [["<from>", "<to>"], ...],
     "reference_quantiles": {"any_change": [sorted floats], "<from>-><to>": [sorted floats], ...},
     "skill": {...}   # optional, copied into the result's metadata
    }

Per from-class, as in training: the booster's class probabilities are multiplied by ``prior_correction`` and
renormalised; a from-class with one target class has no booster (it always stays); a from-class absent from the
bundle gets no rank (NaN). The reference quantiles are the probabilities' quantiles at evenly spaced levels over the
training cells, written when the bundle is exported.
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from dta.dti import artifact_files

from . import features, stats

logger = logging.getLogger(__name__)

BUNDLE_FILE = "lulc_hazard.json"
FORMAT = "dt4lc-lulc-hazard/1"
FORMAT_2 = "dt4lc-lulc-hazard/2"
LEGACY_DEM = "glo30"  # the DEM of a bundle that names none (dataset v1.0)
MAX_AREA_KM2 = 400.0  # about 20 x 20 km: 64 embedding bands of 450k cells stay near 120 MB in memory
INDEX_TYPE = "change_rank"
ANY = "any_change"
TREND_CELLS = 400  # side of the square whose latest pair sets the forecast's amounts: 12 km, as the backtest windows
RANK_RAMP = ("#f2efe6", "#e9c46a", "#d1773a", "#a3322a", "#5c1a1b")  # low -> high rank
ESRI_COLOURS = features.ESRI.colours
# One hue per hazard; its risk levels are shades of it (:func:`level_colours`).
GROUP_COLOURS = {
    ANY: "#6a3d9a",
    "urban_expansion": "#b2182b",
    "forest_loss": "#8c510a",
    "cropland_abandonment": "#d95f02",
    "recultivation": "#2166ac",
    "overgrowth": "#1b7837",
}
OTHER_COLOUR = "#4d4d4d"  # a group of a format-1 bundle (a single transition)
# Risk levels on the national rank, highest first: (raster value, name, lowest percentile, wording).
# Cells below the last level are left uncoloured, so the start map drawn under them shows through.
RISK_LEVELS = (
    (3, "High", 90.0, "national top 10 %"),
    (2, "Elevated", 75.0, "top 10–25 %"),
    (1, "Moderate", 50.0, "top 25–50 %"),
)
LEVEL_TINT = {3: 1.0, 2: 0.6, 1: 0.3}  # share of the hazard's hue mixed with white

GROUP_TITLES = {
    ANY: "Any land-cover change",
    "urban_expansion": "Urban expansion",
    "forest_loss": "Forest loss",
    "cropland_abandonment": "Cropland abandonment",
    "recultivation": "Re-cultivation",
    "overgrowth": "Overgrowth and afforestation",
}


class BundleError(ValueError):
    """The model directory holds no usable bundle."""


class OutsideDomainError(ValueError):
    """The area lies outside the region the model was trained on."""


# ----------------------------------------------------------------- bundle


def load_bundle(model_dir: str | Path) -> dict[str, Any]:
    """The checked bundle of ``model_dir`` in format 2 (a format-1 file is converted; cached per file and mtime)."""
    path = Path(model_dir) / BUNDLE_FILE
    if not path.is_file():
        raise BundleError(f"No {BUNDLE_FILE} in {model_dir}: the change-hazard model is not installed.")
    return _load_bundle(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load_bundle(path: str, mtime_ns: int) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    check_bundle(bundle)
    return as_format_2(bundle)


def source_of(bundle: dict[str, Any]) -> features.LabelSource:
    """The label source of the bundle's ``target``."""
    source = features.SOURCES.get(bundle.get("target", ""))
    if source is None:
        raise BundleError(f"Bundle target must be one of {', '.join(features.SOURCES)}, not {bundle.get('target')!r}.")
    return source


def _check_common(bundle: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in bundle]
    if missing:
        raise BundleError(f"Bundle lacks {', '.join(missing)}.")
    source = source_of(bundle)
    if list(bundle.get("legend", [])) != list(source.classes):
        raise BundleError(f"Bundle legend must be the {source.key} legend " + ", ".join(source.classes) + ".")
    try:
        features.check_columns(bundle["columns"], source)
        features.check_dem(bundle.get("dem", LEGACY_DEM))
    except ValueError as e:
        raise BundleError(str(e)) from e


def _check_from_classes(from_classes: dict[str, Any], names: tuple[str, ...], where: str = "") -> None:
    for name, spec in from_classes.items():
        if name not in names or any(t not in names for t in spec.get("to", [])):
            raise BundleError(f"{where}from_classes[{name!r}] names a class outside the bundle's legend.")
        if len(spec.get("prior_correction", [])) != len(spec["to"]):
            raise BundleError(f"{where}from_classes[{name!r}]: prior_correction needs one value per 'to' class.")
        if spec.get("booster") is None and len(spec["to"]) != 1:
            raise BundleError(f"{where}from_classes[{name!r}] has {len(spec['to'])} target classes and no booster.")


def _check_quantiles(q: Any, key: str) -> None:
    if not q or len(q) < 11 or np.any(np.diff(q) < 0):
        raise BundleError(f"reference_quantiles[{key!r}] must be at least 11 sorted values.")


def check_bundle(bundle: dict[str, Any]) -> None:
    """BundleError unless ``bundle`` has the shape of format 1 or 2 (module docstring)."""
    fmt = bundle.get("format")
    if fmt == FORMAT:
        _check_common(
            bundle,
            (
                "model_id",
                "columns",
                "interval_years",
                "grid",
                "domain",
                "from_classes",
                "transitions",
                "reference_quantiles",
            ),
        )
        _check_from_classes(bundle["from_classes"], source_of(bundle).classes)
        for k in [ANY] + [f"{a}->{b}" for a, b in bundle["transitions"]]:
            _check_quantiles(bundle["reference_quantiles"].get(k), k)
        for a, b in bundle["transitions"]:
            if a not in bundle["from_classes"] or b not in bundle["from_classes"][a]["to"]:
                raise BundleError(f"Transition {a}->{b} is not one the bundle's models predict.")
        return
    if fmt != FORMAT_2:
        raise BundleError(f"Bundle format {fmt!r}; this runner reads {FORMAT!r} and {FORMAT_2!r}.")
    _check_common(bundle, ("model_id", "columns", "grid", "domain", "groups", "horizons", "served"))
    names = source_of(bundle).classes
    for g, spec in bundle["groups"].items():
        classes = list(spec.get("from", [])) + list(spec.get("to") or [])
        if not spec.get("from") or any(c not in names for c in classes):
            raise BundleError(f"groups[{g!r}] needs 'from' classes of the bundle's legend.")
    served = bundle["served"]
    for h, hz in bundle["horizons"].items():
        _check_from_classes(hz["from_classes"], names, f"horizons[{h}].")
        for g, gh in served.get("hazards", {}).items():
            if gh and int(h) <= gh:
                if g not in bundle["groups"]:
                    raise BundleError(f"served hazard {g!r} is not a group of the bundle.")
                _check_quantiles(hz["reference_quantiles"].get(g), f"{h}:{g}")
    needed = {int(v) for v in served.get("hazards", {}).values() if v} | (
        {int(served["map"])} if served.get("map") else set()
    )
    absent = sorted(h for h in needed if str(h) not in bundle["horizons"])
    if absent:
        raise BundleError(f"Served horizons {absent} have no models in the bundle.")


def as_format_2(bundle: dict[str, Any]) -> dict[str, Any]:
    """A checked bundle in format 2; a format-1 bundle becomes one horizon serving its groups' hazards, no map."""
    if bundle["format"] == FORMAT_2:
        return bundle
    h = int(round(bundle["interval_years"]))
    groups = {ANY: {"title": GROUP_TITLES[ANY], "from": list(bundle["from_classes"]), "to": None}}
    refs = {ANY: bundle["reference_quantiles"][ANY]}
    for a, b in bundle["transitions"]:
        g = f"{a.lower()}_to_{b.lower()}"
        groups[g] = {"title": f"{a} to {b}", "from": [a], "to": [b]}
        refs[g] = bundle["reference_quantiles"][f"{a}->{b}"]
    out = {k: v for k, v in bundle.items() if k not in ("from_classes", "transitions", "reference_quantiles")}
    out.update(
        {
            "format": FORMAT_2,
            "groups": groups,
            "horizons": {str(h): {"from_classes": bundle["from_classes"], "reference_quantiles": refs}},
            "served": {"hazards": dict.fromkeys(groups, h), "map": 0, "amounts_expected": {}},
            "skill": {str(h): bundle["skill"]} if bundle.get("skill") else {},
        }
    )
    return out


@lru_cache(maxsize=32)
def _booster(model_id: str, version: str, key: str, text: str) -> Any:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(text, "utf-8"))
    return booster


# ----------------------------------------------------------------- prediction


def probabilities(
    bundle: dict[str, Any], from_classes: dict[str, Any], x: np.ndarray, start: np.ndarray, key: str = ""
) -> np.ndarray:
    """``(cells, classes)`` probability of each legend class at the interval's end; NaN rows where no model applies.

    ``x`` is ``(cells, columns)`` in the bundle's column order, ``start`` the cells' class index (0 = no data).
    Per from-class, as in training: the booster's probabilities times ``prior_correction``, renormalised; a zero row
    means "stays".
    """
    names = source_of(bundle).classes
    out = np.full((start.size, len(names)), np.nan, np.float32)
    for name, spec in from_classes.items():
        idx = np.flatnonzero(start == names.index(name) + 1)
        if idx.size == 0:
            continue
        to = list(spec["to"])
        if spec.get("booster") is None:
            p = np.ones((idx.size, 1))
        else:
            import xgboost as xgb

            booster = _booster(
                bundle["model_id"], str(bundle.get("version", "")), f"{key}:{name}", json.dumps(spec["booster"])
            )
            raw = np.asarray(booster.predict(xgb.DMatrix(x[idx], missing=np.nan)), dtype=np.float64)
            p = np.stack([1.0 - raw, raw], axis=1) if raw.ndim == 1 else raw
            if p.shape[1] != len(to):
                raise BundleError(f"The {name} booster gives {p.shape[1]} classes; the bundle lists {len(to)}.")
        p = p * np.asarray(spec["prior_correction"], dtype=np.float64)[None, :]
        s = p.sum(1, keepdims=True)
        p = np.where(s > 0, p / np.where(s > 0, s, 1.0), 0.0)
        full = np.zeros((idx.size, len(names)))
        full[:, [names.index(t) for t in to]] = p
        stay = names.index(name)
        full[s[:, 0] == 0, stay] = 1.0
        out[idx] = full
    return out


def predict(bundle: dict[str, Any], x: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Format 1: per cell ``P(change)`` and ``P(from -> to)`` per bundle transition; NaN where no model applies."""
    p = probabilities(bundle, bundle["from_classes"], x, start)
    names = source_of(bundle).classes
    p_change = group_score(p, start, {"from": list(bundle["from_classes"]), "to": None}, names)
    p_trans = {f"{a}->{b}": group_score(p, start, {"from": [a], "to": [b]}, names) for a, b in bundle["transitions"]}
    return p_change, p_trans


def group_score(
    p: np.ndarray, start: np.ndarray, group: dict[str, Any], names: tuple[str, ...] = features.ESRI_CLASSES
) -> np.ndarray:
    """Probability of the group's change per cell (NaN outside its from-classes or without a model)."""
    out = np.full(start.shape, np.nan, np.float32)
    eligible = np.isin(start, [names.index(c) + 1 for c in group["from"]])
    rows = np.arange(start.size)
    if group.get("to") is None:
        stay = p[rows, np.maximum(start.astype(np.int64) - 1, 0)]
        out[eligible] = (1.0 - stay)[eligible]
    else:
        out[eligible] = p[:, [names.index(c) for c in group["to"]]].sum(1)[eligible]
    return out


def percentile(p: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Percentile (0-100) of each ``p`` among reference quantiles at evenly spaced levels (mid-rank on ties)."""
    q = np.asarray(quantiles, dtype=np.float64)
    out = np.full(p.shape, np.nan, np.float32)
    ok = np.isfinite(p)
    lo = np.searchsorted(q, p[ok], side="left")
    hi = np.searchsorted(q, p[ok], side="right")
    out[ok] = (100.0 * (lo + hi) / (2.0 * len(q))).astype(np.float32)
    return out


def local_percentile(p: np.ndarray) -> np.ndarray:
    """Percentile (0-100, mid-rank) of each finite ``p`` among the finite values of ``p`` itself."""
    out = np.full(p.shape, np.nan, np.float32)
    ok = np.isfinite(p)
    if ok.any():
        v = np.sort(p[ok])
        lo, hi = np.searchsorted(v, p[ok], "left"), np.searchsorted(v, p[ok], "right")
        out[ok] = (100.0 * (lo + hi) / (2.0 * v.size)).astype(np.float32)
    return out


# ----------------------------------------------------------------- area and features


def footprint_wgs84(raster_path: str) -> list[float]:
    """``[west, south, east, north]`` in degrees of a raster's bounds."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("The input raster has no coordinate reference system, so its area is unknown.")
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))


def check_domain(bbox: list[float], domain: dict[str, Any]) -> None:
    w, s, e, n = domain["bbox"]
    if not (bbox[0] >= w and bbox[1] >= s and bbox[2] <= e and bbox[3] <= n):
        name = domain.get("name", "its training region")
        raise OutsideDomainError(
            f"This area is outside {name}, where the change-hazard model was trained; it has no forecast here."
        )


def _ee() -> Any:
    from dta.dti.data_sources import gee_common
    from dta.dti.data_sources.gee_region import EarthEngineNotConfiguredError

    if not gee_common.initialize_gee():
        raise EarthEngineNotConfiguredError(
            "Earth Engine isn't configured on this server, so I can't build the change-hazard features."
        )
    return gee_common.ee


def _area(RasterPath: str, bundle: dict[str, Any]) -> tuple[list[float], features.Grid]:  # noqa: N803
    bbox = footprint_wgs84(RasterPath)
    check_domain(bbox, bundle["domain"])
    g = bundle["grid"]
    grid = features.grid_for_bbox(bbox, g.get("crs", "auto"), g.get("origin", (0.0, 0.0)))
    if grid.area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"This area is {grid.area_km2:.0f} km²; the change-hazard model runs on at most {MAX_AREA_KM2:.0f} km². "
            "Draw a smaller region."
        )
    return bbox, grid


def _check_t0(bundle: dict[str, Any], t0: int | None) -> None:
    t0_min = int(bundle.get("t0_min", features.T0_FIRST))
    if t0 is not None and t0 < t0_min:
        raise ValueError(f"The change-hazard model runs from t0 >= {t0_min}, not {t0}.")


def _cube(ee: Any, bundle: dict[str, Any], grid: features.Grid, t0: int | None) -> features.FeatureCube:
    _check_t0(bundle, t0)
    t0_min = int(bundle.get("t0_min", features.T0_FIRST))
    need_ae = any(c in features.AE_COLUMNS for c in bundle["columns"])
    source = source_of(bundle)
    latest = t0 is None
    if latest:
        t0 = features.latest_t0(ee, grid, first=t0_min, need_ae=need_ae, source=source)
    return features.build_features(
        grid, t0, bundle["columns"], ee=ee, check_years=not latest, dem=bundle.get("dem", LEGACY_DEM), source=source
    )


# ----------------------------------------------------------------- hazards


def run(RasterPath: str, model_dir: str, t0: int | None = None) -> dict[str, Any]:  # noqa: N803
    """Hazard ranks for the footprint of ``RasterPath`` from ``t0`` (None: the latest year with data)."""
    bundle = load_bundle(model_dir)
    _check_t0(bundle, t0)
    bbox, grid = _area(RasterPath, bundle)
    cube = _cube(_ee(), bundle, grid, t0)
    return write_result(bundle, cube, bbox)


def level_colours(hue: str) -> dict[int, str]:
    """Raster value -> colour of each risk level: the hazard's hue, lighter for lower levels."""
    rgb = np.array([int(hue[i : i + 2], 16) for i in (1, 3, 5)], float)
    return {v: "#" + "".join(f"{int(round(c)):02x}" for c in t * rgb + (1 - t) * 255) for v, t in LEVEL_TINT.items()}


def risk_levels(rank: np.ndarray) -> np.ndarray:
    """Risk level per cell (3 high, 2 elevated, 1 moderate; NaN lower or unranked) of a national rank."""
    out = np.full(rank.shape, np.nan, np.float32)
    for value, _, low, _ in sorted(RISK_LEVELS):
        out[np.isfinite(rank) & (rank >= low)] = value
    return out


def _skill(bundle: dict[str, Any], h: int, group: str) -> dict[str, Any] | None:
    s = bundle.get("skill", {}).get(str(h), {})
    return s.get("hazards", {}).get(group) if isinstance(s.get("hazards"), dict) else None


def write_result(bundle: dict[str, Any], cube: features.FeatureCube, bbox: list[float]) -> dict[str, Any]:
    """Predict hazards on ``cube`` and save their rasters as job artifacts; returns the step result."""
    bundle = as_format_2(bundle)
    source = source_of(bundle)
    names = source.classes
    grid = cube.grid
    shape = (grid.ny, grid.nx)
    start = cube.start_class.ravel()
    x = cube.matrix(bundle["columns"])
    probs: dict[int, np.ndarray] = {}
    rasters: dict[str, np.ndarray] = {}
    style = artifact_files.index_style(RANK_RAMP, 0, 100)
    styles: dict[str, Any] = {}
    hazards: list[dict[str, Any]] = []
    order = [ANY] + [g for g in bundle["groups"] if g != ANY]
    for g in order:
        h = int(bundle["served"]["hazards"].get(g, 0) or 0)
        if g not in bundle["groups"] or h == 0:
            continue
        hz = bundle["horizons"][str(h)]
        if h not in probs:
            probs[h] = probabilities(bundle, hz["from_classes"], x, start, key=str(h))
        score = group_score(probs[h], start, bundle["groups"][g], names)
        key = INDEX_TYPE if g == ANY else f"rank_{g}"
        national = percentile(score, hz["reference_quantiles"][g]).reshape(shape)
        local = local_percentile(score).reshape(shape)
        level = risk_levels(national)
        colour = GROUP_COLOURS.get(g, OTHER_COLOUR)
        shades = level_colours(colour)
        rasters[key], rasters[f"{key}_local"], rasters[f"{key}_level"] = national, local, level
        styles[key] = styles[f"{key}_local"] = style
        styles[f"{key}_level"] = artifact_files.class_style(shades)
        ok = np.isfinite(national)
        n_ok = max(int(ok.sum()), 1)
        hazards.append(
            {
                "group": g,
                "title": bundle["groups"][g].get("title") or GROUP_TITLES.get(g, g),
                "colour": colour,
                "raster": key,
                "level_raster": f"{key}_level",
                "levels": [
                    {
                        "level": name,
                        "range": wording,
                        "colour": shades[value],
                        "km2": stats.km2((level == value).sum()),
                        "percent": round(100.0 * float((level == value).sum()) / n_ok, 1),
                    }
                    for value, name, _, wording in RISK_LEVELS
                ],
                "horizon_years": h,
                "year": cube.t0 + h,
                "eligible_km2": stats.km2(ok.sum()),
                "top_decile_km2": stats.km2((national[ok] >= 90).sum()),
                "top_decile_percent": round(100.0 * float((national[ok] >= 90).mean()), 1) if ok.any() else None,
                "mean_rank": round(float(np.nanmean(national)), 1) if ok.any() else None,
                "skill": _skill(bundle, h, g),
            }
        )
    if not hazards:
        raise BundleError("The bundle serves no hazard.")
    rasters["start_class"] = np.where(cube.start_class > 0, cube.start_class, np.nan).astype(np.float32)
    styles["start_class"] = artifact_files.class_style({i + 1: source.colours[n] for i, n in enumerate(names)})

    main = rasters[hazards[0]["raster"]]
    ranked = np.isfinite(main)
    valid = cube.start_class > 0
    shares = {
        n: round(100.0 * float((cube.start_class == i + 1).sum()) / max(int(valid.sum()), 1), 2)
        for i, n in enumerate(names)
    }
    statistics = {
        "mean": float(np.nanmean(main)) if ranked.any() else None,
        "min": float(np.nanmin(main)) if ranked.any() else None,
        "max": float(np.nanmax(main)) if ranked.any() else None,
        "valid_pixels": int(ranked.sum()),
        "cells": int(main.size),
        "cells_without_model": int((valid & ~ranked).sum()),
        "top_decile_percent": round(100.0 * float((main[ranked] >= 90).mean()), 2) if ranked.any() else None,
        "start_class_percent": shares,
        "start_class_km2": {n: stats.km2((cube.start_class == i + 1).sum()) for i, n in enumerate(names)},
        "hazards": hazards,
        "summary": [
            f"{h['title']} by {h['year']}: {h['top_decile_percent']:.0f} % of the {stats.km2_text(h['eligible_km2'])} "
            f"that could make this change ranks in the national top 10 % ({stats.km2_text(h['top_decile_km2'])})."
            for h in hazards
            if h["top_decile_percent"] is not None
        ],
    }
    h0 = hazards[0]["horizon_years"]
    result: dict[str, Any] = {
        "index_type": hazards[0]["raster"],
        "statistics": statistics,
        "metadata": {
            "model_id": bundle["model_id"],
            "version": bundle.get("version"),
            "t0": cube.t0,
            "source": source.title,
            "legend": list(names),
            "interval_years": h0,
            "forecast_to": cube.t0 + h0,
            "dem": bundle.get("dem", LEGACY_DEM),
            "values": "percentile of the hazard's probability among the training cells that could make it; "
            "<key>_local: among this area's cells; ranks only",
            "domain": bundle["domain"].get("name"),
            "bbox_wgs84": [round(v, 6) for v in bbox],
            "grid": {"crs": grid.crs, "resolution_m": features.RES, "width": grid.nx, "height": grid.ny},
            "feature_seconds": cube.seconds,
            "skill": bundle.get("skill", {}).get(str(h0)),
        },
    }
    bare = {f"{key}_map": key for key in rasters}
    return artifact_files.save(result, rasters, {}, crs=grid.crs, transform=grid.transform(), styles=styles, bare=bare)


# ----------------------------------------------------------------- forecast map


def trend_grid(grid: features.Grid, cells: int = TREND_CELLS) -> features.Grid:
    """A square of at least ``cells`` core cells per side, centred on ``grid`` and on its lattice."""
    nx, ny = max(cells, grid.nx), max(cells, grid.ny)
    x0 = grid.x0 - features.RES * ((nx - grid.nx) // 2)
    y0 = grid.y0 + features.RES * ((ny - grid.ny) // 2)
    return features.Grid(grid.crs, x0, y0, nx, ny, grid.halo)


def run_forecast(RasterPath: str, model_dir: str, horizon: int | None = None) -> dict[str, Any]:  # noqa: N803
    """The land-cover map ``horizon`` years after the latest year with data, for the footprint of ``RasterPath``."""
    bundle = load_bundle(model_dir)
    longest = int(bundle["served"].get("map") or 0)
    if longest == 0:
        raise BundleError("This model's forecast maps did not pass their check, so it serves hazards only.")
    h = longest if horizon is None else int(horizon)
    if not 1 <= h <= longest:
        raise ValueError(f"Forecast maps were checked up to {longest} years ahead; {h} is not offered.")
    bbox, grid = _area(RasterPath, bundle)
    ee = _ee()
    cube = _cube(ee, bundle, grid, None)
    tg = trend_grid(grid)
    last = min(cube.t0, int(bundle.get("trend_last", cube.t0)))
    pair = (last - h, last)
    source = source_of(bundle)
    trend = [features.build_features(tg, y, [], ee=ee, check_years=False, source=source).start_class for y in pair]
    return write_forecast(bundle, cube, h, trend, pair, tg, bbox)


def write_forecast(
    bundle: dict[str, Any],
    cube: features.FeatureCube,
    h: int,
    trend: list[np.ndarray],
    pair: tuple[int, int],
    trend_area: features.Grid,
    bbox: list[float],
) -> dict[str, Any]:
    """Allocate the trend's amounts by the model's probabilities and save the forecast map and its statistics."""
    source = source_of(bundle)
    names, k = source.classes, len(source.classes)
    grid = cube.grid
    shape = (grid.ny, grid.nx)
    start = cube.start_class.ravel()
    p = probabilities(
        bundle, bundle["horizons"][str(h)]["from_classes"], cube.matrix(bundle["columns"]), start, str(h)
    )
    trend_m = stats.matrix(trend[0], trend[1], k)
    quota = stats.demand(start, stats.rates(trend_m))
    end = stats.allocate(start, quota, p, seed=cube.t0 * 100 + h).reshape(shape)
    m = stats.matrix(cube.start_class, end, k)
    year = cube.t0 + h
    expected = bool(bundle["served"].get("amounts_expected", {}).get(str(h)))
    longest = int(bundle["served"].get("map") or 0)
    if expected:
        lead, verb = f"Expected by {year}:", "is expected to {dir}"
        checked = f"Placement and amounts checked in a backtest up to {longest} years ahead."
    else:
        lead, verb = f"If the {pair[0]}–{pair[1]} trend continues, by {year}:", "would {dir}"
        checked = (
            f"Where change lands was checked in a backtest up to {longest} years ahead; how much changes is a "
            f"scenario: the {pair[0]}–{pair[1]} trend did worse than assuming no change in that backtest."
        )
    style = artifact_files.class_style({i + 1: source.colours[n] for i, n in enumerate(names)})
    rasters = {
        "class_start": np.where(cube.start_class > 0, cube.start_class, np.nan).astype(np.float32),
        "class_forecast": np.where(end > 0, end, np.nan).astype(np.float32),
        "changed": np.where((end > 0) & (end != cube.start_class), end, np.nan).astype(np.float32),
    }
    statistics = {
        "t0": cube.t0,
        "year": year,
        "horizon_years": h,
        "classes": stats.class_table(m.sum(1), m.sum(0), names),
        "matrix": stats.matrix_table(m, names),
        "changed_percent": round(100.0 * float(m.sum() - np.trace(m)) / max(int(m.sum()), 1), 1),
        "summary_lead": lead,
        "summary": stats.sentences(m, verb, names),
        "transitions": stats.top_transitions(m, names=names),
        "changed_km2": stats.changed_km2(m),
        "checked": checked,
        "amounts": "expected" if expected else "trend scenario",
        "trend": {
            "pair": list(pair),
            "area_km2": round(trend_area.area_km2, 1),
            "matrix": stats.matrix_table(trend_m, names),
        },
    }
    result: dict[str, Any] = {
        "index_type": "land_cover_forecast",
        "statistics": statistics,
        "metadata": {
            "model_id": bundle["model_id"],
            "version": bundle.get("version"),
            "t0": cube.t0,
            "forecast_to": year,
            "horizon_years": h,
            "source": source.title,
            "legend": list(names),
            "dem": bundle.get("dem", LEGACY_DEM),
            "method": "amounts: Markov trend of the latest pair of the same length in a 12 km square around the area; "
            "placement: the model's probabilities, highest first",
            "domain": bundle["domain"].get("name"),
            "bbox_wgs84": [round(v, 6) for v in bbox],
            "grid": {"crs": grid.crs, "resolution_m": features.RES, "width": grid.nx, "height": grid.ny},
            "feature_seconds": cube.seconds,
            "skill": bundle.get("skill", {}).get(str(h)),
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
