"""Runner of ``models/lulc-change-hazard``: where land cover is most likely to change, for the area of a raster.

The step takes the input raster only for its footprint (an upload, or the Earth Engine composite of a region drawn
on the map; its dates play no part), builds the model's features for that footprint from Earth Engine
(:mod:`.features`) at the origin year ``t0`` = the latest year with both an Esri map and an AlphaEarth embedding of
the area, and applies a trained bundle: the ranks are for change between ``t0`` and ``t0 + interval_years``. It
writes, on a 30 m grid:

- ``change_rank``: rank of the probability that the cell's Esri class changes within the model's training interval,
  as a percentile (0-100) of the same probability over the cells of the model's training windows;
- ``rank_<from>_to_<to>``: the same for each transition the bundle lists (NaN outside the from-class);
- ``start_class``: the Esri class at ``t0`` (1-8, :data:`.features.ESRI_CLASSES`).

The values are ranks, not probabilities: the probability level of this model does not carry over to a new period,
only its ordering of cells does. A percentile of 90 means "more likely to change than 90 % of the training cells".

Bundle (``<model dir>/lulc_hazard.json``, one file so the model manager downloads it as one)::

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

from . import features

logger = logging.getLogger(__name__)

BUNDLE_FILE = "lulc_hazard.json"
FORMAT = "dt4lc-lulc-hazard/1"
LEGACY_DEM = "glo30"  # the DEM of a bundle that names none (dataset v1.0)
MAX_AREA_KM2 = 400.0  # about 20 x 20 km: 64 embedding bands of 450k cells stay near 120 MB in memory
INDEX_TYPE = "change_rank"
RANK_RAMP = ("#f2efe6", "#e9c46a", "#d1773a", "#a3322a", "#5c1a1b")  # low -> high rank
ESRI_COLOURS = {  # Esri/IO legend colours
    "Water": "#419bdf",
    "Trees": "#397d49",
    "FloodedVeg": "#7a87c6",
    "Crops": "#e49635",
    "Built": "#c4281b",
    "Bare": "#a59b8f",
    "Snow": "#a8ebff",
    "Rangeland": "#e3e2c3",
}


class BundleError(ValueError):
    """The model directory holds no usable bundle."""


class OutsideDomainError(ValueError):
    """The area lies outside the region the model was trained on."""


# ----------------------------------------------------------------- bundle


def load_bundle(model_dir: str | Path) -> dict[str, Any]:
    """The checked bundle of ``model_dir`` (cached per file and modification time)."""
    path = Path(model_dir) / BUNDLE_FILE
    if not path.is_file():
        raise BundleError(f"No {BUNDLE_FILE} in {model_dir}: the change-hazard model is not installed.")
    return _load_bundle(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load_bundle(path: str, mtime_ns: int) -> dict[str, Any]:
    bundle = json.loads(Path(path).read_text())
    check_bundle(bundle)
    return bundle


def check_bundle(bundle: dict[str, Any]) -> None:
    """BundleError unless ``bundle`` has the shape in the module docstring."""
    if bundle.get("format") != FORMAT:
        raise BundleError(f"Bundle format {bundle.get('format')!r}; this runner reads {FORMAT!r}.")
    missing = [
        k
        for k in (
            "model_id",
            "columns",
            "interval_years",
            "grid",
            "domain",
            "from_classes",
            "transitions",
            "reference_quantiles",
        )
        if k not in bundle
    ]
    if missing:
        raise BundleError(f"Bundle lacks {', '.join(missing)}.")
    if bundle.get("target") != "esri" or list(bundle.get("legend", [])) != list(features.ESRI_CLASSES):
        raise BundleError("Bundle target must be the Esri own legend " + ", ".join(features.ESRI_CLASSES) + ".")
    try:
        features.check_columns(bundle["columns"])
        features.check_dem(bundle.get("dem", LEGACY_DEM))
    except ValueError as e:
        raise BundleError(str(e)) from e
    for name, spec in bundle["from_classes"].items():
        if name not in features.ESRI_CLASSES or any(t not in features.ESRI_CLASSES for t in spec.get("to", [])):
            raise BundleError(f"from_classes[{name!r}] names a class outside the Esri legend.")
        if len(spec.get("prior_correction", [])) != len(spec["to"]):
            raise BundleError(f"from_classes[{name!r}]: prior_correction needs one value per 'to' class.")
        if spec.get("booster") is None and len(spec["to"]) != 1:
            raise BundleError(f"from_classes[{name!r}] has {len(spec['to'])} target classes and no booster.")
    keys = ["any_change"] + [f"{a}->{b}" for a, b in bundle["transitions"]]
    for k in keys:
        q = bundle["reference_quantiles"].get(k)
        if not q or len(q) < 11 or np.any(np.diff(q) < 0):
            raise BundleError(f"reference_quantiles[{k!r}] must be at least 11 sorted values.")
    for a, b in bundle["transitions"]:
        if a not in bundle["from_classes"] or b not in bundle["from_classes"][a]["to"]:
            raise BundleError(f"Transition {a}->{b} is not one the bundle's models predict.")


@lru_cache(maxsize=16)
def _booster(model_id: str, version: str, from_class: str, text: str) -> Any:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(text, "utf-8"))
    return booster


# ----------------------------------------------------------------- prediction


def predict(bundle: dict[str, Any], x: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per cell: ``P(change)`` and ``P(from -> to)`` for each bundle transition; NaN where no model applies.

    ``x`` is ``(cells, columns)`` in the bundle's column order, ``start`` the cells' class index (0 = no data).
    """
    p_change = np.full(start.shape, np.nan, np.float32)
    p_trans = {f"{a}->{b}": np.full(start.shape, np.nan, np.float32) for a, b in bundle["transitions"]}
    for name, spec in bundle["from_classes"].items():
        idx = np.flatnonzero(start == features.ESRI_CLASSES.index(name) + 1)
        if idx.size == 0:
            continue
        to = list(spec["to"])
        if spec.get("booster") is None:
            p = np.ones((idx.size, 1))
        else:
            import xgboost as xgb

            booster = _booster(bundle["model_id"], str(bundle.get("version", "")), name, json.dumps(spec["booster"]))
            raw = np.asarray(booster.predict(xgb.DMatrix(x[idx], missing=np.nan)), dtype=np.float64)
            p = np.stack([1.0 - raw, raw], axis=1) if raw.ndim == 1 else raw
            if p.shape[1] != len(to):
                raise BundleError(f"The {name} booster gives {p.shape[1]} classes; the bundle lists {len(to)}.")
        p = p * np.asarray(spec["prior_correction"], dtype=np.float64)[None, :]
        s = p.sum(1, keepdims=True)
        p = np.where(s > 0, p / np.where(s > 0, s, 1.0), 0.0)
        stay = p[:, to.index(name)] if name in to else np.zeros(idx.size)
        stay = np.where(s[:, 0] > 0, stay, 1.0)  # a zero row means "stays", as in training
        p_change[idx] = (1.0 - stay).astype(np.float32)
        for b in to:
            key = f"{name}->{b}"
            if key in p_trans:
                p_trans[key][idx] = p[:, to.index(b)].astype(np.float32)
    return p_change, p_trans


def percentile(p: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Percentile (0-100) of each ``p`` among reference quantiles at evenly spaced levels (mid-rank on ties)."""
    q = np.asarray(quantiles, dtype=np.float64)
    out = np.full(p.shape, np.nan, np.float32)
    ok = np.isfinite(p)
    lo = np.searchsorted(q, p[ok], side="left")
    hi = np.searchsorted(q, p[ok], side="right")
    out[ok] = (100.0 * (lo + hi) / (2.0 * len(q))).astype(np.float32)
    return out


# ----------------------------------------------------------------- runner


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


def run(RasterPath: str, model_dir: str, t0: int | None = None) -> dict[str, Any]:  # noqa: N803
    """Change-probability ranks for the footprint of ``RasterPath`` from ``t0`` (None: the latest year with data)."""
    bundle = load_bundle(model_dir)
    t0_min = int(bundle.get("t0_min", features.T0_FIRST))
    if t0 is not None and t0 < t0_min:
        raise ValueError(f"The change-hazard model runs from t0 >= {t0_min}, not {t0}.")
    bbox = footprint_wgs84(RasterPath)
    check_domain(bbox, bundle["domain"])
    g = bundle["grid"]
    grid = features.grid_for_bbox(bbox, g.get("crs", "auto"), g.get("origin", (0.0, 0.0)))
    if grid.area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"This area is {grid.area_km2:.0f} km²; the change-hazard model runs on at most {MAX_AREA_KM2:.0f} km². "
            "Draw a smaller region."
        )

    ee = _ee()
    need_ae = any(c in features.AE_COLUMNS for c in bundle["columns"])
    latest = t0 is None
    if latest:
        t0 = features.latest_t0(ee, grid, first=t0_min, need_ae=need_ae)
    cube = features.build_features(
        grid, t0, bundle["columns"], ee=ee, check_years=not latest, dem=bundle.get("dem", LEGACY_DEM)
    )
    return write_result(bundle, cube, bbox)


def write_result(bundle: dict[str, Any], cube: features.FeatureCube, bbox: list[float]) -> dict[str, Any]:
    """Predict on ``cube`` and save its rasters as job artifacts; returns the step result."""
    grid = cube.grid
    shape = (grid.ny, grid.nx)
    start = cube.start_class.ravel()
    p_change, p_trans = predict(bundle, cube.matrix(bundle["columns"]), start)
    refs = bundle["reference_quantiles"]

    rank = percentile(p_change, refs["any_change"]).reshape(shape)
    rasters: dict[str, np.ndarray] = {INDEX_TYPE: rank}
    styles = {INDEX_TYPE: artifact_files.index_style(RANK_RAMP, 0, 100)}
    for a, b in bundle["transitions"]:
        key = f"rank_{a.lower()}_to_{b.lower()}"
        rasters[key] = percentile(p_trans[f"{a}->{b}"], refs[f"{a}->{b}"]).reshape(shape)
        styles[key] = styles[INDEX_TYPE]
    rasters["start_class"] = np.where(cube.start_class > 0, cube.start_class, np.nan).astype(np.float32)
    styles["start_class"] = artifact_files.class_style(
        {i + 1: ESRI_COLOURS[n] for i, n in enumerate(features.ESRI_CLASSES)}
    )

    valid = cube.start_class > 0
    ranked = np.isfinite(rank)
    shares = {
        n: round(100.0 * float((cube.start_class == i + 1).sum()) / max(int(valid.sum()), 1), 2)
        for i, n in enumerate(features.ESRI_CLASSES)
    }
    statistics = {
        "mean": float(np.nanmean(rank)) if ranked.any() else None,
        "min": float(np.nanmin(rank)) if ranked.any() else None,
        "max": float(np.nanmax(rank)) if ranked.any() else None,
        "valid_pixels": int(ranked.sum()),
        "cells": int(rank.size),
        "cells_without_model": int((valid & ~ranked).sum()),
        "top_decile_percent": round(100.0 * float((rank[ranked] >= 90).mean()), 2) if ranked.any() else None,
        "start_class_percent": shares,
    }
    result: dict[str, Any] = {
        "index_type": INDEX_TYPE,
        "statistics": statistics,
        "metadata": {
            "model_id": bundle["model_id"],
            "version": bundle.get("version"),
            "t0": cube.t0,
            "interval_years": bundle["interval_years"],
            "forecast_to": cube.t0 + bundle["interval_years"],
            "dem": bundle.get("dem", LEGACY_DEM),
            "values": "percentile of P(change within interval_years) among the training cells; a rank only",
            "domain": bundle["domain"].get("name"),
            "bbox_wgs84": [round(v, 6) for v in bbox],
            "grid": {"crs": grid.crs, "resolution_m": features.RES, "width": grid.nx, "height": grid.ny},
            "feature_seconds": cube.seconds,
            "skill": bundle.get("skill"),
        },
    }
    bare = {f"{key}_map": key for key in rasters}
    return artifact_files.save(result, rasters, {}, crs=grid.crs, transform=grid.transform(), styles=styles, bare=bare)
