"""The trained bundle of the change-hazard model: loading, checks, and its per-cell probabilities and ranks.

The bundle (``<model dir>/lulc_hazard.json``, one file, so the model manager downloads it as one)::

    {"format": "dt4lc-lulc-hazard/2", "model_id": str, "version": str, "target": "esri" | "kpi",
     "legend": [the target's class names], "columns": [feature names, in the boosters' order],
     "t0_min": int,                           # optional, default 2017: earliest origin year it accepts
     "dem": "glo30_2024_1" | "glo30",         # the Copernicus DEM release the features were built from
     "grid": {"crs": "EPSG:32636" | "auto", "origin": [x, y]},
     "domain": {"bbox": [west, south, east, north], "name": str, "short": str},
     "trend_last": int,                       # optional: the forecast's trend pair ends by this year
     "groups": {"<group>": {"title": str, "from": [classes], "to": [classes] | null}},   # null: any other class
     "horizons": {"<h>": {"pair": [y0, y1],
                          "from_classes": {"<class>": {"to": [class names], "prior_correction": [float per "to"],
                                                       "booster": <XGBoost JSON model> | null}},
                          "reference_quantiles": {"<group>": [sorted floats]}}},
     "served": {"hazards": {"<group>": h}, "map": h, "amounts_expected": {"<h>": bool}},   # 0: not served
     "skill": {"<h>": {...}}}                 # optional, copied into the result

Per from-class, as in training: the booster's class probabilities are multiplied by ``prior_correction`` and
renormalised; a from-class with one target class has no booster (it always stays); a from-class absent from the
bundle gets no rank (NaN). The reference quantiles are the probabilities' quantiles at evenly spaced levels over the
training cells, written when the bundle is exported.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np

from . import features

BUNDLE_FILE = "lulc_hazard.json"
FORMAT = "dt4lc-lulc-hazard/2"


class BundleError(ValueError):
    """The model directory holds no usable bundle."""


# ----------------------------------------------------------------- bundle


def load_bundle(model_dir: str | Path) -> dict[str, Any]:
    """The checked bundle of ``model_dir`` (cached per file and mtime)."""
    path = Path(model_dir) / BUNDLE_FILE
    if not path.is_file():
        raise BundleError(f"No {BUNDLE_FILE} in {model_dir}: the change-hazard model is not installed.")
    return _load_bundle(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load_bundle(path: str, mtime_ns: int) -> dict[str, Any]:
    bundle: dict[str, Any] = json.loads(Path(path).read_text())
    check_bundle(bundle)
    return bundle


def source_of(bundle: dict[str, Any]) -> features.LabelSource:
    """The label source of the bundle's ``target``."""
    source = features.SOURCES.get(bundle.get("target", ""))
    if source is None:
        raise BundleError(f"Bundle target must be one of {', '.join(features.SOURCES)}, not {bundle.get('target')!r}.")
    return source


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
    """BundleError unless ``bundle`` has the shape in the module docstring."""
    if bundle.get("format") != FORMAT:
        raise BundleError(f"Bundle format {bundle.get('format')!r}; this runner reads {FORMAT!r}.")
    missing = [
        k for k in ("model_id", "columns", "dem", "grid", "domain", "groups", "horizons", "served") if k not in bundle
    ]
    if missing:
        raise BundleError(f"Bundle lacks {', '.join(missing)}.")
    source = source_of(bundle)
    names = source.classes
    if list(bundle.get("legend", [])) != list(names):
        raise BundleError(f"Bundle legend must be the {source.key} legend " + ", ".join(names) + ".")
    try:
        features.check_columns(bundle["columns"], source)
        features.check_dem(bundle["dem"])
    except ValueError as e:
        raise BundleError(str(e)) from e
    for g, spec in bundle["groups"].items():
        classes = list(spec.get("from", [])) + list(spec.get("to") or [])
        if not spec.get("from") or any(c not in names for c in classes):
            raise BundleError(f"groups[{g!r}] needs 'from' classes of the bundle's legend.")
    served = bundle["served"]
    for h, hz in bundle["horizons"].items():
        _check_from_classes(hz["from_classes"], names, f"horizons[{h}].")
        for g, gh in served.get("hazards", {}).items():
            if gh and int(h) == int(gh):  # a group is ranked at its served horizon only
                if g not in bundle["groups"]:
                    raise BundleError(f"served hazard {g!r} is not a group of the bundle.")
                _check_quantiles(hz["reference_quantiles"].get(g), f"{h}:{g}")
    needed = {int(v) for v in served.get("hazards", {}).values() if v} | (
        {int(served["map"])} if served.get("map") else set()
    )
    absent = sorted(h for h in needed if str(h) not in bundle["horizons"])
    if absent:
        raise BundleError(f"Served horizons {absent} have no models in the bundle.")


# id of a bundle's booster JSON -> (that JSON, its parsed booster). Bundles are cached per file, so the same JSON
# object comes back on every run; holding it keeps its id from being reused.
_BOOSTERS: OrderedDict[int, tuple[Any, Any]] = OrderedDict()
_BOOSTERS_MAX = 32
_boosters_lock = threading.Lock()


def _booster(spec: Any) -> Any:
    """The parsed XGBoost booster of the booster JSON ``spec``, parsed once per JSON object."""
    with _boosters_lock:
        hit = _BOOSTERS.get(id(spec))
        if hit is not None and hit[0] is spec:
            _BOOSTERS.move_to_end(id(spec))
            return hit[1]
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(spec), "utf-8"))
    with _boosters_lock:
        _BOOSTERS[id(spec)] = (spec, booster)
        while len(_BOOSTERS) > _BOOSTERS_MAX:
            _BOOSTERS.popitem(last=False)
    return booster


# ----------------------------------------------------------------- prediction


def probabilities(
    bundle: dict[str, Any], from_classes: dict[str, Any], x: np.ndarray, start: np.ndarray
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

            booster = _booster(spec["booster"])
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
