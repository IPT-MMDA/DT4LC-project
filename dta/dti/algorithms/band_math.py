"""Band math: an index from a user formula over band codes or roles.

``config.formula`` is an arithmetic expression such as ``(B5 - B6) / (B5 + B6)``.
It is parsed with :mod:`ast` into a tree that may hold only numbers, band names,
binary ``+ - * /``, unary ``-`` and parentheses, and the tree is evaluated on
NumPy arrays node by node. The formula is never passed to ``eval``.

Band names:

* Codes (``B5``, ``SR_B5``, ``B8A``, any case) name bands by their description,
  parsed like :mod:`dta.dti.raster`. Only if no band has a description does
  ``Bn`` mean band ``n`` (logged).
* Roles (``nir``, ``swir``, ``red``, ``green``, ``blue``) resolve through
  :func:`dta.dti.raster.band_indices`.

Two registry ids share this module: ``algorithms/band-math`` (``RasterPath``)
and ``algorithms/band-math-change`` (``RasterPathBefore`` + ``RasterPathAfter``).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from dta.dti import artifact_files
from dta.dti.algorithms.change_detection import INCREASE_CLASS_NAMES, change_class_statistics
from dta.dti.algorithms.spectral_index import COLORMAPS, HAS_MATPLOTLIB, _render_index_png
from dta.dti.raster import _BAND_CODE, _band_summary, band_indices, pixel_area_km2, reflectance_divisor

logger = logging.getLogger(__name__)

MAX_FORMULA_LENGTH = 256
ROLES = ("red", "nir", "green", "blue", "swir")
# |after - before| above this counts as an increase / decrease in change mode.
DEFAULT_THRESHOLD = 0.2
COLORMAP = "ndwi"  # brown (low) -> white -> blue (high)
# ``class_statistics`` names (increase, decrease, unchanged) by the step's ``name``; other names use
# ``Increase`` / ``Decrease`` / ``Stable``.
CHANGE_CLASS_NAMES = {"NDMI": ("Wetter", "Drier", "No change")}

_ALLOWED = "numbers, band names (B5, SR_B5, nir, swir, red, green, blue), + - * / and parentheses"
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}


# --- Formula parsing -----------------------------------------------------------


def _canonical(name: str) -> str | None:
    """``"b5"`` / ``"SR_B5"`` -> ``"B5"``, ``"NIR"`` -> ``"nir"``; ``None`` if not a band name."""
    low = name.lower()
    if low in ROLES:
        return low
    m = _BAND_CODE.match(low)
    return f"B{m.group(1).upper()}" if m else None


def _check(node: ast.expr, names: list[str]) -> None:
    """Raise ``ValueError`` unless ``node`` is allowed band math; append its band names to ``names``."""
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        _check(node.left, names)
        _check(node.right, names)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        _check(node.operand, names)
    elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
        pass
    elif isinstance(node, ast.Name):
        canonical = _canonical(node.id)
        if canonical is None:
            raise ValueError(
                f"Unknown band name {node.id!r} in the formula. "
                "Use band codes (B5, SR_B5) or roles (nir, swir, red, green, blue)."
            )
        if canonical not in names:
            names.append(canonical)
    else:
        raise ValueError(f"Formula not allowed: `{ast.unparse(node)}`. Band math accepts only {_ALLOWED}.")


def _parse(formula: Any) -> tuple[ast.expr, list[str]]:
    """Parse and check ``formula``; return the expression tree and its band names in order of use."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Band math needs a formula, e.g. (B5 - B6) / (B5 + B6).")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise ValueError(f"Formula is too long ({len(formula)} characters; the limit is {MAX_FORMULA_LENGTH}).")
    try:
        tree = ast.parse(formula.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Formula is not a valid expression: {formula.strip()!r}.") from exc
    names: list[str] = []
    _check(tree, names)
    if not names:
        raise ValueError(f"Formula {formula.strip()!r} uses no band.")
    return tree, names


def _missing_band_message(name: str, available: list[str]) -> str:
    codes: list[str | None] = [b for b in available if b not in ROLES]
    summary = _band_summary(codes) if codes else "no band codes"
    return f"Band {name} is not in this file ({summary}). Available: {', '.join(available)}."


def validate_formula(formula: str, available_bands: Iterable[str] | None = None) -> list[str]:
    """Check that ``formula`` is safe band math, without reading any raster.

    Args:
        formula: Expression over band names, e.g. ``"(B5 - B6) / (B5 + B6)"``.
        available_bands: Names the raster offers (as from :func:`band_names`).
            If given, every band the formula uses must be among them.

    Returns:
        The band names the formula uses, canonical (``"B5"``, ``"nir"``), in order of first use.

    Raises:
        ValueError: If the formula is empty, too long, not an expression, uses
            anything but numbers, band names, ``+ - * /`` and parentheses, or
            names a band that is not available.
    """
    _, names = _parse(formula)
    if available_bands is not None:
        available = [_canonical(b) or b for b in available_bands]
        for name in names:
            if name not in available:
                raise ValueError(_missing_band_message(name, available))
    return names


def _is_normalized_difference(node: ast.expr) -> bool:
    """True for ``(a - b) / (a + b)`` or ``(a - b) / (b + a)`` with ``a`` != ``b``."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return False
    num, den = node.left, node.right
    if not (isinstance(num, ast.BinOp) and isinstance(num.op, ast.Sub)):
        return False
    if not (isinstance(den, ast.BinOp) and isinstance(den.op, ast.Add)):
        return False
    a, b = ast.dump(num.left), ast.dump(num.right)
    return a != b and {a, b} == {ast.dump(den.left), ast.dump(den.right)}


def _evaluate(node: ast.expr, bands: dict[str, np.ndarray]) -> Any:
    """Evaluate a tree accepted by :func:`_check` on the band arrays."""
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_evaluate(node.left, bands), _evaluate(node.right, bands))
    if isinstance(node, ast.UnaryOp):
        return np.negative(_evaluate(node.operand, bands))
    if isinstance(node, ast.Constant):
        return float(node.value)
    assert isinstance(node, ast.Name)
    return bands[str(_canonical(node.id))]


# --- Rasters ---------------------------------------------------------------------


def band_names(src: Any) -> dict[str, int]:
    """Names a formula can use on an open raster -> 1-based band index.

    Codes (``"B5"``) come from the band descriptions; if no band has a
    description, ``"Bn"`` is band ``n`` (a warning is logged). Roles
    (``"nir"``, ...) come from :func:`dta.dti.raster.band_indices`.
    """
    descriptions = [(d or "").strip() for d in (src.descriptions or ())]
    names: dict[str, int] = {}
    if any(descriptions):
        for i, desc in enumerate(descriptions, start=1):
            if m := _BAND_CODE.match(desc.lower()):
                names.setdefault(f"B{m.group(1).upper()}", i)
    else:
        logger.warning("%s has no band descriptions; band math reads Bn as band n", getattr(src, "name", "raster"))
        names = {f"B{i}": i for i in range(1, src.count + 1)}
    for role, idx in band_indices(src).items():
        if idx is not None:
            names[role] = idx
    return names


def _stats(arr: np.ndarray) -> dict[str, float | int]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"valid_pixels": 0, "total_pixels": int(arr.size)}
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(arr.size),
    }


def _percentile_range(arr: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        hi = float(np.percentile(np.abs(valid), 98)) or 1.0
        return (-hi, hi)
    lo, hi = np.percentile(valid, [2, 98])
    return (float(lo), float(hi))


def _compute(raster_path: str, tree: ast.expr, formula: str) -> tuple[np.ndarray, dict[str, Any], tuple[Any, Any]]:
    """Evaluate ``tree`` on ``raster_path``; return ``(index, metadata, (crs, transform))``.

    Bands are scaled to 0-1 reflectance by the source dtype; a pixel that is
    nodata or non-finite in any band used, or where the result is not finite
    (e.g. a zero denominator), is NaN. A normalized difference is clipped to [-1, 1].
    """
    if not Path(raster_path).exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    with rasterio.open(raster_path) as src:
        available = band_names(src)
        names = validate_formula(formula, available)
        bands: dict[str, np.ndarray] = {}
        dtypes: list[str] = []
        for name in names:
            idx = available[name]
            dtypes.append(src.dtypes[idx - 1])
            bands[name] = np.ma.filled(src.read(idx, masked=True).astype(np.float64), np.nan)

        input_valid = np.logical_and.reduce([np.isfinite(b) for b in bands.values()])
        divisor = reflectance_divisor(np.result_type(*dtypes), np.stack([b[input_valid] for b in bands.values()]))
        if divisor != 1.0:
            bands = {name: b / divisor for name, b in bands.items()}

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            index = np.asarray(_evaluate(tree, bands), dtype=np.float64)
        index = np.array(np.broadcast_to(index, input_valid.shape))
        index[~input_valid | ~np.isfinite(index)] = np.nan
        if _is_normalized_difference(tree):
            index = np.clip(index, -1.0, 1.0)

        metadata = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform) if src.transform else None,
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "bands": {name: available[name] for name in names},
        }
        geo = (src.crs, src.transform)
    return index, metadata, geo


def _display_range(tree: ast.expr, index: np.ndarray) -> tuple[tuple[float, float], str]:
    if _is_normalized_difference(tree):
        return (-1.0, 1.0), "normalized difference"
    return _percentile_range(index), "2nd-98th percentile of the data"


def _png(arr: np.ndarray, vrange: tuple[float, float], label: str, title: str) -> bytes | None:
    if not HAS_MATPLOTLIB:
        return None
    return _render_index_png(arr, colormap_name=COLORMAP, vmin=vrange[0], vmax=vrange[1], title=title, label=label)


def _settings(config: dict[str, Any] | None) -> tuple[ast.expr, str, str, float]:
    """``(tree, formula, name, threshold)`` from a band-math step config."""
    config = config or {}
    formula = config.get("formula")
    tree, _ = _parse(formula)
    formula = str(formula).strip()
    name = str(config.get("name") or formula).strip()
    threshold = float(config.get("threshold", DEFAULT_THRESHOLD))
    if not threshold > 0:
        raise ValueError(f"Band-math threshold must be positive, got {threshold}.")
    return tree, formula, name, threshold


def calculate(raster_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Band math on one raster; writes ``rasters["index"]`` and ``visualizations["index_map"]`` (+ ``index_map_bare``).

    Returns ``{metadata, statistics, visualizations, rasters, path, index_type, formula, name}``;
    ``statistics`` holds min/max/mean/std/median, valid/total pixels, and the
    PNG's ``display_range`` with its ``display_range_source``.
    """
    tree, formula, name, _ = _settings(config)
    index, metadata, (crs, transform) = _compute(raster_path, tree, formula)
    vrange, source = _display_range(tree, index)
    stats: dict[str, Any] = {**_stats(index), "display_range": list(vrange), "display_range_source": source}
    result: dict[str, Any] = {
        "metadata": metadata,
        "statistics": stats,
        "path": str(raster_path),
        "index_type": "band-math",
        "formula": formula,
        "name": name,
    }
    visualizations = {"index_map": _png(index, vrange, name, f"{name} - {Path(raster_path).stem}")}
    return artifact_files.save(
        result,
        {"index": index},
        visualizations,
        crs=crs,
        transform=transform,
        styles={"index": artifact_files.index_style(COLORMAPS[COLORMAP], *vrange)},
        bare={"index_map": "index"},
    )


def calculate_change(before_path: str, after_path: str, config: dict[str, Any]) -> dict[str, Any]:
    """Band math on two dates and the difference ``after - before``.

    Writes rasters ``change``, ``index_before``, ``index_after`` and PNGs
    ``change_map``, ``index_before``, ``index_after`` (each with a map-only
    ``<key>_bare``: change on the legend ramp over ``display_range``). ``statistics`` holds the
    change statistics (as :func:`dta.dti.algorithms.spectral_index.run_change`,
    with ``threshold`` from ``config.threshold``, default 0.2, and the areas in
    km² when the pixel size is in metres: ``increase_area_km2``, ``decrease_area_km2``
    and ``areas_km2: {increase, decrease, stable}``, else ``areas_km2: None``);
    ``index_statistics`` the per-date ones. ``class_statistics`` (as
    :func:`dta.dti.algorithms.change_detection.change_class_statistics`) classes the change at
    ±``threshold``: ``Wetter`` / ``Drier`` / ``No change`` for NDMI, else ``Increase`` /
    ``Decrease`` / ``Stable``.
    """
    tree, formula, name, threshold = _settings(config)
    before, _, _ = _compute(before_path, tree, formula)
    after, metadata, (crs, transform) = _compute(after_path, tree, formula)
    if before.shape != after.shape:
        raise ValueError(f"{name} arrays must have same shape; got {before.shape} and {after.shape}.")

    change = after - before
    valid = change[np.isfinite(change)]
    stats: dict[str, Any] = {"threshold": threshold, "total_valid_pixels": int(valid.size)}
    if valid.size > 0:
        stats.update(
            {
                "min_change": float(np.min(valid)),
                "max_change": float(np.max(valid)),
                "mean_change": float(np.mean(valid)),
                "std_change": float(np.std(valid)),
                "median_change": float(np.median(valid)),
                "increase_pixels": int(np.sum(valid > threshold)),
                "decrease_pixels": int(np.sum(valid < -threshold)),
                "stable_pixels": int(np.sum(np.abs(valid) <= threshold)),
            }
        )
        pixel_km2 = pixel_area_km2(crs, transform)
        if pixel_km2 is not None:
            stats["increase_area_km2"] = round(stats["increase_pixels"] * pixel_km2, 3)
            stats["decrease_area_km2"] = round(stats["decrease_pixels"] * pixel_km2, 3)
            # Same shape as algorithms/change-detection's.
            stats["areas_km2"] = {
                "increase": stats["increase_area_km2"],
                "decrease": stats["decrease_area_km2"],
                "stable": round(stats["stable_pixels"] * pixel_km2, 3),
            }
        else:
            stats["areas_km2"] = None
    with np.errstate(invalid="ignore"):
        increase, decrease = change > threshold, change < -threshold
    names = CHANGE_CLASS_NAMES.get(name.upper(), INCREASE_CLASS_NAMES)
    class_statistics = change_class_statistics(change, increase, decrease, names, pixel_area_km2(crs, transform))
    change_range = _percentile_range(change, symmetric=True)
    stats.update({"display_range": list(change_range), "display_range_source": "98th percentile of |change|"})

    index_range, _ = _display_range(tree, np.concatenate([before.ravel(), after.ravel()]))
    result: dict[str, Any] = {
        "statistics": stats,
        "class_statistics": class_statistics,
        "index_statistics": {"before": _stats(before), "after": _stats(after)},
        "metadata": {**metadata, "before_path": str(before_path), "after_path": str(after_path)},
        "before_path": str(before_path),
        "after_path": str(after_path),
        "index_type": "band-math",
        "formula": formula,
        "name": name,
    }
    visualizations = {
        "change_map": _png(change, change_range, f"Δ{name}", f"Δ{name} (after - before)"),
        "index_before": _png(before, index_range, name, f"{name} - {Path(before_path).stem}"),
        "index_after": _png(after, index_range, name, f"{name} - {Path(after_path).stem}"),
    }
    rasters = {"change": change, "index_before": before, "index_after": after}
    index_look = artifact_files.index_style(COLORMAPS[COLORMAP], *index_range)
    styles = {
        "change": artifact_files.change_style(change_range[1]),
        "index_before": index_look,
        "index_after": index_look,
    }
    bare = {"change_map": "change", "index_before": "index_before", "index_after": "index_after"}
    return artifact_files.save(result, rasters, visualizations, crs=crs, transform=transform, styles=styles, bare=bare)


def run(
    RasterPath: str | None = None,  # noqa: N803  # CamelCase mirrors registry input key
    RasterPathBefore: str | None = None,  # noqa: N803
    RasterPathAfter: str | None = None,  # noqa: N803
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Registry entrypoint: one raster (``algorithms/band-math``) or two dates (``algorithms/band-math-change``).

    ``config`` is the plan step's ``{"formula": ..., "name": ..., "threshold": ...}``.
    """
    if RasterPath and not (RasterPathBefore or RasterPathAfter):
        return calculate(RasterPath, config or {})
    if RasterPathBefore and RasterPathAfter and not RasterPath:
        return calculate_change(RasterPathBefore, RasterPathAfter, config or {})
    raise ValueError("Band math needs either RasterPath, or RasterPathBefore and RasterPathAfter.")
