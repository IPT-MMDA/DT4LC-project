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

The trained bundle and its predictions are :mod:`.model`'s.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dta.dti import artifact_files

from . import features, stats
from .model import BundleError, group_score, load_bundle, local_percentile, percentile, probabilities, source_of

logger = logging.getLogger(__name__)

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
OTHER_COLOUR = "#4d4d4d"  # a group without its own hue
# Risk levels on the national rank, highest first: (raster value, name, lowest percentile, wording; {region}: the
# bundle's region, :func:`region_name`). Cells below the last level are left uncoloured, so the start map drawn under
# them shows through.
RISK_LEVELS = (
    (3, "High", 90.0, "top 10 % in {region}"),
    (2, "Elevated", 75.0, "next 15 %"),
    (1, "Moderate", 50.0, "next 25 %"),
)
# What each class is called in sentences ("21.4 km² of 169 km² of cropland and grassland").
LAND_WORDS = {
    "Crop": "cropland",
    "Crops": "cropland",
    "Grass": "grassland",
    "Rangeland": "rangeland",
    "Tree": "tree cover",
    "Trees": "tree cover",
    "Built": "built-up land",
    "Water": "water",
    "Wetland": "wetland",
    "FloodedVeg": "flooded vegetation",
    "Bare": "bare land",
    "Snow": "snow and ice",
}


def land_text(classes: list[str], legend: tuple[str, ...]) -> str:
    """The land a group starts from, in words: "cropland and grassland"; "land" for every class."""
    if set(classes) >= set(legend):
        return "land"
    words = [LAND_WORDS.get(c, c.lower()) for c in classes]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


LEVEL_TINT = {3: 1.0, 2: 0.6, 1: 0.3}  # share of the hazard's hue mixed with white

# Named after the classes they move between (KPI "Built", "Tree"), not after "urban" or "forest", which the labels
# do not separate; they win over a bundle's own titles.
GROUP_TITLES = {
    ANY: "Any change",
    "urban_expansion": "Built-up expansion",
    "forest_loss": "Tree cover loss",
    "cropland_abandonment": "Cropland abandonment",
    "recultivation": "Cropland expansion",
    "overgrowth": "Tree cover gain",
}


class OutsideDomainError(ValueError):
    """The area lies outside the region the model was trained on."""


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


def region_name(domain: dict[str, Any]) -> str:
    """The short name of the region the ranks compare with: ``domain.short``, else ``domain.name`` without its
    bracket ("Ukraine (its bounding box)" -> "Ukraine")."""
    return domain.get("short") or domain.get("name", "").split(" (")[0].strip() or "the training region"


def _ee() -> Any:
    import ee

    from dta.dti.data_sources import gee_common
    from dta.dti.data_sources.gee_region import EarthEngineNotConfiguredError

    if not gee_common.initialize_gee():
        raise EarthEngineNotConfiguredError(
            "Earth Engine isn't configured on this server, so I can't build the change-hazard features."
        )
    return ee


def _area(RasterPath: str, bundle: dict[str, Any]) -> tuple[list[float], features.Grid]:  # noqa: N803
    bbox = footprint_wgs84(RasterPath)
    check_domain(bbox, bundle["domain"])
    g = bundle["grid"]
    grid = features.grid_for_bbox(bbox, g.get("crs", "auto"), g.get("origin", (0.0, 0.0)))
    check_size(grid, "the change-hazard model")
    return bbox, grid


def check_size(grid: features.Grid, what: str) -> None:
    """ValueError if ``grid`` is larger than :data:`MAX_AREA_KM2`; ``what`` names the analysis in the message."""
    if grid.area_km2 > MAX_AREA_KM2:
        raise ValueError(
            f"This area is {grid.area_km2:.0f} km²; {what} runs on at most {MAX_AREA_KM2:.0f} km². "
            "Draw a smaller region."
        )


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
    year = features.latest_t0(ee, grid, first=t0_min, need_ae=need_ae, source=source) if t0 is None else t0
    return features.build_features(
        grid, year, bundle["columns"], ee=ee, check_years=not latest, dem=bundle["dem"], source=source
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


def _base_rate(hazard: dict[str, Any], region: str) -> str:
    """The sentence on how common the change was: " In the backtest, 2.8 % of such land across Ukraine made this
    change within 3 years."; "" without a backtest rate."""
    rate = (hazard.get("skill") or {}).get("observed_rate")
    if not isinstance(rate, int | float):
        return ""
    return (
        f" In the backtest, {100.0 * rate:.2g} % of such land across {region} made this change within "
        f"{hazard['horizon_years']} years."
    )


def write_result(bundle: dict[str, Any], cube: features.FeatureCube, bbox: list[float]) -> dict[str, Any]:
    """Predict hazards on ``cube`` and save their rasters as job artifacts; returns the step result."""
    source = source_of(bundle)
    names = source.classes
    region = region_name(bundle["domain"])
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
            probs[h] = probabilities(bundle, hz["from_classes"], x, start)
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
                "title": GROUP_TITLES.get(g) or bundle["groups"][g].get("title") or g,
                "colour": colour,
                "raster": key,
                "level_raster": f"{key}_level",
                "levels": [
                    {
                        "level": name,
                        "range": wording.format(region=region),
                        "colour": shades[value],
                        "km2": stats.km2((level == value).sum()),
                        "percent": round(100.0 * float((level == value).sum()) / n_ok, 1),
                    }
                    for value, name, _, wording in RISK_LEVELS
                ],
                "horizon_years": h,
                "year": cube.t0 + h,
                "eligible_km2": stats.km2(ok.sum()),
                "land": land_text(list(bundle["groups"][g]["from"]), names),
                "to_land": land_text(list(to), names) if (to := bundle["groups"][g].get("to")) else "another class",
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
            *(
                f"{h['title']} ({h['land']} to {h['to_land']}) by {h['year']}: {stats.km2_text(h['top_decile_km2'])}"
                f" of the {stats.km2_text(h['eligible_km2'])} of {h['land']} ({h['top_decile_percent']:g} %) is at "
                f"high risk, i.e. in the top 10 % of such land across {region}.{_base_rate(h, region)}"
                for h in hazards
                if h["top_decile_percent"] is not None
            ),
            "The hazards overlap (one cell can be at risk of several) and each is ranked on its own scale, so their "
            "high-risk areas do not add up to the high-risk area of any change.",
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
            "dem": bundle["dem"],
            "values": "percentile of the hazard's probability among the training cells that could make it; "
            "<key>_local: among this area's cells; ranks only",
            "domain": bundle["domain"].get("name"),
            "compared_with": region,
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
    p = probabilities(bundle, bundle["horizons"][str(h)]["from_classes"], cube.matrix(bundle["columns"]), start)
    trend_m = stats.matrix(trend[0], trend[1], k)
    if not trend_m.any():  # a year with no map reads as all no data, which is no trend rather than no change
        raise features.FeatureUnavailableError(
            f"{source.title} has no {pair[0]} or {pair[1]} map around this area, so there is no trend to follow."
        )
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
            "dem": bundle["dem"],
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
