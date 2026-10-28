"""Tests for the LULC change-hazard model: feature builder (offline parts), bundle, prediction and runner output.

The parity test against the training cube is behind ``@pytest.mark.network`` and needs ``GEE_PROJECT_ID`` and
``LULC_REPO`` (a checkout of the lulc repository with its dataset and Stage A cube); CI never runs it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.dti import artifact_files
from dta.dti.models.lulc_hazard import features, history, inference, stats

KPI_ORIGIN = (-303030.0, 5803330.0)


# ----------------------------------------------------------------- feature builder, offline parts


def test_mode3_ignores_nodata_and_breaks_ties_to_centre() -> None:
    a = np.array(
        [
            [1, 1, 0, 2, 2, 3],
            [2, 2, 0, 3, 3, 2],
            [0, 0, 0, 2, 3, 0],
        ],
        np.uint8,
    )
    # block 1: two 1s, two 2s, centre 2 -> 2; block 2: three 2s, three 3s, centre 3 -> 3
    assert features.mode3(a, 3).tolist() == [[2, 3]]
    assert features.mode3(np.zeros((3, 3), np.uint8), 3).tolist() == [[0]]


def test_mean3_is_nan_aware() -> None:
    a = np.full((3, 6), np.nan, np.float32)
    a[0, 0], a[1, 1] = 1.0, 3.0
    assert features.mean3(a)[0, 0] == 2.0
    assert np.isnan(features.mean3(a)[0, 1])


def test_neighbourhood_shares_count_valid_cells_only() -> None:
    cls = np.zeros((5, 5), np.uint8)
    cls[2, 2], cls[2, 3] = 1, 2
    shares = features.neighbourhood_shares(cls, radii=(3,), k=2)[3]
    assert shares[0, 2, 2] == pytest.approx(0.5) and shares[1, 2, 2] == pytest.approx(0.5)
    assert np.isnan(shares[0, 0, 0])  # no valid cell in the square


def test_grid_snaps_to_the_training_lattice_with_a_halo() -> None:
    g = features.grid_for_bbox([30.40, 50.35, 30.55, 50.45], "EPSG:32636", KPI_ORIGIN)
    core_x0, core_y0 = g.x0 + features.HALO * features.RES, g.y0 - features.HALO * features.RES
    assert (core_x0 - KPI_ORIGIN[0]) % features.RES == 0 and (core_y0 - KPI_ORIGIN[1]) % features.RES == 0
    assert (g.width, g.height) == (g.nx + 2 * features.HALO, g.ny + 2 * features.HALO)
    assert 10 < g.nx * features.RES / 1000 < 12 and 10 < g.ny * features.RES / 1000 < 12
    assert g.transform().c == core_x0 and g.transform().f == core_y0


def test_grid_auto_crs_is_the_utm_zone_of_the_centre() -> None:
    assert features.grid_for_bbox([7.9, 46.5, 8.1, 46.6]).crs == "EPSG:32632"  # Bernese Oberland
    assert features.utm_crs(30.5, 50.4) == "EPSG:32636"
    assert features.utm_crs(-70.0, -33.0) == "EPSG:32719"


def test_unknown_columns_are_refused_before_any_earth_engine_call() -> None:
    with pytest.raises(ValueError, match="osm_d_major"):
        features.build_features(
            features.Grid("EPSG:32636", 0, 0, 1, 1), 2020, ["elevation_m", "osm_d_major"], ee=object()
        )


# ----------------------------------------------------------------- bundle and prediction


def _booster_json(n_classes: int, n_cols: int, seed: int) -> dict[str, Any]:
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, n_cols)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int) + (n_classes > 2) * (x[:, 1] > 0.5).astype(int)
    # Native API, as the runner loads it: no scikit-learn needed.
    params = {"max_depth": 2, "seed": seed}
    params |= (
        {"objective": "multi:softprob", "num_class": n_classes} if n_classes > 2 else {"objective": "binary:logistic"}
    )
    booster = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=5)
    return json.loads(booster.save_raw("json").decode())


def _bundle(columns: list[str]) -> dict[str, Any]:
    n = len(columns)
    quantiles = list(np.linspace(0.0, 1.0, 101))
    return {
        "format": inference.FORMAT,
        "model_id": "test-hazard",
        "version": "0",
        "target": "esri",
        "legend": list(features.ESRI_CLASSES),
        "columns": columns,
        "interval_years": 4,
        "t0_min": 2017,
        "dem": "glo30_2024_1",
        "grid": {"crs": "EPSG:32636", "origin": list(KPI_ORIGIN)},
        "domain": {"bbox": [22.0, 44.0, 41.0, 53.0], "name": "Ukraine"},
        "from_classes": {
            "Trees": {
                "to": ["Trees", "Crops", "Rangeland"],
                "prior_correction": [1.0, 0.5, 2.0],
                "booster": _booster_json(3, n, 1),
            },
            "Crops": {"to": ["Crops", "Built"], "prior_correction": [1.0, 1.0], "booster": _booster_json(2, n, 2)},
            "Water": {"to": ["Water"], "prior_correction": [1.0], "booster": None},
        },
        "transitions": [["Trees", "Crops"], ["Crops", "Built"]],
        "reference_quantiles": {"any_change": quantiles, "Trees->Crops": quantiles, "Crops->Built": quantiles},
    }


def _cube(columns: list[str], ny: int = 6, nx: int = 5) -> features.FeatureCube:
    rng = np.random.default_rng(0)
    grid = features.Grid("EPSG:32636", 300000.0, 5600000.0, nx, ny)
    start = np.zeros((ny, nx), np.uint8)
    start[0:2] = features.ESRI_CLASSES.index("Trees") + 1
    start[2:4] = features.ESRI_CLASSES.index("Crops") + 1
    start[4, :2] = features.ESRI_CLASSES.index("Water") + 1
    start[4, 2:] = features.ESRI_CLASSES.index("Built") + 1  # no model for Built
    feats = {c: rng.normal(size=(ny, nx)).astype(np.float32) for c in columns}
    return features.FeatureCube(grid, 2021, start, feats)


COLUMNS = ["elevation_m", "slope_deg", "esri_nbr3_Trees"]


def test_check_bundle_accepts_the_documented_shape_and_names_what_is_wrong() -> None:
    bundle = _bundle(COLUMNS)
    inference.check_bundle(bundle)
    bad = {**bundle, "transitions": [["Trees", "Built"]]}
    with pytest.raises(inference.BundleError, match="Trees->Built"):
        inference.check_bundle(bad)
    bad = {**bundle, "columns": ["elevation_m", "osm_d_major"]}
    with pytest.raises(inference.BundleError, match="osm_d_major"):
        inference.check_bundle(bad)


def test_predict_applies_the_prior_correction_and_leaves_unmodelled_classes_unranked() -> None:
    xgb = pytest.importorskip("xgboost")
    bundle = _bundle(COLUMNS)
    cube = _cube(COLUMNS)
    x, start = cube.matrix(COLUMNS), cube.start_class.ravel()
    p_change, p_trans = inference.predict(bundle, x, start)

    trees = np.flatnonzero(start == features.ESRI_CLASSES.index("Trees") + 1)
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(bundle["from_classes"]["Trees"]["booster"]), "utf-8"))
    raw = booster.predict(xgb.DMatrix(x[trees])) * np.array([1.0, 0.5, 2.0])
    p = raw / raw.sum(1, keepdims=True)
    np.testing.assert_allclose(p_change[trees], 1 - p[:, 0], rtol=1e-5)
    np.testing.assert_allclose(p_trans["Trees->Crops"][trees], p[:, 1], rtol=1e-5)

    water = start == features.ESRI_CLASSES.index("Water") + 1
    assert np.all(p_change[water] == 0.0)  # one target class: always stays
    built = start == features.ESRI_CLASSES.index("Built") + 1
    assert np.all(np.isnan(p_change[built]))  # no model for the class
    assert np.all(np.isnan(p_trans["Crops->Built"][trees]))  # outside the from-class


def test_percentile_is_mid_rank_against_the_reference() -> None:
    q = list(np.linspace(0, 1, 11))
    out = inference.percentile(np.array([-1.0, 0.0, 0.55, 2.0, np.nan]), q)
    assert out[0] == 0.0 and out[3] == 100.0 and np.isnan(out[4])
    assert out[1] == pytest.approx(100 * 1 / 22)  # equals the lowest quantile: half of one step
    assert out[2] == pytest.approx(100 * 6 / 11)


# ----------------------------------------------------------------- runner output


def test_write_result_saves_styled_rasters_and_no_arrays(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    bundle = _bundle(COLUMNS)
    with artifact_files.step_outputs(tmp_path, "1_ChangeHazardMap"):
        result = inference.write_result(bundle, _cube(COLUMNS), [30.0, 50.0, 30.1, 50.1])

    assert result["index_type"] == "change_rank"
    ranks = {"change_rank", "rank_trees_to_crops", "rank_crops_to_built"}
    assert set(result["rasters"]) == ranks | {f"{k}_local" for k in ranks} | {"start_class"}
    assert "change_rank_map_bare" in result["visualizations"]
    json.dumps(result)  # references and numbers only
    with rasterio.open(result["rasters"]["change_rank"]["path"]) as src:
        rank = src.read(1)
        assert src.tags()["DT4LC_KIND"] == "index"
        assert src.crs.to_string() == "EPSG:32636" and src.shape == (6, 5)
    assert np.nanmin(rank) >= 0 and np.nanmax(rank) <= 100
    assert np.isnan(rank[4, 2:]).all()  # Built: no model
    with rasterio.open(result["rasters"]["start_class"]["path"]) as src:
        assert src.tags()["DT4LC_KIND"] == "class"
    stats = result["statistics"]
    # 30 cells: 5 no data, 10 Trees, 10 Crops, 2 Water (ranked), 3 Built (no model)
    assert stats["cells_without_model"] == 3 and stats["valid_pixels"] == 22
    assert stats["start_class_percent"]["Trees"] == pytest.approx(100 * 10 / 25)


def _write_bundle(tmp_path: Path) -> Path:
    (tmp_path / inference.BUNDLE_FILE).write_text(json.dumps(_bundle(COLUMNS)))
    return tmp_path


def _footprint(tmp_path: Path, x0: float, y0: float, size_m: float, crs: str = "EPSG:32636") -> str:
    path = tmp_path / "area.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(x0, y0, size_m / 10, size_m / 10),
    ) as dst:
        dst.write(np.zeros((1, 10, 10), np.uint8))
    return str(path)


def test_run_refuses_areas_outside_the_domain_and_too_large(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    model_dir = _write_bundle(tmp_path)
    swiss = _footprint(tmp_path, 400000.0, 5170000.0, 5000.0, "EPSG:32632")
    with pytest.raises(inference.OutsideDomainError, match="Ukraine"):
        inference.run(swiss, str(model_dir))
    big = _footprint(tmp_path, 300000.0, 5600000.0, 30000.0)  # 900 km² near Kyiv
    with pytest.raises(ValueError, match="at most 400 km²"):
        inference.run(big, str(model_dir))
    with pytest.raises(ValueError, match="t0 >= 2017"):
        inference.run(big, str(model_dir), t0=2016)


def test_missing_bundle_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(inference.BundleError, match="not installed"):
        inference.load_bundle(tmp_path)


def test_bundle_dem_defaults_to_the_v1_release_and_unknown_ones_are_refused() -> None:
    pytest.importorskip("xgboost")
    bundle = _bundle(COLUMNS)
    del bundle["dem"]
    inference.check_bundle(bundle)  # a bundle from dataset v1.0 names no DEM
    assert inference.LEGACY_DEM == "glo30" and features.DEM_DEFAULT == "glo30_2024_1"
    with pytest.raises(inference.BundleError, match="Unknown DEM"):
        inference.check_bundle({**bundle, "dem": "srtm"})


def test_latest_t0_is_the_newest_year_with_both_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    have = {"esri": [2017, 2024, 2025], "ae": [2017, 2024]}
    monkeypatch.setattr(features, "available_years", lambda ee, grid, years: have)
    grid = features.Grid("EPSG:32636", 0, 0, 1, 1)
    assert features.latest_t0(None, grid, last=2026) == 2024
    assert features.latest_t0(None, grid, last=2026, need_ae=False) == 2025
    have = {"esri": [2025], "ae": [2024]}
    with pytest.raises(features.FeatureUnavailableError, match="AlphaEarth"):
        features.latest_t0(None, grid, last=2026)


def test_registry_wires_the_model() -> None:
    from dta.dti.executor import STEP_TO_MODEL_MAP
    from dta.dti.models.model_manager import AVAILABLE_MODELS
    from dta.dti.registry import get_item, load_registry

    item = get_item(load_registry(), "models/lulc-change-hazard")
    assert item.outputs == ["ChangeHazardMap"] and item.inputs == ["RasterPath"]
    assert item.runner.entrypoint == "dta.dti.models.lulc_hazard.inference"
    model_id = STEP_TO_MODEL_MAP["models/lulc-change-hazard"]
    assert AVAILABLE_MODELS[model_id].files == [inference.BUNDLE_FILE]


# ----------------------------------------------------------------- parity with the training cube (engineers only)


@pytest.mark.network
@pytest.mark.skipif(
    not (os.environ.get("GEE_PROJECT_ID") and os.environ.get("LULC_REPO")),
    reason="GEE_PROJECT_ID and LULC_REPO (lulc checkout with the Stage A cube) needed",
)
@pytest.mark.parametrize(("window", "t0"), [("u02_019", 2020), ("u26_008", 2021)])
def test_features_match_the_training_cube(window: str, t0: int) -> None:
    import ee
    import pandas as pd

    repo = Path(os.environ["LULC_REPO"])
    sys.path.insert(0, str(repo / "analyses" / "2026-10-06_industry-lulc-model" / "scripts"))
    import cube_io  # type: ignore[import-not-found]

    ee.Initialize(project=os.environ["GEE_PROJECT_ID"])
    w = pd.read_csv(repo / "artifacts" / "dataset_v1" / "windows.csv").set_index("window_id").loc[window]
    grid = features.Grid("EPSG:32636", float(w.x0_core) - 600, float(w.y0_core) + 600, 400, 400)
    labels, x, names = cube_io.load(window, t0, "esri", ("static", "hist_esri", "ae"))
    cols = [c for c in features.COLUMNS if c in names]
    cube = features.build_features(grid, t0, cols, ee=ee, dem="glo30")  # the DEM release the cube was built from

    assert np.array_equal(cube.start_class.ravel(), labels[t0])
    for c in cols:
        a, b = cube.features[c].ravel(), x[:, names.index(c)]
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), c
        tol = 2e-4 if c in features.AE_COLUMNS else 1e-4  # the cube stores AlphaEarth as float16
        assert np.nanmax(np.abs(a - b)) <= tol, c


# ----------------------------------------------------------------- format 2: hazard groups, horizons, forecast


def _bundle_2(columns: list[str]) -> dict[str, Any]:
    b1 = _bundle(columns)
    q = list(np.linspace(0.0, 1.0, 101))
    groups = {
        "any_change": {"title": "Any land-cover change", "from": ["Trees", "Crops", "Water"], "to": None},
        "forest_loss": {"title": "Forest loss", "from": ["Trees"], "to": ["Crops", "Rangeland"]},
        "urban_expansion": {"title": "Urban expansion", "from": ["Crops", "Rangeland"], "to": ["Built"]},
    }
    hz = {"from_classes": b1["from_classes"], "reference_quantiles": dict.fromkeys(groups, q)}
    return {
        **{
            k: b1[k] for k in ("model_id", "version", "target", "legend", "columns", "t0_min", "dem", "grid", "domain")
        },
        "format": inference.FORMAT_2,
        "groups": groups,
        "horizons": {"2": {"pair": [2023, 2025], **hz}, "3": {"pair": [2022, 2025], **hz}},
        "served": {
            "hazards": {"any_change": 3, "forest_loss": 2, "urban_expansion": 0},
            "map": 3,
            "amounts_expected": {"2": True, "3": False},
        },
        "skill": {"3": {"hazards": {"any_change": {"auc": 0.8}}}},
    }


def test_format_1_converts_to_one_horizon_serving_hazards_only() -> None:
    pytest.importorskip("xgboost")
    b2 = inference.as_format_2(_bundle(COLUMNS))
    inference.check_bundle(b2)
    assert set(b2["groups"]) == {"any_change", "trees_to_crops", "crops_to_built"}
    assert b2["served"] == {"hazards": dict.fromkeys(b2["groups"], 4), "map": 0, "amounts_expected": {}}


def test_check_bundle_2_refuses_a_served_horizon_without_models() -> None:
    pytest.importorskip("xgboost")
    b = _bundle_2(COLUMNS)
    inference.check_bundle(b)
    bad = json.loads(json.dumps(b))
    bad["served"]["map"] = 4
    with pytest.raises(inference.BundleError, match=r"\[4\]"):
        inference.check_bundle(bad)


def test_hazards_take_each_group_at_its_own_horizon_and_skip_unserved_groups(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    with artifact_files.step_outputs(tmp_path, "1_ChangeHazardMap"):
        result = inference.write_result(_bundle_2(COLUMNS), _cube(COLUMNS), [30.0, 50.0, 30.1, 50.1])
    hazards = {h["group"]: h for h in result["statistics"]["hazards"]}
    assert set(hazards) == {"any_change", "forest_loss"}  # urban expansion is not served
    assert hazards["any_change"]["year"] == 2024 and hazards["forest_loss"]["year"] == 2023
    assert hazards["any_change"]["skill"] == {"auc": 0.8}
    assert result["metadata"]["forecast_to"] == 2024
    with rasterio.open(result["rasters"]["rank_forest_loss"]["path"]) as src:
        r = src.read(1)
    assert np.isfinite(r[0:2]).all() and np.isnan(r[2:]).all()  # Trees cells only
    with rasterio.open(result["rasters"]["rank_forest_loss_local"]["path"]) as src:
        loc = src.read(1)[0:2]
    assert loc.min() > 0 and loc.max() < 100 and np.isclose(np.mean(loc), 50.0)


def test_local_percentile_is_mid_rank_within_the_area() -> None:
    out = inference.local_percentile(np.array([0.1, 0.2, 0.2, np.nan, 0.4]))
    assert np.allclose(out[[0, 1, 2, 4]], [12.5, 50.0, 50.0, 87.5]) and np.isnan(out[3])


def test_trend_grid_is_a_12_km_square_on_the_area_lattice_and_centred() -> None:
    g = features.Grid("EPSG:32636", 300000.0, 5600000.0, 100, 60)
    t = inference.trend_grid(g)
    assert (t.nx, t.ny) == (400, 400)
    assert t.x0 + features.RES * (t.nx / 2) == pytest.approx(g.x0 + features.RES * (g.nx / 2))
    assert ((t.x0 - g.x0) / features.RES).is_integer() and ((t.y0 - g.y0) / features.RES).is_integer()
    big = features.Grid("EPSG:32636", 0.0, 0.0, 500, 450)
    assert (inference.trend_grid(big).nx, inference.trend_grid(big).ny) == (500, 450)


def test_forecast_follows_the_trend_amounts_and_words_them_by_the_bundle(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    bundle = _bundle_2(COLUMNS)
    cube = _cube(COLUMNS)
    crops, built = features.ESRI_CLASSES.index("Crops") + 1, features.ESRI_CLASSES.index("Built") + 1
    a = np.full((10, 10), crops, np.uint8)
    b = a.copy()
    b[:5] = built  # half of the trend area's Crops became Built
    grid = cube.grid
    with artifact_files.step_outputs(tmp_path, "1_LULCForecast"):
        res3 = inference.write_forecast(bundle, cube, 3, [a, b], (2018, 2021), grid, [30.0, 50.0, 30.1, 50.1])
        res2 = inference.write_forecast(bundle, cube, 2, [a, b], (2019, 2021), grid, [30.0, 50.0, 30.1, 50.1])
    st = res3["statistics"]
    rows = {r["class"]: r for r in st["classes"]}
    assert rows["Crops"]["after_ha"] == pytest.approx(5 * stats.CELL_HA)  # 10 Crops cells, half move
    assert rows["Built"]["change_ha"] == pytest.approx(5 * stats.CELL_HA)
    assert rows["Trees"]["change_ha"] == 0  # Trees did not change in the trend area
    assert st["amounts"] == "trend scenario" and st["summary"][0].startswith("If the 2018–2021 trend continues")
    assert res2["statistics"]["amounts"] == "expected" and res2["statistics"]["summary"][0].startswith("By 2023,")
    assert set(res3["rasters"]) == {"class_start", "class_forecast", "changed"}


def test_forecast_is_refused_beyond_the_checked_horizon(tmp_path: Path) -> None:
    pytest.importorskip("xgboost")
    (tmp_path / inference.BUNDLE_FILE).write_text(json.dumps(_bundle_2(COLUMNS)))
    area = _footprint(tmp_path, 300000.0, 5600000.0, 3000.0)
    with pytest.raises(ValueError, match="up to 3 years"):
        inference.run_forecast(area, str(tmp_path), horizon=4)
    _write_bundle(tmp_path)  # format 1: no map
    with pytest.raises(inference.BundleError, match="hazards only"):
        inference.run_forecast(area, str(tmp_path))


# ----------------------------------------------------------------- statistics and history


def test_majority_breaks_ties_with_the_latest_year() -> None:
    stack = np.array([[[1, 2]], [[1, 3]], [[2, 4]]], np.uint8)  # cell 0: 1,1,2; cell 1: 2,3,4 (all differ)
    assert stats.majority(stack).tolist() == [[1, 4]]


def test_demand_splits_each_class_by_its_rate_and_keeps_classes_absent_from_the_trend() -> None:
    start = np.array([4] * 10 + [2] * 3, np.uint8)
    rate = np.zeros((8, 8))
    rate[3, 3], rate[3, 7] = 0.75, 0.25  # Crops: a quarter to Rangeland
    q = stats.demand(start, rate)
    assert q[3, 3] + q[3, 7] == 10 and q[3, 7] in (2, 3)
    assert q[1, 1] == 3  # Trees absent from the trend: all stay


def test_allocate_fills_quotas_by_score() -> None:
    start = np.array([4, 4, 4, 0], np.uint8)
    quota = np.zeros((8, 8), int)
    quota[3, 4] = 1  # one Crops cell becomes Built
    score = np.zeros((4, 8), np.float32)
    score[:, 4] = [0.2, 0.9, 0.5, 1.0]
    assert stats.allocate(start, quota, score).tolist() == [4, 5, 4, 0]


def test_sentences_name_the_largest_net_change_and_its_main_destination() -> None:
    m = np.zeros((8, 8), int)
    m[3, 3], m[3, 7], m[7, 7] = 60, 40, 100  # Crops: 100 -> 60, 40 to Rangeland
    out = stats.sentences(m, "Between 2021 and 2025,", "{past}")
    assert out[0] == "Between 2021 and 2025, Crops decreased by 40 % (4 ha), mainly to Rangeland (4 ha)."
    assert out[1].startswith("Between 2021 and 2025, Rangeland increased by 40 %")


def test_history_summary_counts_areas_per_year_and_damps_flicker() -> None:
    crops, rng_ = 4, 8
    cls = np.full((9, 2, 2), crops, np.uint8)
    cls[4, 0, 0] = rng_  # one-year flicker: not a change between the periods
    cls[6:, 1, 1] = rng_  # a lasting change
    s = history.summarise(cls, list(range(2017, 2026)))
    assert s["area_ha"]["Crops"][0] == pytest.approx(4 * stats.CELL_HA)
    assert s["changed_percent"] == 25.0 and s["periods"] == {"start": [2017, 2019], "end": [2023, 2025]}
    assert s["summary"][0].startswith("Between 2017–2019 and 2023–2025, Crops decreased by 25 %")
