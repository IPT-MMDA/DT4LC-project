"""Band-math algorithm: safe formulas, band names, NDMI/NDVI on the sample, change mode, plans."""

import builtins
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.config import ROOT_DIR
from dta.dti.algorithms import band_math, spectral_index
from dta.dti.coe.llm_planner import plan_with_llm
from dta.dti.coe.plan_validator import PlanError, validate
from dta.dti.executor import PipelineExecutor
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import ContextUnderstanding, ExecutionPlan, PlanStep

BEFORE = str(ROOT_DIR / "resources/kahovka_data/hlsl_20230601.tif")
AFTER = str(ROOT_DIR / "resources/kahovka_data/hlsl_20230609.tif")
NDMI = {"formula": "(B5 - B6) / (B5 + B6)", "name": "NDMI"}
SAMPLE_BANDS = ["B2", "B3", "B4", "B5", "B6", "red", "nir", "green", "blue", "swir"]


def _scaled(path: str, band: int) -> np.ndarray:
    """Band ``band`` of an int16 x10000 HLS crop as 0-1 reflectance, nodata NaN."""
    with rasterio.open(path) as src:
        return np.ma.filled(src.read(band, masked=True).astype(np.float64), np.nan) / 10000


def _direct_ndmi(path: str) -> np.ndarray:
    """(B5 - B6) / (B5 + B6) on the sample (B5 = band 4, B6 = band 5), non-finite NaN, clipped to [-1, 1]."""
    b5, b6 = _scaled(path, 4), _scaled(path, 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (b5 - b6) / (b5 + b6)
    nd[~np.isfinite(nd)] = np.nan
    return np.clip(nd, -1, 1)


def _read(ref: dict) -> np.ndarray:
    with rasterio.open(ref["path"]) as src:
        return src.read(1).astype(np.float64)


def _assert_same(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)


def _write(path: Path, data: np.ndarray, descriptions: list[str] | None) -> str:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(0, 120, 30, 30),
    ) as dst:
        dst.write(data.astype("float32"))
        for i, desc in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, desc)
    return str(path)


# --- Formulas ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("__import__('os')", r"Formula not allowed: `__import__\('os'\)`"),
        ("B5.real", "Formula not allowed: `B5.real`"),
        ("abs(B5)", r"Formula not allowed: `abs\(B5\)`"),
        ("B5**2", r"Formula not allowed: `B5 \*\* 2`"),
        ("B5 if B4 else B6", "Formula not allowed: `B5 if B4 else B6`"),
        ("B5[0]", "Formula not allowed"),
        ("B5 > B4", "Formula not allowed"),
        ("lambda: B5", "Formula not allowed"),
        ("'B5'", "Formula not allowed"),
        ("B5 // B4", "Formula not allowed"),
        ("(B5 - foo) / B4", "Unknown band name 'foo'"),
        ("", "needs a formula"),
        ("   ", "needs a formula"),
        ("B5 + " * 200 + "B4", "too long"),
        ("(B5 - B6", "not a valid expression"),
        ("1 + 2", "uses no band"),
    ],
)
def test_unsafe_formulas_are_rejected(formula: str, message: str) -> None:
    assert len("B5 + " * 200 + "B4") == 1002
    with pytest.raises(ValueError, match=message):
        band_math.validate_formula(formula)


def test_validate_formula_returns_canonical_bands_in_order() -> None:
    assert band_math.validate_formula("(sr_b5 - B6) / (NIR + b6) * -2.5") == ["B5", "B6", "nir"]


def test_missing_band_names_the_file_bands() -> None:
    with pytest.raises(ValueError) as exc:
        band_math.validate_formula("(B5 - B7) / (B5 + B7)", SAMPLE_BANDS)
    assert str(exc.value) == (
        "Band B7 is not in this file (B2–B6). Available: B2, B3, B4, B5, B6, red, nir, green, blue, swir."
    )


def test_b7_on_the_sample_is_not_in_this_file() -> None:
    with pytest.raises(ValueError, match=r"Band B7 is not in this file \(B2–B6\)"):
        band_math.run(RasterPath=BEFORE, config={"formula": "(B5 - B7) / (B5 + B7)", "name": "NBR"})


def test_formula_is_never_evaluated_with_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    real_eval = builtins.eval
    seen: list[str] = []

    def spy(source: object, *args: object, **kwargs: object) -> object:
        seen.append(str(source))
        return real_eval(source, *args, **kwargs)  # type: ignore[arg-type]  # libraries (namedtuple) use eval too

    monkeypatch.setattr(builtins, "eval", spy)
    result = band_math.run(RasterPath=BEFORE, config={"formula": "(B5 - B6) / (B5 + B6) * 0.987"})
    assert result["statistics"]["valid_pixels"] > 0
    assert not [s for s in seen if "0.987" in s]


# --- Single raster ----------------------------------------------------------------


def test_ndmi_matches_direct_computation() -> None:
    result = band_math.run(RasterPath=BEFORE, config=NDMI)
    expected = _direct_ndmi(BEFORE)

    _assert_same(_read(result["rasters"]["index"]), expected)
    valid = expected[np.isfinite(expected)]
    stats = result["statistics"]
    for key, value in {
        "mean": valid.mean(),
        "std": valid.std(),
        "min": valid.min(),
        "max": valid.max(),
        "median": np.median(valid),
    }.items():
        assert stats[key] == pytest.approx(value, abs=1e-6), key
    assert stats["valid_pixels"] == valid.size
    assert stats["total_pixels"] == expected.size
    assert stats["display_range"] == [-1.0, 1.0]
    assert stats["display_range_source"] == "normalized difference"
    assert result["formula"] == "(B5 - B6) / (B5 + B6)"
    assert result["name"] == "NDMI"
    assert result["metadata"]["bands"] == {"B5": 4, "B6": 5}
    png = Path(result["visualizations"]["index_map"]["path"])
    assert png.read_bytes().startswith(b"\x89PNG")


def test_roles_resolve_through_band_indices() -> None:
    by_role = band_math.run(RasterPath=BEFORE, config={"formula": "(nir - swir) / (swir + nir)"})
    _assert_same(_read(by_role["rasters"]["index"]), _direct_ndmi(BEFORE))
    assert by_role["name"] == "(nir - swir) / (swir + nir)"  # defaults to the formula


def test_ndvi_matches_spectral_index() -> None:
    """Same values where both are defined; band math leaves x/0 as NaN where spectral_index writes 0."""
    result = band_math.run(RasterPath=BEFORE, config={"formula": "(B5 - B4)/(B5 + B4)", "name": "NDVI"})
    ours = _read(result["rasters"]["index"])
    theirs = spectral_index.calculate_index(BEFORE, get_item(load_registry(), "algorithms/ndvi").config)[1]

    zero_denominator = (_scaled(BEFORE, 4) + _scaled(BEFORE, 3)) == 0
    assert zero_denominator.any()
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(theirs) | zero_denominator)
    both = np.isfinite(ours)
    np.testing.assert_allclose(ours[both], theirs[both], atol=1e-6)
    stats = result["statistics"]
    assert stats["mean"] == pytest.approx(theirs[both].mean(), abs=1e-6)
    assert stats["std"] == pytest.approx(theirs[both].std(), abs=1e-6)
    assert stats["median"] == pytest.approx(np.median(theirs[both]), abs=1e-6)


def test_other_formulas_use_the_percentile_display_range() -> None:
    result = band_math.run(RasterPath=BEFORE, config={"formula": "B5 / B4"})
    arr = _read(result["rasters"]["index"])
    lo, hi = np.percentile(arr[np.isfinite(arr)], [2, 98])
    assert result["statistics"]["display_range"] == pytest.approx([lo, hi], abs=1e-6)
    assert result["statistics"]["display_range_source"] == "2nd-98th percentile of the data"
    assert np.nanmax(arr) > 1  # not clipped


def test_codes_resolve_through_descriptions(tmp_path: Path) -> None:
    data = np.stack([np.full((4, 4), v) for v in (0.1, 0.2, 0.3)])
    path = _write(tmp_path / "sr.tif", data, ["SR_B4", "SR_B5", "SR_B6"])
    result = band_math.run(RasterPath=path, config={"formula": "B6 - SR_B4"})
    np.testing.assert_allclose(_read(result["rasters"]["index"]), 0.2, atol=1e-6)
    with pytest.raises(ValueError, match=r"Band B1 is not in this file \(B4–B6\)"):
        band_math.run(RasterPath=path, config={"formula": "B1"})


def test_index_fallback_only_without_descriptions(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = np.stack([np.full((4, 4), v) for v in (0.1, 0.2, 0.3, 0.4)])
    bare = _write(tmp_path / "bare.tif", data, None)
    with caplog.at_level(logging.WARNING, logger="dta.dti.algorithms.band_math"):
        result = band_math.run(RasterPath=bare, config={"formula": "B4 - B1"})
    np.testing.assert_allclose(_read(result["rasters"]["index"]), 0.3, atol=1e-6)
    assert "reads Bn as band n" in caplog.text

    named = _write(tmp_path / "named.tif", data, ["red", "green", "blue", "nir"])
    with pytest.raises(ValueError, match="Band B4 is not in this file"):
        band_math.run(RasterPath=named, config={"formula": "B4 - B1"})


def test_nodata_and_zero_denominators_are_nan(tmp_path: Path) -> None:
    data = np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.4)])
    data[:, 0, 0] = 0.0  # 0 / 0
    data[0, 1, 1] = np.nan
    path = _write(tmp_path / "nan.tif", data, ["B4", "B5"])
    arr = _read(band_math.run(RasterPath=path, config={"formula": "(B5 - B4) / (B5 + B4)"})["rasters"]["index"])
    assert np.isnan(arr[0, 0]) and np.isnan(arr[1, 1])
    assert np.isfinite(arr).sum() == 14


def test_run_needs_one_mode() -> None:
    with pytest.raises(ValueError, match="either RasterPath"):
        band_math.run(RasterPath=BEFORE, RasterPathBefore=BEFORE, RasterPathAfter=AFTER, config=NDMI)
    with pytest.raises(ValueError, match="either RasterPath"):
        band_math.run(RasterPathBefore=BEFORE, config=NDMI)


# --- Change mode ------------------------------------------------------------------


def test_change_is_after_minus_before() -> None:
    result = band_math.run(RasterPathBefore=BEFORE, RasterPathAfter=AFTER, config=NDMI)
    before, after = _direct_ndmi(BEFORE), _direct_ndmi(AFTER)

    _assert_same(_read(result["rasters"]["index_before"]), before)
    _assert_same(_read(result["rasters"]["index_after"]), after)
    _assert_same(_read(result["rasters"]["change"]), after - before)

    change = (after - before)[np.isfinite(after - before)]
    stats = result["statistics"]
    assert stats["mean_change"] == pytest.approx(change.mean(), abs=1e-6)
    assert stats["threshold"] == 0.2
    assert stats["increase_pixels"] == int((change > 0.2).sum())
    assert stats["decrease_pixels"] == int((change < -0.2).sum())
    assert stats["increase_area_km2"] == pytest.approx(stats["increase_pixels"] * 0.0009, abs=1e-3)
    assert stats["areas_km2"] == {
        "increase": stats["increase_area_km2"],
        "decrease": stats["decrease_area_km2"],
        "stable": round(stats["stable_pixels"] * 0.0009, 3),
    }
    assert stats["areas_km2"] == {"increase": 37.927, "decrease": 41.961, "stable": 450.94}
    assert result["index_statistics"]["before"]["mean"] == pytest.approx(np.nanmean(before), abs=1e-6)
    assert (result["formula"], result["name"]) == (NDMI["formula"], "NDMI")
    assert set(result["visualizations"]) == {
        "change_map",
        "index_before",
        "index_after",
        "change_map_bare",
        "index_before_bare",
        "index_after_bare",
    }


# --- Plans ------------------------------------------------------------------------


def _change_plan(config: dict) -> ExecutionPlan:
    return ExecutionPlan(
        flow="NDMI change",
        steps=[
            PlanStep(uses="input/file-before", binds={"RasterPathBefore": BEFORE}),
            PlanStep(uses="input/file-after", binds={"RasterPathAfter": AFTER}),
            PlanStep(uses="algorithms/band-math-change", config=config),
        ],
        outputs=["publish: chat"],
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"formula": "abs(B5)"}, "Formula not allowed"),
        ({"formula": "__import__('os').system('true')"}, "Formula not allowed"),
        ({"name": "NDMI"}, "needs a formula"),
        ({"formula": "(B5 - B7) / (B5 + B7)"}, r"Band B7 is not in this file \(B2–B6\)"),
    ],
)
def test_plan_validation_rejects_bad_band_math(config: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        validate(_change_plan(config), load_registry())


def test_plan_validation_rejects_config_on_other_steps() -> None:
    plan = ExecutionPlan(
        steps=[
            PlanStep(uses="input/file", binds={"RasterPath": BEFORE}),
            PlanStep(uses="algorithms/ndvi", config={"formula": "B5"}),
        ]
    )
    with pytest.raises(PlanError, match="takes no config"):
        validate(plan, load_registry())


def test_executor_runs_composed_ndmi_change(_outputs_in_tmp: Path) -> None:
    plan = validate(_change_plan(NDMI), load_registry())
    result = PipelineExecutor().execute(plan)

    out = result["artifacts"]["BandMathChangeMap"]
    assert (out["formula"], out["name"]) == (NDMI["formula"], "NDMI")
    files = [Path(r["path"]) for r in [*out["rasters"].values(), *out["visualizations"].values()]]
    assert len(files) == 9  # 3 GeoTIFFs, 3 PNGs and their 3 bare maps
    for f in files:
        assert f.is_file() and f.is_relative_to(_outputs_in_tmp)
        assert f.name.startswith("2_BandMathChangeMap_")
    _assert_same(_read(out["rasters"]["change"]), _direct_ndmi(AFTER) - _direct_ndmi(BEFORE))


def test_executor_runs_single_band_math() -> None:
    plan = ExecutionPlan(
        steps=[
            PlanStep(uses="input/file", binds={"RasterPath": BEFORE}),
            PlanStep(uses="algorithms/band-math", config=NDMI),
        ]
    )
    out = PipelineExecutor().execute(validate(plan, load_registry()))["artifacts"]["BandMathMap"]
    _assert_same(_read(out["rasters"]["index"]), _direct_ndmi(BEFORE))


@patch("dta.dti.coe.llm_planner.get_llm_router")
def test_llm_planner_keeps_band_math_config(mock_get_router: MagicMock) -> None:
    reply = {
        "steps": [
            {"uses": "input/file-before"},
            {"uses": "input/file-after"},
            {"uses": "algorithms/band-math-change", "config": NDMI},
        ],
        "reasoning": "NDMI for both dates, then the difference",
    }
    mock_get_router.return_value.generate.return_value = MagicMock(text=json.dumps(reply), provider="mock")
    ctx = ContextUnderstanding(goal="NDMI change", desired_outputs=[], required_inputs=[])
    plan = plan_with_llm(ctx, load_registry())
    assert plan.steps[2].config == NDMI
    assert "algorithms/band-math-change" in mock_get_router.return_value.generate.call_args[0][0][0].content
