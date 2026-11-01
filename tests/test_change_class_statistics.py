"""Per-class statistics of change results (v0.6.0 Result detail's Statistics table).

``class_statistics`` has the LULC/snow shape plus ``mean_change``; the class areas add up to
the valid region and match the cover areas the result already reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import rasterio

from dta.config import ROOT_DIR
from dta.dti.algorithms import band_math, change_detection
from dta.dti.artifact_files import CHANGE_RAMP
from dta.dti.raster import pixel_area_km2

BEFORE = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
AFTER = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
LOST, STABLE, GAINED = CHANGE_RAMP


def _pixel_km2() -> float:
    with rasterio.open(AFTER) as src:
        km2 = pixel_area_km2(src.crs, src.transform)
    assert km2 is not None
    return km2


def _check_shape(classes: dict[str, dict[str, Any]], names: list[str], valid_pixels: int) -> None:
    assert list(classes) == names
    assert [(c["id"], c["name"], c["colour"]) for c in classes.values()] == [
        (1, names[0], GAINED),
        (2, names[1], LOST),
        (0, names[2], STABLE),
    ]
    assert sum(c["pixels"] for c in classes.values()) == valid_pixels
    assert sum(c["percent"] for c in classes.values()) == pytest.approx(100, abs=0.02)
    valid_km2 = valid_pixels * _pixel_km2()
    assert sum(c["area_km2"] for c in classes.values()) == pytest.approx(valid_km2, abs=0.002)
    gained, lost, _ = classes.values()
    assert gained["mean_change"] > 0 > lost["mean_change"]


def test_ndwi_change_classes_follow_the_water_rule() -> None:
    result = change_detection.run(BEFORE, AFTER, "ndwi")

    classes = result["class_statistics"]
    _check_shape(classes, ["Water gained", "Water lost", "No change"], result["classification"]["total_valid_pixels"])
    water = result["statistics"]["water"]
    assert classes["Water gained"]["area_km2"] == water["gained_km2"] == 156.761
    assert classes["Water lost"]["area_km2"] == water["lost_km2"] == 0.603
    assert classes["Water gained"]["percent"] == 29.53


def test_ndvi_change_classes_follow_the_vegetation_rule() -> None:
    result = change_detection.run(BEFORE, AFTER, "ndvi")

    classes = result["class_statistics"]
    _check_shape(
        classes, ["Vegetation gained", "Vegetation lost", "No change"], result["classification"]["total_valid_pixels"]
    )
    vegetation = result["statistics"]["vegetation"]
    assert classes["Vegetation gained"]["area_km2"] == vegetation["gained_km2"]
    assert classes["Vegetation lost"]["area_km2"] == vegetation["lost_km2"]


def test_other_index_change_classes_are_increase_decrease_stable() -> None:
    result = change_detection.run(BEFORE, AFTER, "ndsi")

    classes = result["class_statistics"]
    _check_shape(classes, ["Increase", "Decrease", "Stable"], result["classification"]["total_valid_pixels"])
    areas = result["statistics"]["areas_km2"]
    assert (classes["Increase"]["area_km2"], classes["Decrease"]["area_km2"]) == (areas["increase"], areas["decrease"])


@pytest.mark.parametrize(("name", "names"), [("NDMI", ["Wetter", "Drier", "No change"]), ("NDBI", None)])
def test_band_math_change_classes_at_its_threshold(name: str, names: list[str] | None) -> None:
    formula = "(B5 - B6) / (B5 + B6)" if name == "NDMI" else "(B6 - B5) / (B6 + B5)"
    config = {"formula": formula, "name": name, "threshold": 0.2}

    result = band_math.run(RasterPathBefore=BEFORE, RasterPathAfter=AFTER, config=config)

    classes = result["class_statistics"]
    stats = result["statistics"]
    _check_shape(classes, names or ["Increase", "Decrease", "Stable"], stats["total_valid_pixels"])
    increase, decrease, _ = classes.values()
    assert (increase["pixels"], decrease["pixels"]) == (stats["increase_pixels"], stats["decrease_pixels"])
    if name == "NDMI":
        assert (increase["area_km2"], decrease["area_km2"]) == (37.927, 41.961)
