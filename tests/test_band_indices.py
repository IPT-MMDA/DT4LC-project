"""Band-role resolution from band descriptions (#64)."""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.config import ROOT_DIR
from dta.dti.algorithms import change_detection, ndvi
from dta.dti.raster import _roles_from_count, band_indices

# Constant reflectance by role, from the JOSS review.
REFLECTANCE = {"blue": 0.04, "green": 0.08, "red": 0.05, "nir": 0.40, "swir": 0.20, "other": 0.10}


def _write(path: Path, roles: list[str], descriptions: tuple[str, ...] | None) -> str:
    data = np.stack([np.full((8, 8), REFLECTANCE[r], dtype=np.float32) for r in roles])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=len(roles),
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(0, 240, 30, 30),
    ) as dst:
        dst.write(data)
        if descriptions is not None:
            dst.descriptions = descriptions
    return str(path)


def _roles(path: str) -> dict[str, int | None]:
    with rasterio.open(path) as src:
        return band_indices(src)


@pytest.mark.parametrize("descriptions", [("B", "G", "R", "NIR"), ("Blue", "Green", "Red", "NIR")])
def test_four_band_bgrn_with_descriptions(tmp_path: Path, descriptions: tuple[str, ...]) -> None:
    path = _write(tmp_path / "bgrn.tif", ["blue", "green", "red", "nir"], descriptions)

    assert _roles(path) == {"red": 3, "nir": 4, "green": 2, "blue": 1, "swir": None}
    assert round(ndvi.run(path)["statistics"]["mean"], 6) == 0.777778


def test_without_descriptions_uses_count_heuristic_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write(tmp_path / "rgbn.tif", ["red", "green", "blue", "nir"], None)

    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        roles = _roles(path)

    assert roles == {"red": 1, "nir": 4, "green": 2, "blue": 3, "swir": None}
    assert any("band count" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.parametrize("descriptions", [("Red", "Green", "Blue", ""), ("red", "", "", "nir")])
def test_partial_semantic_names_fill_rest_from_band_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, descriptions: tuple[str, ...]
) -> None:
    path = _write(tmp_path / "rgbn.tif", ["red", "green", "blue", "nir"], descriptions)

    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        roles = _roles(path)

    assert roles == {"red": 1, "nir": 4, "green": 2, "blue": 3, "swir": None}
    assert any("band count" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert round(ndvi.run(path)["statistics"]["mean"], 6) == 0.777778
    for index_type in ("ndvi", "ndwi", "evi"):
        with rasterio.open(change_detection.run(path, path, index_type)["rasters"]["index_before"]["path"]) as src:
            assert src.shape == (8, 8)


def test_change_detection_rejects_missing_band_role(tmp_path: Path) -> None:
    # Named B, G, R: the count heuristic's NIR (band 2) is already green, so NIR stays None.
    path = _write(tmp_path / "bgr.tif", ["blue", "green", "red"], ("blue", "green", "red"))

    assert _roles(path)["nir"] is None
    with pytest.raises(ValueError, match="nir"):
        change_detection.run(path, path, "ndvi")


def test_sentinel2_codes(tmp_path: Path) -> None:
    descriptions = ("B2", "B3", "B4", "B5", "B8", "B11", "B12")
    path = _write(tmp_path / "s2.tif", ["blue", "green", "red", "other", "nir", "swir", "other"], descriptions)

    assert _roles(path) == {"red": 3, "nir": 5, "green": 2, "blue": 1, "swir": 6}


def test_sentinel2_b8a_when_b8_absent(tmp_path: Path) -> None:
    descriptions = ("B8A", "B4", "B3", "B2", "B11", "B12")
    path = _write(tmp_path / "s2.tif", ["nir", "red", "green", "blue", "swir", "other"], descriptions)

    assert _roles(path) == {"red": 2, "nir": 1, "green": 3, "blue": 4, "swir": 5}


def test_unknown_codes_fall_back_to_band_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write(tmp_path / "mixed.tif", ["red", "green", "blue", "nir"], ("B2", "B3", "B4", "B13"))

    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        roles = _roles(path)

    assert roles == {"red": 1, "nir": 4, "green": 2, "blue": 3, "swir": None}
    assert any("band count" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_hls_l30_codes_reordered(tmp_path: Path) -> None:
    # Same codes as the Kahovka crop, in an order the 5-band count heuristic would misread.
    path = _write(tmp_path / "l30.tif", ["nir", "red", "green", "blue", "swir"], ("B5", "B4", "B3", "B2", "B6"))

    assert _roles(path) == {"red": 2, "nir": 1, "green": 3, "blue": 4, "swir": 5}


def test_hls_l30_codes_of_kahovka_crop(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        with rasterio.open(ROOT_DIR / "resources/kahovka_data/hlsl_20230601.tif") as src:
            assert src.descriptions == ("B2", "B3", "B4", "B5", "B6")
            roles = band_indices(src)

    assert roles == {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5}
    assert not any("band count" in r.getMessage() for r in caplog.records)


LANDSAT = {"red": "B4", "nir": "B5", "green": "B3", "blue": "B2", "swir": "B6"}
SENTINEL2 = {"red": "B4", "nir": "B8", "green": "B3", "blue": "B2", "swir": "B11"}


def _prefixed(roles: dict[str, str], prefix: str) -> dict[str, str]:
    return {role: prefix + code for role, code in roles.items()}


@pytest.mark.parametrize(
    ("descriptions", "expected"),
    [
        pytest.param(tuple(f"B{i}" for i in range(1, 12)), LANDSAT, id="landsat-8-9-B1..B11"),
        pytest.param(tuple(f"SR_B{i}" for i in range(1, 8)), _prefixed(LANDSAT, "SR_"), id="SR_B1..SR_B7"),
        pytest.param(
            (*(f"SR_B{i}" for i in range(1, 8)), "ST_B10"), _prefixed(LANDSAT, "SR_"), id="SR_B1..SR_B7+ST_B10"
        ),
        pytest.param(
            ("SR_B2", "SR_B3", "SR_B4", "SR_B5"), {**_prefixed(LANDSAT, "SR_"), "swir": None}, id="SR_B2..B5"
        ),
        pytest.param(("B2", "B3", "B4", "B5", "B6"), LANDSAT, id="hls-l30-B2..B6"),
        pytest.param(
            ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09", "B10", "B11"),
            {"red": "B04", "nir": "B05", "green": "B03", "blue": "B02", "swir": "B06"},
            id="hls-l30-full",
        ),
        pytest.param(
            (*(f"B{i}" for i in range(1, 9)), "B8A", *(f"B{i}" for i in range(9, 13))),
            SENTINEL2,
            id="sentinel2-B1..B12+B8A",
        ),
        pytest.param(("B2", "B3", "B4", "B8", "B11"), SENTINEL2, id="frontend-default-B2,B3,B4,B8,B11"),
        pytest.param(("B2", "B3", "B4", "B8"), {**SENTINEL2, "swir": None}, id="B2,B3,B4,B8"),
        pytest.param(("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11"), SENTINEL2, id="sentinel2-red-edge-B2..B8,B11"),
    ],
)
def test_band_code_families(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, descriptions: tuple[str, ...], expected: dict[str, str | None]
) -> None:
    path = _write(tmp_path / "codes.tif", ["other"] * len(descriptions), descriptions)

    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        roles = _roles(path)

    assert {role: descriptions[i - 1] if i else None for role, i in roles.items()} == expected
    assert not any("band count" in r.getMessage() for r in caplog.records)


def test_ambiguous_band_codes_use_count_heuristic_and_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    descriptions = ("B1", "B2", "B3", "B4", "B13")
    path = _write(tmp_path / "ambiguous.tif", ["other"] * len(descriptions), descriptions)

    with caplog.at_level(logging.WARNING, logger="dta.dti.raster"):
        roles = _roles(path)

    assert roles == _roles_from_count(len(descriptions))
    assert any("band count" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
