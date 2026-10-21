"""Band-role resolution from band descriptions (#64)."""

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.config import ROOT_DIR
from dta.dti.algorithms import ndvi
from dta.dti.raster import band_indices

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


def test_sentinel2_codes(tmp_path: Path) -> None:
    descriptions = ("B2", "B3", "B4", "B5", "B8", "B11", "B12")
    path = _write(tmp_path / "s2.tif", ["blue", "green", "red", "other", "nir", "swir", "other"], descriptions)

    assert _roles(path) == {"red": 3, "nir": 5, "green": 2, "blue": 1, "swir": 6}


def test_hls_l30_codes_of_kahovka_crop() -> None:
    with rasterio.open(ROOT_DIR / "resources/kahovka_data/hlsl_20230601.tif") as src:
        assert src.descriptions == ("B2", "B3", "B4", "B5", "B6")
        roles = band_indices(src)

    assert roles == {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5}
