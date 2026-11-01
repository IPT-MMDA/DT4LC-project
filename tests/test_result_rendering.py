"""Result rendering: styles in GeoTIFF tags, bare map PNGs and class legends.

Tiles and bare PNGs share :func:`dta.dti.artifact_files.colourize`: a change raster
goes lost -> no change -> gained in the map legend colours.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.dti import artifact_files

LOST, NO_CHANGE, GAINED = (140, 74, 47), (196, 202, 214), (45, 95, 158)  # #8c4a2f, #c4cad6, #2d5f9e


# --- Tags ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "style",
    [
        artifact_files.change_style(0.37),
        artifact_files.index_style([(0.6, 0.4, 0.2), (0.0, 0.2, 0.6)], -0.5, 0.5),
        artifact_files.class_style({1: "#0077be", 6: "#2e7d32"}),
    ],
)
def test_style_round_trips_through_geotiff_tags(tmp_path: Path, style: dict) -> None:
    tif = tmp_path / "x.tif"
    artifact_files.write_geotiff(tif, np.zeros((4, 4)), "EPSG:32636", from_origin(0, 0, 30, 30), style)
    with rasterio.open(tif) as src:
        tags = src.tags()
    assert tags["DT4LC_KIND"] == style["kind"]
    assert artifact_files.read_style(tags) == style


def test_untagged_or_malformed_tags_have_no_style() -> None:
    assert artifact_files.read_style({}) is None
    assert artifact_files.read_style({"DT4LC_KIND": "change", "DT4LC_VMIN": "x"}) is None


def test_change_detection_tags_change_and_index_rasters(synthetic_raster_pair: tuple[str, str]) -> None:
    from dta.dti.algorithms import change_detection

    result = change_detection.run(*synthetic_raster_pair, "ndwi")

    with rasterio.open(result["rasters"]["change"]["path"]) as src:
        assert artifact_files.read_style(src.tags()) == artifact_files.change_style(0.5)
    config = change_detection.INDEX_CONFIG["ndwi"]
    with rasterio.open(result["rasters"]["index_after"]["path"]) as src:
        assert artifact_files.read_style(src.tags()) == artifact_files.index_style(
            config["index_colors"], config["vmin"], config["vmax"]
        )


# --- Bare map PNGs ---------------------------------------------------------------


def _scene(path: Path, height: int = 40, width: int = 100) -> str:
    """A 7-band Landsat-like float scene in EPSG:32636 at 30 m: water on the left, vegetation on the right,
    a NaN block in the top-left corner."""
    green = np.full((height, width), 0.15, np.float32)
    red = np.full((height, width), 0.10, np.float32)
    nir = np.where(np.arange(width) < width // 2, 0.05, 0.50).astype(np.float32) * np.ones((height, 1), np.float32)
    swir = np.full((height, width), 0.12, np.float32)
    flat = np.full((height, width), 0.1, np.float32)
    bands = np.stack([flat, flat, green, red, nir, swir, flat])
    bands[:, :5, :10] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=7,
        dtype="float32",
        nodata=np.nan,
        crs="EPSG:32636",
        transform=from_origin(600000, 5150000, 30, 30),
    ) as dst:
        dst.write(bands)
    return str(path)


def _open_bare(ref: dict) -> np.ndarray:
    assert ref["kind"] == "png" and set(ref) == {"kind", "path", "url", "stats", "size_bytes"}
    img = Image.open(ref["path"])
    assert img.mode == "RGBA"
    assert img.size == (ref["stats"]["width"], ref["stats"]["height"])
    return np.asarray(img)


def test_band_math_change_writes_bare_pngs(tmp_path: Path) -> None:
    from dta.dti.algorithms import band_math

    before = _scene(tmp_path / "before.tif")
    with rasterio.open(before) as src:
        bands = src.read()
    bands[4, :, :] = np.where(np.arange(100) < 50, 0.05, 0.50)  # NIR
    bands[4, 20:, 50:] = 0.05  # drier bottom-right
    bands[:, :5, :10] = np.nan
    after = tmp_path / "after.tif"
    with rasterio.open(before) as src:
        profile = src.profile
    with rasterio.open(after, "w", **profile) as dst:
        dst.write(bands)

    result = band_math.calculate_change(before, str(after), {"formula": "(B5 - B6) / (B5 + B6)", "name": "NDMI"})
    vis = result["visualizations"]

    assert {"change_map", "index_before", "index_after"} <= set(vis)
    for key in ("change_map", "index_before", "index_after"):
        img = _open_bare(vis[f"{key}_bare"])
        assert img.shape[:2] == (40, 100)  # native aspect, no margin
        assert (img[:5, :10, 3] == 0).all() and (img[5:, 10:, 3] == 255).all()
    change = _open_bare(vis["change_map_bare"])
    assert tuple(change[30, 80, :3]) == LOST  # NDMI fell: drier
    assert tuple(change[10, 80, :3]) == NO_CHANGE
    style = vis["change_map_bare"]["stats"]["style"]
    assert style["kind"] == "change" and style["range"] == result["statistics"]["display_range"]


def test_change_detection_writes_bare_pngs(synthetic_raster_pair: tuple[str, str]) -> None:
    from dta.dti.algorithms import change_detection

    vis = change_detection.run(*synthetic_raster_pair, "ndwi")["visualizations"]

    assert list(vis) == [
        "change_map",
        "ndwi_before",
        "ndwi_after",
        "change_map_bare",
        "ndwi_before_bare",
        "ndwi_after_bare",
    ]
    for key in ("change_map_bare", "ndwi_before_bare", "ndwi_after_bare"):
        assert _open_bare(vis[key]).shape[:2] == (64, 64)
    assert vis["change_map_bare"]["stats"]["style"] == artifact_files.change_style(0.5)


def test_bare_png_is_capped_at_1024_on_the_long_side(tmp_path: Path) -> None:
    data = np.zeros((1000, 2050), dtype=np.float32)
    data[:100, :] = np.nan

    ref = artifact_files.write_bare_png(tmp_path / "big.png", data, artifact_files.change_style(0.5))
    img = _open_bare(ref)

    assert img.shape[1] == 1024 and abs(img.shape[0] - 1000 * 1024 / 2050) <= 1
    assert (img[:40, :, 3] == 0).all() and tuple(img[-1, -1]) == (*NO_CHANGE, 255)


# --- Class legends ---------------------------------------------------------------


def test_lulc_class_statistics_carry_colour_and_area(tmp_path: Path) -> None:
    from dta.dti.algorithms import lulc_classifier

    result = lulc_classifier.run(_scene(tmp_path / "scene.tif"))
    classes = result["class_statistics"]
    valid = result["statistics"]["valid_pixels"]

    assert valid == 40 * 100 - 5 * 10
    for name, c in classes.items():
        assert {"id", "name", "colour", "pixels", "percent", "area_km2"} <= set(c)
        assert c["name"] == name and c["colour"] == lulc_classifier.LULC_CLASSES[c["id"]]["color"]
        assert (c["class_id"], c["percentage"]) == (c["id"], c["percent"])  # the old keys keep their meaning
    assert sum(c["pixels"] for c in classes.values()) == valid
    assert sum(c["area_km2"] for c in classes.values()) == pytest.approx(valid * 0.0009, abs=1e-3)
    assert classes["Water"]["pixels"] > 0 and classes["Dense Vegetation"]["pixels"] > 0

    bare = _open_bare(result["visualizations"]["classification_map_bare"])
    assert bare.shape[:2] == (40, 100) and (bare[:5, :10, 3] == 0).all()
    assert tuple(bare[20, 20, :3]) == (0x00, 0x77, 0xBE)  # water
    with rasterio.open(result["rasters"]["classification"]["path"]) as src:
        assert artifact_files.read_style(src.tags())["kind"] == "class"


def test_snow_class_statistics_carry_colour_and_area(tmp_path: Path, synthetic_raster_path: str) -> None:
    from dta.dti.algorithms import snow_classifier

    result = snow_classifier.run(_scene(tmp_path / "scene.tif"))
    classes = result["class_statistics"]
    valid = result["statistics"]["valid_pixels"]

    assert set(classes) == {"No Snow", "Snow"}
    assert [c["colour"] for c in classes.values()] == ["#667f4c", "#ffcc66"]
    assert sum(c["pixels"] for c in classes.values()) == valid
    assert sum(c["area_km2"] for c in classes.values()) == pytest.approx(valid * 0.0009, abs=1e-3)
    assert _open_bare(result["visualizations"]["snow_classification_bare"]).shape[:2] == (40, 100)

    # Unprojected CRS: no area.
    unprojected = snow_classifier.run(synthetic_raster_path)["class_statistics"]
    assert all(c["area_km2"] is None for c in unprojected.values())
