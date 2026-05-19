from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# Resolve repo root via your config helper if available; else fall back.
try:
    from dta.config import ROOT_DIR

    ROOT = Path(ROOT_DIR)
except Exception:
    # repo root assumed = parent of tests/
    ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


# --- Synthetic raster fixtures for demo-flow smoke tests ---------------------
#
# resources/kahovka_data/ is empty in the repo (vendored samples not committed).
# These fixtures generate small Landsat-8-like 7-band GeoTIFFs on disk so
# Layer-A demo tests can exercise the full algorithm + executor pipeline in CI
# without depending on outside data.
#
# Layout: 7 bands, Landsat 8 / 9 surface-reflectance convention.
#   B1 coastal, B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1, B7 SWIR2
# This matches the `src.count >= 7` branch in algorithms/ndvi.py and friends,
# which read band 4 as red and band 5 as NIR.


def _write_landsat8_like(path: Path, *, vegetation_strength: float, snow_strength: float) -> None:
    """Write a small 7-band Landsat-8-like GeoTIFF with controllable signals.

    Args:
        path: Destination path.
        vegetation_strength: Higher → larger NIR-Red gap → higher mean NDVI.
            0.0 = no vegetation, 0.4 = healthy.
        snow_strength: Higher → larger Green-SWIR gap → higher mean NDSI.
            0.0 = no snow, 0.5 = bright snow.
    """
    height = width = 64
    rng = np.random.default_rng(seed=42)

    # Realistic-ish surface-reflectance values in [0, 1].
    base_red = 0.10 + 0.02 * rng.standard_normal((height, width)).astype(np.float32)
    base_nir = base_red + vegetation_strength + 0.02 * rng.standard_normal(base_red.shape).astype(np.float32)
    base_green = 0.15 + 0.02 * rng.standard_normal((height, width)).astype(np.float32)
    base_swir = base_green - snow_strength + 0.02 * rng.standard_normal(base_green.shape).astype(np.float32)

    bands = np.stack(
        [
            np.full((height, width), 0.08, dtype=np.float32),  # B1 coastal
            np.full((height, width), 0.12, dtype=np.float32),  # B2 blue
            base_green.astype(np.float32),  # B3 green
            base_red.astype(np.float32),  # B4 red
            base_nir.astype(np.float32),  # B5 NIR
            base_swir.astype(np.float32),  # B6 SWIR1
            np.full((height, width), 0.05, dtype=np.float32),  # B7 SWIR2
        ]
    )

    transform = from_origin(west=30.0, north=47.0, xsize=0.0001, ysize=0.0001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=7,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(bands)


@pytest.fixture(scope="session")
def synthetic_raster_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Single 7-band Landsat-8-like GeoTIFF with vegetation + some snow.

    NDVI and NDSI both compute to clearly positive means.
    """
    out = tmp_path_factory.mktemp("rasters") / "synthetic_landsat.tif"
    _write_landsat8_like(out, vegetation_strength=0.40, snow_strength=0.20)
    return str(out)


@pytest.fixture(scope="session")
def synthetic_raster_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Pair of 7-band GeoTIFFs simulating vegetation loss between two dates.

    "Before" has healthy vegetation and ice; "after" has both reduced.
    Change-detection tests can assert on the loss direction without
    pinning exact pixel values.
    """
    out_dir = tmp_path_factory.mktemp("raster_pair")
    before = out_dir / "before.tif"
    after = out_dir / "after.tif"
    _write_landsat8_like(before, vegetation_strength=0.45, snow_strength=0.30)
    _write_landsat8_like(after, vegetation_strength=0.10, snow_strength=0.05)
    return str(before), str(after)
