from collections.abc import Callable, Iterator
import functools
import itertools
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


@functools.cache
def _no_llm_provider_reason() -> str | None:
    """Return a skip reason if the router has no available provider, else None."""
    from dta.dti.coe.llm.config import create_router_from_env

    router = create_router_from_env()
    if router.get_available_providers():
        return None
    checked = ", ".join(p.name for p in router.providers) or "none"
    return (
        f"no LLM provider available (checked: {checked}); set an API key such as "
        "GEMINI_API_KEY or GROQ_API_KEY, or run Ollama with OLLAMA_MODEL pulled"
    )


@pytest.fixture(autouse=True, scope="session")
def _outputs_in_tmp(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Write job artifact files (GeoTIFF/PNG) under a temp dir instead of resources/.cache/outputs."""
    import dta.dti.artifact_files
    import server.paths

    outputs = tmp_path_factory.mktemp("outputs")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dta.dti.artifact_files, "OUTPUTS_PATH", outputs)
        mp.setattr(server.paths, "OUTPUTS_PATH", outputs)
        yield outputs


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip ``llm``-marked tests, instead of failing them, when no provider is configured."""
    if item.get_closest_marker("llm") and (reason := _no_llm_provider_reason()):
        pytest.skip(reason)


# --- Synthetic raster fixtures for demo-flow smoke tests ---------------------
#
# resources/kahovka_data/ holds two committed HLS L30 crops of the lower Dnipro
# around Kherson, downstream of the Kakhovka dam (hlsl_20230601.tif,
# hlsl_20230609.tif: 768x768 px, 30 m, EPSG:32636, 5 bands B2..B6, int16
# reflectance x10000, nodata -9999) used by the real-data tests. These
# fixtures generate small Landsat-8-like 7-band GeoTIFFs on disk so
# Layer-A demo tests can exercise the full algorithm + executor pipeline with
# controllable signals.
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


@pytest.fixture
def write_scaled_scene(tmp_path: Path) -> Callable[..., str]:
    """Factory for 16x16 5-band constant scenes (``B2,B3,B4,B8,B11``) stored as ``dtype``.

    ``write(dtype, scale, reflectance)`` writes ``reflectance`` (blue, green,
    red, NIR, SWIR1) multiplied by ``scale`` (rounded for integer dtypes) and
    returns the path. The default is the dark scene from the v0.5.1 QA baseline
    (#64): every int16 x10000 value is below 1000. Signed integer scenes carry
    ``nodata=-9999``, like the HLS crops.
    """
    counter = itertools.count()

    def write(dtype: str, scale: float, reflectance: tuple[float, ...] = (0.004, 0.008, 0.005, 0.040, 0.020)) -> str:
        data = np.stack([np.full((16, 16), v * scale, dtype=np.float64) for v in reflectance])
        if np.issubdtype(np.dtype(dtype), np.integer):
            data = np.round(data)
        path = tmp_path / f"scene_{next(counter)}_{dtype}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=5,
            dtype=dtype,
            nodata=-9999 if np.issubdtype(np.dtype(dtype), np.signedinteger) else None,
            crs="EPSG:32636",
            transform=from_origin(0, 480, 30, 30),
        ) as dst:
            dst.write(data.astype(dtype))
            for i, name in enumerate(("B2", "B3", "B4", "B8", "B11"), start=1):
                dst.set_band_description(i, name)
        return str(path)

    return write
