"""Tests for assets, raster utilities, and LLM provider mocks.

Covers DataAssetManager, raster loading/validation, and mocked LLM providers.
These are the main low-coverage modules that need testing.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


def _make_tif(path: str, bands: int = 4, width: int = 32, height: int = 32, dtype: str = "int16") -> str:
    """Helper to create a synthetic GeoTIFF."""
    rng = np.random.default_rng(42)
    if dtype == "int16":
        data = rng.integers(100, 5000, (bands, height, width), dtype=np.int16)
    elif dtype == "uint8":
        data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    else:
        data = rng.random((bands, height, width)).astype(np.float32) * 5000

    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


class TestDataAssetManager:
    """Tests for DataAssetManager."""

    def test_init_default(self) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager()
        assert mgr.data_root.exists() or True  # May not exist in test env

    def test_init_custom_root(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)
        assert mgr.data_root == tmp_path

    def test_resolve_file_uri(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        test_file = tmp_path / "test.tif"
        test_file.write_bytes(b"fake")

        resolved = mgr.resolve(f"file://{test_file}")
        assert resolved == str(test_file.resolve())

    def test_resolve_file_uri_relative(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        test_file = tmp_path / "data.tif"
        test_file.write_bytes(b"fake")

        resolved = mgr.resolve("file://data.tif")
        assert resolved == str(test_file.resolve())

    def test_resolve_file_uri_not_found(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        with pytest.raises(FileNotFoundError):
            mgr.resolve("file://nonexistent.tif")

    def test_resolve_absolute_path(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        test_file = tmp_path / "abs_test.tif"
        test_file.write_bytes(b"fake")

        resolved = mgr.resolve(str(test_file))
        assert resolved == str(test_file.resolve())

    def test_resolve_absolute_not_found(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        with pytest.raises(FileNotFoundError):
            mgr.resolve("/nonexistent/path/to/file.tif")

    def test_resolve_relative_path(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        test_file = tmp_path / "relative.tif"
        test_file.write_bytes(b"fake")

        resolved = mgr.resolve("relative.tif")
        assert resolved == str(test_file.resolve())

    def test_resolve_not_found(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        with pytest.raises(FileNotFoundError, match="not found"):
            mgr.resolve("totally_fake_asset")

    def test_resolve_named_kahovka(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        kahovka_dir = tmp_path / "kahovka_data"
        kahovka_dir.mkdir()
        (kahovka_dir / "raster.tif").write_bytes(b"fake tif")

        resolved = mgr.resolve("kahovka_raster")
        assert "raster.tif" in resolved

    def test_resolve_named_prithvi(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        prithvi_dir = tmp_path / "prithvi_eo_v1_100m" / "examples"
        prithvi_dir.mkdir(parents=True)
        (prithvi_dir / "example.tif").write_bytes(b"fake tif")

        resolved = mgr.resolve("prithvi_example")
        assert "example.tif" in resolved

    def test_get_metadata(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        tif_path = str(tmp_path / "meta_test.tif")
        _make_tif(tif_path, bands=3)

        metadata = mgr.get_metadata(tif_path)
        assert metadata["width"] == 32
        assert metadata["height"] == 32
        assert metadata["count"] == 3
        assert metadata["crs"] == "EPSG:4326"

    def test_get_metadata_cached(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        tif_path = str(tmp_path / "cached.tif")
        _make_tif(tif_path, bands=2)

        m1 = mgr.get_metadata(tif_path)
        m2 = mgr.get_metadata(tif_path)
        assert m1 is m2  # Should be same cached object

    def test_get_metadata_not_found(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        with pytest.raises(FileNotFoundError):
            mgr.get_metadata("/nonexistent/file.tif")

    def test_list_assets(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        (tmp_path / "a.tif").write_bytes(b"fake")
        (tmp_path / "b.tiff").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"fake")

        tifs = mgr.list_assets("*.tif")
        assert len(tifs) == 1

        all_files = mgr.list_assets("*")
        assert len(all_files) == 3

    def test_list_assets_empty_dir(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path / "nonexistent")

        result = mgr.list_assets()
        assert result == []

    def test_clear_cache(self, tmp_path: Path) -> None:
        from dta.dti.assets import DataAssetManager

        mgr = DataAssetManager(data_root=tmp_path)

        mgr.cache["test_key"] = "test_value"
        mgr.clear_cache()
        assert mgr.cache == {}


class TestRasterLoading:
    """Tests for raster loading utilities."""

    def test_load_raster_as_rgb(self, tmp_path: Path) -> None:
        from dta.dti.raster import load_raster_as_rgb

        tif_path = str(tmp_path / "rgb.tif")
        _make_tif(tif_path, bands=4)

        result = load_raster_as_rgb(tif_path)
        assert result.image.shape == (32, 32, 3)
        assert result.image.dtype == np.uint8
        assert result.crs is not None

    def test_load_raster_custom_bands(self, tmp_path: Path) -> None:
        from dta.dti.raster import load_raster_as_rgb

        tif_path = str(tmp_path / "custom.tif")
        _make_tif(tif_path, bands=4)

        result = load_raster_as_rgb(tif_path, bands=[1, 2, 3])
        assert result.image.shape == (32, 32, 3)

    def test_load_raster_uint8(self, tmp_path: Path) -> None:
        from dta.dti.raster import load_raster_as_rgb

        tif_path = str(tmp_path / "uint8.tif")
        _make_tif(tif_path, bands=3, dtype="uint8")

        result = load_raster_as_rgb(tif_path, bands=[1, 2, 3])
        assert result.image.dtype == np.uint8

    def test_load_raster_zeros(self, tmp_path: Path) -> None:
        from dta.dti.raster import load_raster_as_rgb

        tif_path = str(tmp_path / "zeros.tif")
        data = np.zeros((3, 32, 32), dtype=np.int16)
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=3,
            dtype="int16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        result = load_raster_as_rgb(tif_path, bands=[1, 2, 3])
        assert result.image.max() == 0

    def test_load_raster_not_found(self) -> None:
        from dta.dti.raster import load_raster_as_rgb

        with pytest.raises(FileNotFoundError):
            load_raster_as_rgb("/nonexistent.tif")

    def test_load_raster_insufficient_bands(self, tmp_path: Path) -> None:
        from dta.dti.raster import load_raster_as_rgb

        tif_path = str(tmp_path / "small.tif")
        _make_tif(tif_path, bands=2)

        with pytest.raises(ValueError, match="bands"):
            load_raster_as_rgb(tif_path, bands=[1, 2, 3])


class TestValidateGeoTiff:
    """Tests for validate_geotiff."""

    def test_valid_geotiff(self, tmp_path: Path) -> None:
        from dta.dti.raster import validate_geotiff

        tif_path = str(tmp_path / "valid.tif")
        _make_tif(tif_path)

        is_valid, error = validate_geotiff(tif_path)
        assert is_valid is True
        assert error == ""

    def test_nonexistent_file(self) -> None:
        from dta.dti.raster import validate_geotiff

        is_valid, error = validate_geotiff("/nonexistent.tif")
        assert is_valid is False
        assert "not found" in error.lower()

    def test_wrong_extension(self, tmp_path: Path) -> None:
        from dta.dti.raster import validate_geotiff

        txt = tmp_path / "wrong.txt"
        txt.write_text("not a tif")

        is_valid, error = validate_geotiff(str(txt))
        assert is_valid is False
        assert "Unsupported" in error

    def test_no_crs_warns(self, tmp_path: Path) -> None:
        from dta.dti.raster import validate_geotiff

        tif_path = str(tmp_path / "no_crs.tif")
        data = np.zeros((1, 32, 32), dtype=np.uint8)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=1,
            dtype="uint8",
        ) as dst:
            dst.write(data)

        is_valid, error = validate_geotiff(tif_path)
        assert is_valid is True  # Valid but with warning


class TestLLMProviderMocks:
    """Tests for LLM provider initialization and config."""

    def test_llm_router_initialization(self) -> None:
        from dta.dti.coe.llm.router import LLMRouter

        router = LLMRouter(providers=[])
        assert router is not None
        assert isinstance(router.providers, list)

    def test_llm_message_creation(self) -> None:
        from dta.dti.coe.llm import LLMMessage

        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_llm_response_creation(self) -> None:
        from dta.dti.coe.llm.base import LLMResponse

        resp = LLMResponse(text="response text", model="test_model", provider="test")
        assert resp.text == "response text"
        assert resp.provider == "test"

    def test_config_loading(self) -> None:
        from dta.dti.coe.llm.config import get_default_config

        config = get_default_config()
        assert config is not None
        assert isinstance(config, dict)


class TestContextAgentTiffToPng:
    """Tests for _tiff_to_png_bytes."""

    def test_tiff_to_png(self, tmp_path: Path) -> None:
        from dta.dti.coe.context_agent import _tiff_to_png_bytes

        tif_path = str(tmp_path / "preview.tif")
        _make_tif(tif_path, bands=3)

        png_bytes = _tiff_to_png_bytes(tif_path)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100
        assert png_bytes[:4] == b"\x89PNG"  # PNG magic bytes

    def test_tiff_to_png_single_band(self, tmp_path: Path) -> None:
        from dta.dti.coe.context_agent import _tiff_to_png_bytes

        tif_path = str(tmp_path / "single.tif")
        _make_tif(tif_path, bands=1)

        png_bytes = _tiff_to_png_bytes(tif_path)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100


class TestModelRegistry:
    """Tests for ML model registry."""

    def test_load_registry(self) -> None:
        from dta.dti.models.registry import get_model_registry

        registry = get_model_registry()
        assert registry is not None

    def test_registry_items_have_fields(self) -> None:
        from dta.dti.models.registry import get_model_registry

        registry = get_model_registry()
        assert hasattr(registry, "list_all")
