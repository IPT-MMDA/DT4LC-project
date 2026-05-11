"""Tests for scale_to_hls preprocessor."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from dta.dti.preprocessors.scale_to_hls import run


def _make_tif(path: str, data: np.ndarray) -> str:
    transform = from_bounds(0, 0, 1, 1, data.shape[2], data.shape[1])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_scale_0_1_to_hls(tmp_path: Path) -> None:
    input_file = str(tmp_path / "input.tif")
    out_dir = str(tmp_path / "out")
    Path(out_dir).mkdir()

    data = np.array([[[0.0, 0.5], [1.0, 0.2]]], dtype=np.float32)
    _make_tif(input_file, data)

    out_file = run(input_file, out_dir)
    assert out_file.endswith("hls_scaled_input.tif")

    with rasterio.open(out_file) as src:
        scaled = src.read()
        assert scaled.dtype == np.int16
        assert scaled[0, 0, 0] == 0
        assert scaled[0, 0, 1] == 5000
        assert scaled[0, 1, 0] == 10000


def test_scale_large_values(tmp_path: Path) -> None:
    input_file = str(tmp_path / "large.tif")
    out_dir = str(tmp_path / "out2")
    Path(out_dir).mkdir()

    data = np.array([[[0.0, 40000.0], [-40000.0, 1000.0]]], dtype=np.float32)
    _make_tif(input_file, data)

    out_file = run(input_file, out_dir)

    with rasterio.open(out_file) as src:
        scaled = src.read()
        assert scaled.dtype == np.int16
        assert scaled[0, 0, 1] == 32767
        assert scaled[0, 1, 0] == -32768


def test_already_hls_scale(tmp_path: Path) -> None:
    input_file = str(tmp_path / "hls.tif")
    out_dir = str(tmp_path / "out3")
    Path(out_dir).mkdir()

    data = np.array([[[0.0, 5000.0], [8000.0, 1000.0]]], dtype=np.float32)
    _make_tif(input_file, data)

    out_file = run(input_file, out_dir)

    with rasterio.open(out_file) as src:
        scaled = src.read()
        assert scaled.dtype == np.int16
        assert scaled[0, 0, 1] == 5000


def test_error_fallback(tmp_path: Path) -> None:
    # Pass nonexistent file
    out_file = run("/nonexistent/file.tif", str(tmp_path))
    assert out_file == "/nonexistent/file.tif"
