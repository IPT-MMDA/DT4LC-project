"""Tests for scale_to_hls preprocessor."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from dta.dti.preprocessors.scale_to_hls import HLS_BAND_COUNT, run


def _make_tif(path: str, data: np.ndarray, nodata: float | None = None) -> str:
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
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def test_scale_0_1_to_hls(tmp_path: Path) -> None:
    input_file = str(tmp_path / "input.tif")
    out_dir = str(tmp_path / "out")
    Path(out_dir).mkdir()

    data = np.repeat(np.array([[[0.0, 0.5], [1.0, 0.2]]], dtype=np.float32), HLS_BAND_COUNT, axis=0)
    _make_tif(input_file, data)

    out_file = run(input_file, out_dir)
    assert out_file.endswith("hls_scaled_input.tif")

    with rasterio.open(out_file) as src:
        scaled = src.read()
        assert scaled.dtype == np.int16
        assert scaled[0, 0, 0] == 0
        assert scaled[0, 0, 1] == 5000
        assert scaled[0, 1, 0] == 10000


def test_scale_reflectance_with_negative_pixels(tmp_path: Path) -> None:
    # Surface reflectance has small negative values (atmospheric-correction
    # artefacts, and the synthetic test raster). It is still 0-1 data and must be
    # scaled, not truncated to int16 zeros as "already HLS" (#60).
    input_file = str(tmp_path / "neg.tif")
    out_dir = str(tmp_path / "out_neg")
    Path(out_dir).mkdir()

    data = np.repeat(np.array([[[-0.15, 0.5], [0.6, 0.2]]], dtype=np.float32), HLS_BAND_COUNT, axis=0)
    _make_tif(input_file, data)

    with rasterio.open(run(input_file, out_dir)) as src:
        scaled = src.read()
    assert scaled.dtype == np.int16
    assert scaled[0].tolist() == [[-1500, 5000], [6000, 2000]]


@pytest.mark.parametrize("nodata", [None, float("nan")], ids=["untagged", "nodata_nan"])
def test_nan_pixels_become_hls_fill(tmp_path: Path, nodata: float | None) -> None:
    # NaN pixels (the HLS L30 sample uses NaN, often tagged nodata=nan) must be
    # written as the HLS fill value -9999, not an undefined int16 cast, and valid
    # pixels must be rounded, not truncated (0.12345 -> 1235, -0.0123 -> -123).
    values = [0.12345, 0.56789, -0.0123, -0.00007, 0.0, 1.0, 0.33337, np.nan, 0.98765]
    data = np.repeat(np.array(values, dtype=np.float32).reshape(1, 3, 3), HLS_BAND_COUNT, axis=0)
    data[2, 0, 0] = np.nan  # single-band NaN
    input_file = _make_tif(str(tmp_path / "nan.tif"), data, nodata=nodata)

    with rasterio.open(run(input_file, str(tmp_path))) as src:
        scaled = src.read()
        assert src.nodata == -9999

    assert scaled.dtype == np.int16
    nan = np.isnan(data)
    assert (scaled[nan] == -9999).all()
    expected = [round(float(v) * 10000) for v in data[~nan]]
    assert scaled[~nan].tolist() == expected
    assert scaled[0, 0, 0] == 1235
    assert scaled[0, 0, 2] == -123
    assert scaled[0, 1, 0] == -1


@pytest.mark.parametrize(
    ("dtype", "nodata", "values", "expected"),
    [
        (np.int16, -1, [-1, 5000, 8000, 1000], [-1, 5000, 8000, 1000]),
        (np.float32, -9999.0, [-9999.0, 0.5, 0.12345, 0.2], [-9999, 5000, 1235, 2000]),
    ],
    ids=["int16_nodata_-1", "float32_nodata_-9999"],
)
def test_integer_nodata_carried_through(
    tmp_path: Path, dtype: type, nodata: float, values: list[float], expected: list[int]
) -> None:
    # A source with its own integer nodata keeps it: those pixels keep the value,
    # the profile keeps the tag, and the pixels do not take part in the scaling.
    data = np.repeat(np.array(values, dtype=dtype).reshape(1, 2, 2), HLS_BAND_COUNT, axis=0)
    input_file = _make_tif(str(tmp_path / "nodata.tif"), data, nodata=nodata)

    with rasterio.open(run(input_file, str(tmp_path))) as src:
        scaled = src.read()
        assert src.nodata == nodata

    assert scaled.dtype == np.int16
    assert scaled[0].ravel().tolist() == expected


def test_scale_large_values(tmp_path: Path) -> None:
    input_file = str(tmp_path / "large.tif")
    out_dir = str(tmp_path / "out2")
    Path(out_dir).mkdir()

    data = np.repeat(np.array([[[0.0, 40000.0], [-40000.0, 1000.0]]], dtype=np.float32), HLS_BAND_COUNT, axis=0)
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

    data = np.repeat(np.array([[[0.0, 5000.0], [8000.0, 1000.0]]], dtype=np.float32), HLS_BAND_COUNT, axis=0)
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


@pytest.mark.parametrize("count", [5, 7])
def test_wrong_band_count_raises(tmp_path: Path, count: int) -> None:
    # A 7-band Landsat stack (or 5-band HLS subset) must fail with a clear
    # band-count error here, not a numpy broadcast error inside Prithvi (#60).
    input_file = _make_tif(str(tmp_path / "bands.tif"), np.full((count, 2, 2), 0.1, dtype=np.float32))

    with pytest.raises(ValueError, match=f"expected 6 bands .*, got {count} in"):
        run(input_file, str(tmp_path))
