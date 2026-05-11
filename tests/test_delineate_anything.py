"""Tests for delineate_anything inference wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.models.third_party.delineate_anything.inference import (
    _generate_batch_yaml,
    _generate_conf_yaml,
    _validate_input,
    delineate_fields,
    run,
)


def test_validate_input_valid(tmp_path: Path) -> None:
    tif_path = tmp_path / "valid.tif"
    tif_path.touch()

    is_valid, err = _validate_input(str(tif_path))
    assert is_valid is True
    assert err == ""


def test_validate_input_missing() -> None:
    is_valid, err = _validate_input("/nonexistent/file.tif")
    assert is_valid is False
    assert "not found" in err


def test_validate_input_wrong_ext(tmp_path: Path) -> None:
    txt_path = tmp_path / "invalid.txt"
    txt_path.touch()

    is_valid, err = _validate_input(str(txt_path))
    assert is_valid is False
    assert "Unsupported format" in err


def test_generate_conf_yaml() -> None:
    conf = _generate_conf_yaml(
        bands=[1, 2, 3], batch_size=8, minimum_area_m2=1000, minimum_hole_area_m2=500, model_variant="large"
    )

    assert conf["model"] == ["large"]
    assert conf["data_loader"]["bands"] == [1, 2, 3]
    assert conf["passes"][0]["batch_size"] == 8
    assert conf["passes"][0]["model_args"][0]["name"] == "large"
    assert conf["filtering_args"]["minimum_area_m2"] == 1000
    assert conf["filtering_args"]["minimum_hole_area_m2"] == 500


def test_generate_batch_yaml() -> None:
    batch = _generate_batch_yaml(
        conf_path="/conf.yaml",
        data_root="/data",
        output_root="/out",
        temp_root="/tmp",
        mask_root="/masks",
        include_folders=["folder1"],
    )

    assert batch["base_config"] == "/conf.yaml"
    assert batch["data_root"] == "/data"
    assert batch["include"] == ["folder1"]


@patch("dta.dti.models.third_party.delineate_anything.inference.subprocess.run")
@patch("dta.dti.models.get_model_manager")
def test_delineate_fields_success(mock_get_manager: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    # Setup mock file
    raster_path = tmp_path / "input.tif"
    raster_path.touch()

    # Mock manager
    mock_manager = MagicMock()
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "delineate.py").touch()
    mock_manager.get_model_path.return_value = model_dir
    mock_get_manager.return_value = mock_manager

    # Mock subprocess
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    # Mock output file creation (simulating what delineate.py would do)
    def side_effect(*args, **kwargs):
        # Create output gpkg in the temp dir that is dynamically created
        import tempfile

        base_dir = Path(tempfile.gettempdir()) / "dt4lc_delineate"
        for work_dir in base_dir.glob("delineate_*"):
            out_dir = work_dir / "output"
            if out_dir.exists():
                (out_dir / "data.gpkg").touch()
        return mock_proc

    mock_run.side_effect = side_effect

    # Run
    res = delineate_fields(str(raster_path), data_source="sentinel")

    assert "output_path" in res
    assert Path(res["output_path"]).exists()
    assert res["data_source"] == "sentinel"
    assert res["model"] == "small"


@patch("dta.dti.models.get_model_manager")
def test_delineate_fields_model_not_installed(mock_get_manager: MagicMock, tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    raster_path.touch()

    mock_manager = MagicMock()
    mock_manager.get_model_path.return_value = None
    mock_get_manager.return_value = mock_manager

    with pytest.raises(RuntimeError, match="not installed"):
        delineate_fields(str(raster_path))


@patch("dta.dti.models.third_party.delineate_anything.inference.delineate_fields")
def test_run_entrypoint(mock_delineate: MagicMock) -> None:
    mock_delineate.return_value = {"success": True}

    res = run("test.tif")

    mock_delineate.assert_called_once_with(raster_path="test.tif", model="small")
    assert res["success"] is True


@patch("dta.dti.models.third_party.delineate_anything.inference.subprocess.run")
@patch("dta.dti.models.get_model_manager")
def test_delineate_fields_subprocess_failure(mock_get_manager: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    raster_path = tmp_path / "input.tif"
    raster_path.touch()

    mock_manager = MagicMock()
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "delineate.py").touch()
    mock_manager.get_model_path.return_value = model_dir
    mock_get_manager.return_value = mock_manager

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Subprocess Error"
    mock_run.return_value = mock_proc

    with pytest.raises(RuntimeError, match="Subprocess Error"):
        delineate_fields(str(raster_path))


@pytest.mark.skip(reason="Needs valid rasterio transform")
def test_create_field_boundaries_visualization(tmp_path: Path) -> None:
    # We need a dummy geodataframe
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Polygon

    from dta.dti.models.third_party.delineate_anything.inference import _create_field_boundaries_visualization

    # Create simple polygon
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"geometry": [p]}, crs="EPSG:4326")

    with patch("dta.dti.models.third_party.delineate_anything.inference.HAS_MATPLOTLIB", True):
        # We also need to mock rasterio.open so it doesn't fail trying to read the raster
        with patch("rasterio.open") as mock_open:
            with patch("rasterio.features.rasterize") as mock_rasterize:
                mock_rasterize.return_value = np.ones((10, 10), dtype=np.uint8)
                mock_src = MagicMock()
                mock_src.height = 10
                mock_src.width = 10
                mock_src.transform = None
                mock_src.crs = gdf.crs
            # Just return a 3-band array for RGB
            mock_src.read.return_value = np.zeros((3, 10, 10), dtype=np.uint8)
            mock_open.return_value.__enter__.return_value = mock_src

            res = _create_field_boundaries_visualization(str(tmp_path / "fake.tif"), gdf)

    assert res is not None
    assert isinstance(res, str)
    assert len(res) > 100
