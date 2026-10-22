"""Tests for layer metadata store."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from server import layer_metadata_store


@pytest.fixture
def mock_metadata_dir(tmp_path: Path):
    """Fixture to mock METADATA_DIR with a temporary directory."""
    with patch("server.layer_metadata_store.METADATA_DIR", tmp_path):
        yield tmp_path


def test_save_layer_metadata(mock_metadata_dir: Path) -> None:
    layer_id = "test-layer-1"
    metadata = {"id": layer_id, "name": "Test Layer"}

    layer_metadata_store.save_layer_metadata(layer_id, metadata)

    saved_file = mock_metadata_dir / f"{layer_id}.json"
    assert saved_file.exists()

    with open(saved_file) as f:
        loaded = json.load(f)
        assert loaded == metadata


def test_save_layer_metadata_error(mock_metadata_dir: Path) -> None:
    # Force an error by passing a non-serializable object
    with pytest.raises(TypeError):
        layer_metadata_store.save_layer_metadata("test-layer-error", {"obj": object()})


def test_get_layer_metadata_exists(mock_metadata_dir: Path) -> None:
    layer_id = "test-layer-2"
    metadata = {"id": layer_id, "name": "Test Layer 2"}

    layer_metadata_store.save_layer_metadata(layer_id, metadata)

    result = layer_metadata_store.get_layer_metadata(layer_id)
    assert result == metadata


def test_get_layer_metadata_not_found(mock_metadata_dir: Path) -> None:
    result = layer_metadata_store.get_layer_metadata("nonexistent-layer")
    assert result is None


def test_get_layer_metadata_error(mock_metadata_dir: Path) -> None:
    layer_id = "test-layer-error"
    metadata_file = mock_metadata_dir / f"{layer_id}.json"
    metadata_file.write_text("{invalid_json}")

    result = layer_metadata_store.get_layer_metadata(layer_id)
    assert result is None


def test_delete_layer_metadata_exists(mock_metadata_dir: Path) -> None:
    layer_id = "test-layer-delete"
    layer_metadata_store.save_layer_metadata(layer_id, {"id": layer_id})

    assert layer_metadata_store.delete_layer_metadata(layer_id) is True
    assert not (mock_metadata_dir / f"{layer_id}.json").exists()


def test_delete_layer_metadata_not_found(mock_metadata_dir: Path) -> None:
    assert layer_metadata_store.delete_layer_metadata("nonexistent") is False


def test_delete_layer_metadata_error(mock_metadata_dir: Path) -> None:
    layer_id = "test-layer-delete-error"
    metadata_file = mock_metadata_dir / f"{layer_id}.json"
    metadata_file.touch()

    # Mock unlink to raise an exception
    with patch.object(Path, "unlink", side_effect=PermissionError):
        assert layer_metadata_store.delete_layer_metadata(layer_id) is False


def test_list_all_layers(mock_metadata_dir: Path) -> None:
    m1 = {"id": "1", "name": "L1"}
    m2 = {"id": "2", "name": "L2"}

    layer_metadata_store.save_layer_metadata("1", m1)
    layer_metadata_store.save_layer_metadata("2", m2)

    layers = layer_metadata_store.list_all_layers()
    assert len(layers) == 2
    assert m1 in layers
    assert m2 in layers


def test_list_all_layers_with_error(mock_metadata_dir: Path) -> None:
    layer_metadata_store.save_layer_metadata("1", {"id": "1"})

    # Create invalid file
    (mock_metadata_dir / "error.json").write_text("{invalid}")

    layers = layer_metadata_store.list_all_layers()
    assert len(layers) == 1
    assert layers[0]["id"] == "1"


def test_list_all_layers_error(mock_metadata_dir: Path) -> None:
    # Mock glob to raise an exception
    with patch.object(Path, "glob", side_effect=Exception):
        layers = layer_metadata_store.list_all_layers()
        assert layers == []
