"""Tests for model registry and Prithvi model.

Tests model registration, lookup, and Prithvi model wrapper.
"""

import numpy as np

from dta.dti.models.prithvi import PrithviModel
from dta.dti.models.registry import ModelRegistry, get_model_registry


class TestPrithviModel:
    """Tests for Prithvi model wrapper."""

    def test_initialization(self) -> None:
        """Test Prithvi model initialization."""
        model = PrithviModel()

        assert model.name == "prithvi"
        assert model.version == "v1.0"
        assert "Raster" in model.required_inputs
        assert "Features" in model.outputs
        assert model.is_available()

    def test_prediction(self) -> None:
        """Test Prithvi prediction."""
        model = PrithviModel()

        result = model.predict({"raster_path": "test.tif"})

        assert "features" in result
        assert "embeddings" in result
        assert "model" in result
        assert result["model"] == "prithvi"

    def test_extract_features(self) -> None:
        """Test feature extraction."""
        model = PrithviModel()

        raster = np.random.rand(10, 10, 6)
        features = model.extract_features(raster)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestModelRegistry:
    """Tests for model registry."""

    def test_register_and_get(self) -> None:
        """Test model registration and retrieval."""
        registry = ModelRegistry()

        model = PrithviModel()
        registry.register(model, metadata={"gpu_required": False})

        retrieved = registry.get("prithvi")
        assert retrieved.name == "prithvi"

    def test_list_available(self) -> None:
        """Test listing available models."""
        registry = ModelRegistry()

        model = PrithviModel()
        registry.register(model)

        available = registry.list_available()
        assert len(available) > 0

    def test_check_requirements(self) -> None:
        """Test checking model requirements."""
        registry = ModelRegistry()

        model = PrithviModel()
        registry.register(model)

        reqs = registry.check_requirements("prithvi:v1.0")
        assert reqs["name"] == "prithvi"
        assert reqs["available"] is True

    def test_global_registry(self) -> None:
        """Test global model registry."""
        registry = get_model_registry()

        available = registry.list_available()
        assert any("prithvi" in m for m in available)
