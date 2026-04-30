"""Tests for component registry functionality.

Tests the registry loading, component lookup, and registry item validation.
"""

import pytest

from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Registry, RegistryItem, Runner


class TestRegistryLoading:
    """Tests for registry loading."""

    def test_registry_loads_successfully(self) -> None:
        """Test that registry loads from YAML file."""
        registry = load_registry()

        assert registry.version == "0.1"
        assert len(registry.types) > 0
        assert len(registry.instances) > 0

    def test_registry_contains_required_types(self) -> None:
        """Test that registry contains all expected types."""
        registry = load_registry()

        expected_types = ["RasterPath", "NDVIMap", "Statistics", "ChangeMap", "Reconstruction"]
        for t in expected_types:
            assert t in registry.types, f"Missing type: {t}"

    def test_registry_contains_core_algorithms(self) -> None:
        """Test that registry contains core algorithms."""
        registry = load_registry()

        # Find algorithm instances
        algorithms = [i for i in registry.instances if i.kind == "algorithm"]
        algorithm_ids = [a.id for a in algorithms]

        assert "algorithms/ndvi" in algorithm_ids
        assert "algorithms/statistics" in algorithm_ids
        assert "algorithms/change-detection" in algorithm_ids

    def test_registry_contains_input_components(self) -> None:
        """Test that registry contains input components."""
        registry = load_registry()

        inputs = [i for i in registry.instances if i.kind == "input"]
        input_ids = [i.id for i in inputs]

        assert "input/file" in input_ids
        assert "input/file-before" in input_ids
        assert "input/file-after" in input_ids

    def test_registry_contains_models(self) -> None:
        """Test that registry contains ML models."""
        registry = load_registry()

        models = [i for i in registry.instances if i.kind == "model"]
        model_ids = [m.id for m in models]

        assert "models/prithvi-reconstruction" in model_ids
        assert "models/delineate-anything" in model_ids


class TestRegistryItemLookup:
    """Tests for registry item lookup."""

    def test_get_item_by_id(self) -> None:
        """Test getting item by ID."""
        registry = load_registry()

        item = get_item(registry, "algorithms/ndvi")

        assert item is not None
        assert item.id == "algorithms/ndvi"
        assert item.kind == "algorithm"
        assert item.runner.type == "python"

    def test_get_item_not_found(self) -> None:
        """Test getting non-existent item raises error."""
        registry = load_registry()

        with pytest.raises(KeyError, match="not found"):
            get_item(registry, "nonexistent/component")

    def test_get_item_returns_correct_structure(self) -> None:
        """Test that retrieved item has correct structure."""
        registry = load_registry()

        item = get_item(registry, "algorithms/ndvi")

        assert isinstance(item.keywords, list)
        assert isinstance(item.inputs, list)
        assert isinstance(item.outputs, list)
        assert isinstance(item.runner, Runner)


class TestRegistryItemValidation:
    """Tests for registry item structure."""

    def test_all_items_have_required_fields(self) -> None:
        """Test that all registry items have required fields."""
        registry = load_registry()

        for item in registry.instances:
            assert item.id, "Item missing id"
            assert item.kind, f"Item {item.id} missing kind"
            # Hosted models (with integration) don't require a runner
            if not item.integration:
                assert item.runner, f"Item {item.id} missing runner"
            assert isinstance(item.keywords, list), f"Item {item.id} keywords not a list"
            assert isinstance(item.inputs, list), f"Item {item.id} inputs not a list"
            assert isinstance(item.outputs, list), f"Item {item.id} outputs not a list"

    def test_python_runners_have_entrypoints(self) -> None:
        """Test that Python runners have entrypoints."""
        registry = load_registry()

        for item in registry.instances:
            if item.runner and item.runner.type == "python":
                assert item.runner.entrypoint, f"Item {item.id} missing entrypoint"

    def test_algorithm_entrypoints_exist(self) -> None:
        """Verify algorithm entrypoints resolve — either as importable modules
        (dotted path) or as on-disk files (filesystem path).
        """
        import importlib

        from dta.config import ROOT_DIR

        registry = load_registry()

        for item in registry.instances:
            if item.kind != "algorithm" or not (item.runner and item.runner.entrypoint):
                continue
            ep = item.runner.entrypoint
            if ep.endswith(".py") or "/" in ep or "\\" in ep:
                assert (ROOT_DIR / ep).exists(), f"Entrypoint file not found: {ep}"
            else:
                # Dotted module path — should be importable
                importlib.import_module(ep)


class TestMockRegistry:
    """Tests with mock registry for isolated testing."""

    @pytest.fixture
    def mock_registry(self) -> Registry:
        """Create mock registry for testing."""
        return Registry(
            version="1.0",
            types=["Raster", "NDVIMap", "Statistics"],
            instances=[
                RegistryItem(
                    id="input/test",
                    kind="input",
                    runner=Runner(type="passthrough"),
                    keywords=["test"],
                    inputs=[],
                    outputs=["Raster"],
                ),
                RegistryItem(
                    id="algorithms/test",
                    kind="algorithm",
                    runner=Runner(type="python", entrypoint="test.py"),
                    keywords=["test", "algo"],
                    inputs=["Raster"],
                    outputs=["NDVIMap"],
                ),
            ],
        )

    def test_mock_registry_get_item(self, mock_registry: Registry) -> None:
        """Test getting item from mock registry."""
        item = get_item(mock_registry, "algorithms/test")

        assert item.id == "algorithms/test"
        assert "algo" in item.keywords
