"""Tests for miscellaneous DTA modules.

Tests context_agent, plan_validator, planner, data_source,
and other utility modules that need coverage.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.schemas import (
    Attachment,
    ChatRequest,
    ContextUnderstanding,
    ExecutionPlan,
    PlanStep,
    RegistryItem,
    Runner,
)


class TestPlanValidator:
    """Tests for plan_validator.validate."""

    def test_valid_plan_passes(self) -> None:
        from dta.dti.coe.plan_validator import validate
        from dta.dti.registry import load_registry

        reg = load_registry()
        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )
        result = validate(plan, reg)
        assert result == plan

    def test_invalid_dependency_raises(self) -> None:
        from dta.dti.coe.plan_validator import PlanError, validate
        from dta.dti.registry import load_registry

        reg = load_registry()
        # NDVI requires RasterPath but input/file is not before it
        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="algorithms/ndvi", binds={})],
            outputs=["NDVIMap"],
        )

        with pytest.raises(PlanError, match="requires"):
            validate(plan, reg)

    def test_missing_entrypoint_raises(self) -> None:
        from dta.dti.coe.plan_validator import PlanError, validate
        from dta.dti.schemas import Registry

        # Create a registry with a Python runner missing entrypoint
        item = RegistryItem(
            id="test/bad",
            kind="algorithm",
            runner=Runner(type="python", entrypoint=None),
            inputs=[],
            outputs=["TestOut"],
        )
        reg = Registry(version="1.0", types=["TestOut"], instances=[item])
        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="test/bad", binds={})],
            outputs=["TestOut"],
        )

        with pytest.raises(PlanError, match="missing python entrypoint"):
            validate(plan, reg)


class TestContextAgent:
    """Tests for context_agent.analyze."""

    def test_analyze_with_mock_llm(self) -> None:
        from dta.dti.coe.context_agent import analyze

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "goal": "calculate ndvi",
                "desired_outputs": ["NDVIMap"],
                "required_inputs": ["RasterPath"],
                "hints": {"keywords": ["ndvi", "vegetation"]},
            }
        )

        mock_router = MagicMock()
        mock_router.generate.return_value = mock_response

        with patch("dta.dti.coe.context_agent.get_llm_router", return_value=mock_router):
            req = ChatRequest(prompt="calculate ndvi", attachments=[])
            result = analyze(req, registry_types=["NDVIMap", "RasterPath"])

        assert isinstance(result, ContextUnderstanding)
        assert result.goal == "calculate ndvi"
        assert "NDVIMap" in result.desired_outputs

    def test_analyze_fallback_on_bad_json(self) -> None:
        from dta.dti.coe.context_agent import analyze

        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all"

        mock_router = MagicMock()
        mock_router.generate.return_value = mock_response

        with patch("dta.dti.coe.context_agent.get_llm_router", return_value=mock_router):
            req = ChatRequest(prompt="calculate ndvi", attachments=[])
            result = analyze(req, registry_types=["NDVIMap"])

        assert isinstance(result, ContextUnderstanding)
        assert result.goal == "calculate ndvi"  # Falls back to prompt as goal

    def test_image_part_returns_none(self) -> None:
        from dta.dti.coe.context_agent import _image_part

        att = Attachment(id="t", filename="t.tif", mime_type="image/tiff", path="/tmp/t.tif")
        assert _image_part(att) is None


class TestPlanner:
    """Tests for planner.plan."""

    def test_plan_ndvi(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="calculate ndvi",
            desired_outputs=["NDVIMap"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["ndvi"]},
        )

        result = plan(ctx, reg)
        assert isinstance(result, ExecutionPlan)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps
        assert "algorithms/ndvi" in steps

    def test_plan_statistics(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="extract statistics",
            desired_outputs=["Statistics"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["statistics"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps
        assert "algorithms/statistics" in steps

    def test_plan_change_detection(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="detect changes",
            desired_outputs=["ChangeMap"],
            required_inputs=["RasterPathBefore", "RasterPathAfter"],
            hints={"keywords": ["change", "detection"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert any("change" in s for s in steps)

    def test_plan_field_boundaries(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="detect field boundaries",
            desired_outputs=["FieldBoundaries"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["field", "boundaries"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps

    def test_plan_ndwi(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="detect water",
            desired_outputs=["NDWIMap"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["water", "ndwi"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps
        assert "algorithms/ndwi" in steps

    def test_plan_ndsi(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="detect snow",
            desired_outputs=["NDSIMap"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["snow", "ndsi"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps
        assert "algorithms/ndsi" in steps

    def test_plan_lulc(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="classify land cover",
            desired_outputs=["LULCMap"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["lulc", "land cover"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps

    def test_plan_snow_classifier(self) -> None:
        from dta.dti.coe.planner import plan
        from dta.dti.registry import load_registry

        reg = load_registry()
        ctx = ContextUnderstanding(
            goal="classify snow",
            desired_outputs=["SnowClassification"],
            required_inputs=["RasterPath"],
            hints={"keywords": ["snow", "classifier"]},
        )

        result = plan(ctx, reg)
        steps = [s.uses for s in result.steps]
        assert "input/file" in steps


class TestDataSource:
    """Tests for data_source utility."""

    def test_detect_data_source_nonexistent(self) -> None:
        from dta.dti.utils.data_source import detect_data_source

        source = detect_data_source("/nonexistent/file.tif")
        assert source == "sentinel"  # Default fallback

    def test_detect_data_source_synthetic(self, tmp_path: Path) -> None:
        """Test detection with a synthetic GeoTIFF."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from dta.dti.utils.data_source import detect_data_source

        path = str(tmp_path / "test.tif")
        data = np.zeros((1, 32, 32), dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 32, 32)  # ~3400m/px in degrees
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        source = detect_data_source(path)
        assert source in ("sentinel", "planet", "maxar")

    def test_get_filtering_thresholds(self) -> None:
        from dta.dti.utils.data_source import get_filtering_thresholds

        sentinel = get_filtering_thresholds("sentinel")
        assert sentinel["minimum_area_m2"] == 2500

        planet = get_filtering_thresholds("planet")
        assert planet["minimum_area_m2"] == 1000

        maxar = get_filtering_thresholds("maxar")
        assert maxar["minimum_area_m2"] == 500

        unknown = get_filtering_thresholds("unknown")
        assert unknown == sentinel  # Default to sentinel


class TestLayerMetadataStore:
    """Tests for layer_metadata_store."""

    def test_save_and_get(self, tmp_path: Path) -> None:
        with patch("server.layer_metadata_store.METADATA_DIR", tmp_path):
            from server.layer_metadata_store import get_layer_metadata, save_layer_metadata

            save_layer_metadata("test-layer", {"name": "Test", "type": "ndvi"})

            result = get_layer_metadata("test-layer")
            assert result is not None
            assert result["name"] == "Test"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        with patch("server.layer_metadata_store.METADATA_DIR", tmp_path):
            from server.layer_metadata_store import get_layer_metadata

            result = get_layer_metadata("nonexistent")
            assert result is None

    def test_delete(self, tmp_path: Path) -> None:
        with patch("server.layer_metadata_store.METADATA_DIR", tmp_path):
            from server.layer_metadata_store import delete_layer_metadata, save_layer_metadata

            save_layer_metadata("to-delete", {"name": "Delete Me"})
            assert delete_layer_metadata("to-delete") is True
            assert delete_layer_metadata("to-delete") is False  # Already deleted

    def test_list_all(self, tmp_path: Path) -> None:
        with patch("server.layer_metadata_store.METADATA_DIR", tmp_path):
            from server.layer_metadata_store import list_all_layers, save_layer_metadata

            save_layer_metadata("layer-1", {"name": "Layer 1"})
            save_layer_metadata("layer-2", {"name": "Layer 2"})

            layers = list_all_layers()
            assert len(layers) == 2


class TestJobsHelpers:
    """Tests for jobs.py helper functions."""

    def test_make_json_serializable_numpy(self) -> None:
        import numpy as np

        from server.jobs import _make_json_serializable

        arr = np.array([1, 2, 3])
        result = _make_json_serializable(arr)
        assert result == [1, 2, 3]

    def test_make_json_serializable_nan(self) -> None:
        import numpy as np

        from server.jobs import _make_json_serializable

        assert _make_json_serializable(float("nan")) is None
        assert _make_json_serializable(float("inf")) is None
        assert _make_json_serializable(np.float64("nan")) is None
        assert _make_json_serializable(np.int64(42)) == 42.0

    def test_make_json_serializable_nested(self) -> None:
        import numpy as np

        from server.jobs import _make_json_serializable

        data = {"values": [np.float64(1.0), np.int32(2)], "nested": {"arr": np.array([3])}}
        result = _make_json_serializable(data)
        assert result["values"] == [1.0, 2.0]
        assert result["nested"]["arr"] == [3]

    def test_make_json_serializable_passthrough(self) -> None:
        from server.jobs import _make_json_serializable

        assert _make_json_serializable("hello") == "hello"
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable(True) is True
        assert _make_json_serializable(None) is None
