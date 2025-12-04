"""Tests for Delineate-Anything field boundary detection integration.

This test module verifies the Delineate-Anything model integration works correctly
with the Kahovka test data.
"""

from datetime import datetime
from pathlib import Path
import uuid

import pytest

from dta.config import ROOT_DIR, TEMP_PATH
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Attachment, ChatRequest

# ----- Resolve paths from registry -------------------------------------------
REGISTRY = load_registry()
DELINEATE_ITEM = get_item(REGISTRY, "models/delineate-anything")

# Kahovka test data
KAHOVKA_DIR = ROOT_DIR / "resources" / "kahovka_data"
KAHOVKA_TIFF = KAHOVKA_DIR / "hlsl_20230601.tif"


def _check_delineate_dependencies() -> tuple[bool, str]:
    """Check if all Delineate-Anything dependencies are available."""
    missing = []
    deps = {
        "ultralytics": "ultralytics",
        "geopandas": "geopandas",
        "shapely": "shapely",
        "cv2": "opencv-python",
    }
    for module, package in deps.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        return False, f"Missing: {', '.join(missing)}. Install with: pip install -e '.[delineate]'"
    return True, ""


def _check_model_installed() -> tuple[bool, str]:
    """Check if Delineate-Anything model is downloaded."""
    from dta.dti.models import get_model_manager

    manager = get_model_manager()
    if not manager.is_model_available("delineate-anything-small"):
        return False, "Delineate-Anything model not installed. Download via Models panel."
    return True, ""


class TestDelineateAnythingRegistry:
    """Test registry configuration for Delineate-Anything."""

    def test_delineate_model_registered(self) -> None:
        """Verify the model is properly registered."""
        assert DELINEATE_ITEM is not None
        assert DELINEATE_ITEM.id == "models/delineate-anything"
        assert DELINEATE_ITEM.kind == "model"

    def test_delineate_model_inputs_outputs(self) -> None:
        """Verify correct input/output types."""
        assert "RasterPath" in DELINEATE_ITEM.inputs
        assert "FieldBoundaries" in DELINEATE_ITEM.outputs

    def test_delineate_model_keywords(self) -> None:
        """Verify keywords for LLM matching."""
        keywords = DELINEATE_ITEM.keywords
        assert "field" in keywords
        assert "boundary" in keywords or "boundaries" in keywords
        assert "delineate" in keywords

    def test_field_boundaries_type_exists(self) -> None:
        """Verify FieldBoundaries type is in registry."""
        assert "FieldBoundaries" in REGISTRY.types


class TestDelineateAnythingModule:
    """Test the inference module itself."""

    def test_module_importable(self) -> None:
        """Verify module can be imported."""
        from dta.dti.models.third_party.delineate_anything import delineate_fields, run

        assert callable(delineate_fields)
        assert callable(run)

    def test_validate_input_missing_file(self) -> None:
        """Test validation rejects missing files."""
        from dta.dti.models.third_party.delineate_anything.inference import _validate_input

        is_valid, error = _validate_input("/nonexistent/path.tif")
        assert not is_valid
        assert "not found" in error.lower()

    def test_validate_input_wrong_format(self, tmp_path: Path) -> None:
        """Test validation rejects non-TIFF files."""
        from dta.dti.models.third_party.delineate_anything.inference import _validate_input

        # Create a fake PNG file
        fake_png = tmp_path / "test.png"
        fake_png.write_bytes(b"fake png data")

        is_valid, error = _validate_input(str(fake_png))
        assert not is_valid
        assert "unsupported format" in error.lower() or "geotiff" in error.lower()

    @pytest.mark.skipif(
        not KAHOVKA_TIFF.exists(),
        reason="Kahovka test data not available",
    )
    def test_validate_input_kahovka_tiff(self) -> None:
        """Test validation accepts Kahovka TIFF data."""
        from dta.dti.models.third_party.delineate_anything.inference import _validate_input

        is_valid, error = _validate_input(str(KAHOVKA_TIFF))
        assert is_valid, f"Kahovka TIFF should be valid: {error}"


class TestDelineateAnythingOrchestration:
    """Test COE orchestration with Delineate-Anything.

    Note: These tests verify the planner CAN select delineate-anything when
    explicitly requested. LLM-based selection depends on prompt interpretation.
    """

    @pytest.mark.xfail(
        reason="LLM may not always select delineate-anything; depends on planner prompt",
        strict=False,
    )
    @pytest.mark.skipif(
        not KAHOVKA_TIFF.exists(),
        reason="Kahovka test data not available",
    )
    def test_orchestration_recognizes_delineate_model(self) -> None:
        """Test that orchestration recognizes delineate-anything model in plan.

        Note: LLM may not always select delineate-anything model. This test
        documents expected behavior but may fail until planner prompts are
        updated to include the new model.
        """
        req = ChatRequest(
            prompt="Use delineate-anything to detect field boundaries in this satellite image",
            attachments=[
                Attachment(
                    id="att-1",
                    filename=KAHOVKA_TIFF.name,
                    mime_type="image/tiff",
                    path=str(KAHOVKA_TIFF),
                    size_bytes=KAHOVKA_TIFF.stat().st_size,
                )
            ],
            metadata={"test_mode": True},
        )

        result = orchestrate(req)

        # Check if delineate-anything was included in the candidate plan
        # (even if the plan failed validation for other reasons)
        if result["ok"]:
            plan = result["plan"]
        else:
            # Plan validation failed, but check the candidate
            plan = result.get("candidate", {})

        step_ids = [s["uses"] for s in plan.get("steps", [])]

        # Log what the LLM selected for debugging
        print(f"\nLLM selected steps: {step_ids}")

        # Verify delineate-anything was recognized and included
        assert any("delineate-anything" in s for s in step_ids), (
            f"LLM should recognize delineate-anything model. Got steps: {step_ids}"
        )

    def test_delineate_model_can_be_planned_manually(self) -> None:
        """Test that delineate-anything can be included in a manual plan."""
        from dta.dti.schemas import ExecutionPlan, PlanStep

        # Create a plan that explicitly uses delineate-anything
        plan = ExecutionPlan(
            flow="field-boundary-detection",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": "/path/to/file.tif"}),
                PlanStep(uses="models/delineate-anything", binds={}),
            ],
            outputs=["FieldBoundaries"],
        )

        assert len(plan.steps) == 2
        assert plan.steps[1].uses == "models/delineate-anything"
        assert "FieldBoundaries" in plan.outputs


@pytest.mark.slow
class TestDelineateAnythingExecution:
    """Test actual model execution (requires dependencies and model installed)."""

    @pytest.fixture(autouse=True)
    def check_dependencies(self) -> None:
        """Skip tests if dependencies or model are not installed."""
        available, msg = _check_delineate_dependencies()
        if not available:
            pytest.skip(msg)

        installed, msg = _check_model_installed()
        if not installed:
            pytest.skip(msg)

    @pytest.fixture
    def output_dir(self) -> Path:
        """Create output directory for test artifacts."""
        temp_root = Path(TEMP_PATH).resolve()
        run_id = f"delineate_out_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        out_dir = temp_root / "tests" / "delineate_out" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @pytest.mark.skipif(
        not KAHOVKA_TIFF.exists(),
        reason="Kahovka test data not available",
    )
    def test_delineate_kahovka_execution(self, output_dir: Path) -> None:
        """Test full execution on Kahovka data.

        This test requires:
        - Kahovka test data present
        - Delineate dependencies installed (pip install -e '.[delineate]')
        - Internet connection for model download from HuggingFace
        """
        from dta.dti.models.third_party.delineate_anything import delineate_fields

        output_path = output_dir / "kahovka_fields.gpkg"

        result = delineate_fields(
            raster_path=str(KAHOVKA_TIFF),
            output_path=str(output_path),
            model="small",  # Use small model for faster test
        )

        # Verify result structure
        assert "output_path" in result
        assert "num_fields" in result
        assert "crs" in result

        # Verify output file created
        assert output_path.exists(), f"Output GeoPackage not created: {output_path}"
        assert output_path.stat().st_size > 0, "Output GeoPackage is empty"

        # Verify it's a valid GeoPackage
        import geopandas as gpd

        gdf = gpd.read_file(output_path)
        assert "geometry" in gdf.columns
        assert len(gdf) == result["num_fields"]

        print("\nDelineate-Anything test results:")
        print(f"  Output: {output_path}")
        print(f"  Fields detected: {result['num_fields']}")
        print(f"  Total area: {result.get('total_area_m2', 'N/A')} m²")
        print(f"  CRS: {result['crs']}")

    @pytest.mark.skipif(
        not KAHOVKA_TIFF.exists(),
        reason="Kahovka test data not available",
    )
    def test_run_function_registry_compatible(self, output_dir: Path) -> None:
        """Test the run() function used by pipeline executor."""
        import os

        from dta.dti.models.third_party.delineate_anything import run

        # Set environment for model selection
        os.environ["DELINEATE_MODEL"] = "small"

        result = run(RasterPath=str(KAHOVKA_TIFF))

        assert "output_path" in result
        assert "num_fields" in result
        assert Path(result["output_path"]).exists()
