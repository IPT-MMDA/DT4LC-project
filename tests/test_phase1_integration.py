"""Phase 1 Integration Test - End-to-end COE → Executor flow."""

from pathlib import Path

import pytest

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ChatRequest, ExecutionPlan


def test_simple_orchestration_no_execution() -> None:
    """Test that COE can generate a valid plan."""
    req = ChatRequest(
        prompt="Calculate NDVI for vegetation analysis",
        attachments=[],
    )

    result = orchestrate(req)

    assert result["ok"], f"Orchestration failed: {result}"
    assert "plan" in result
    plan_dict = result["plan"]

    assert "steps" in plan_dict
    assert len(plan_dict["steps"]) > 0

    # Verify plan structure
    for step in plan_dict["steps"]:
        assert "uses" in step
        assert isinstance(step["uses"], str)


def test_registry_loading() -> None:
    """Test that registry loads successfully."""
    from dta.dti.registry import load_registry

    registry = load_registry()

    assert registry.version == "0.1"
    assert len(registry.types) > 0
    assert len(registry.instances) > 0

    # Verify NDVI algorithm exists
    ndvi_items = [i for i in registry.instances if i.id == "algorithms/ndvi"]
    assert len(ndvi_items) == 1
    assert ndvi_items[0].runner.type == "python"


def test_executor_initialization() -> None:
    """Test that executor can be initialized."""
    executor = PipelineExecutor()

    assert executor.registry is not None
    assert len(executor.registry.instances) > 0


def test_algorithm_entrypoints_exist() -> None:
    """Verify algorithm entrypoint files exist."""
    from dta.config import ROOT_DIR

    ndvi_path = ROOT_DIR / "dta/dti/algorithms/ndvi.py"
    stats_path = ROOT_DIR / "dta/dti/algorithms/statistics.py"

    assert ndvi_path.exists(), f"NDVI algorithm not found: {ndvi_path}"
    assert stats_path.exists(), f"Statistics algorithm not found: {stats_path}"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
    reason="Kahovka data not available",
)
def test_ndvi_algorithm_direct() -> None:
    """Test NDVI algorithm directly (if data available)."""
    from dta.config import ROOT_DIR
    from dta.dti.algorithms.ndvi import calculate_ndvi

    # Try to find a test raster
    data_dir = ROOT_DIR / "resources/kahovka_data"
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

    if not tif_files:
        pytest.skip("No GeoTIFF files found in kahovka_data")

    result = calculate_ndvi(str(tif_files[0]))

    assert "ndvi_array" in result
    assert "metadata" in result
    assert "statistics" in result

    # Check statistics
    stats = result["statistics"]
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert stats["valid_pixels"] > 0


def test_passthrough_runner() -> None:
    """Test passthrough runner for input/file."""
    from dta.dti.schemas import PlanStep

    executor = PipelineExecutor()

    # Create a simple plan with passthrough
    plan = ExecutionPlan(
        flow="test",
        steps=[
            PlanStep(
                uses="input/file",
                binds={"RasterPath": "/fake/path/to/file.tif"},
            )
        ],
        outputs=["RasterPath"],
    )

    result = executor.execute(plan)

    assert result["flow"] == "test"
    assert "RasterPath" in result["artifacts"]
    assert result["artifacts"]["RasterPath"] == "/fake/path/to/file.tif"


def test_server_endpoints_importable() -> None:
    """Verify server can be imported without errors."""
    try:
        from server import app

        assert app.app.title == "DT4LC API"
    except ImportError as e:
        pytest.skip(f"Server dependencies not installed: {e}")
