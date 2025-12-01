"""Tests for pipeline executor functionality.

Tests the PipelineExecutor class and runner dispatch.
"""

from pathlib import Path

import pytest

from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ExecutionPlan, PlanStep


class TestExecutorInitialization:
    """Tests for executor initialization."""

    def test_executor_can_be_initialized(self) -> None:
        """Test that executor can be initialized."""
        executor = PipelineExecutor()

        assert executor.registry is not None
        assert len(executor.registry.instances) > 0

    def test_executor_has_empty_artifacts_initially(self) -> None:
        """Test that artifacts dict is empty on init."""
        executor = PipelineExecutor()

        assert executor.artifacts == {}


class TestPassthroughRunner:
    """Tests for passthrough runner."""

    def test_passthrough_runner_sets_artifact(self) -> None:
        """Test passthrough runner for input/file."""
        executor = PipelineExecutor()

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

    def test_passthrough_multiple_outputs(self) -> None:
        """Test passthrough with multiple outputs."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test-dual",
            steps=[
                PlanStep(
                    uses="input/file-before",
                    binds={"RasterPathBefore": "/path/before.tif"},
                ),
                PlanStep(
                    uses="input/file-after",
                    binds={"RasterPathAfter": "/path/after.tif"},
                ),
            ],
            outputs=["RasterPathBefore", "RasterPathAfter"],
        )

        result = executor.execute(plan)

        assert result["artifacts"]["RasterPathBefore"] == "/path/before.tif"
        assert result["artifacts"]["RasterPathAfter"] == "/path/after.tif"


class TestPythonRunner:
    """Tests for Python script runner."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "resources/kahovka_data").exists(),
        reason="Kahovka data not available",
    )
    def test_python_runner_executes_ndvi(self) -> None:
        """Test Python runner executes NDVI algorithm."""
        from dta.config import ROOT_DIR

        data_dir = ROOT_DIR / "resources/kahovka_data"
        tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

        if not tif_files:
            pytest.skip("No GeoTIFF files found")

        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="ndvi-test",
            steps=[
                PlanStep(
                    uses="input/file",
                    binds={"RasterPath": str(tif_files[0])},
                ),
                PlanStep(uses="algorithms/ndvi", binds={}),
            ],
            outputs=["NDVIMap"],
        )

        result = executor.execute(plan)

        assert "NDVIMap" in result["artifacts"]
        assert result["artifacts"]["NDVIMap"] is not None


class TestProgressCallback:
    """Tests for execution progress callback."""

    def test_progress_callback_called(self) -> None:
        """Test that progress callback is called."""
        executor = PipelineExecutor()
        progress_events = []

        def on_progress(event: dict) -> None:
            progress_events.append(event)

        plan = ExecutionPlan(
            flow="callback-test",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"}),
            ],
            outputs=["RasterPath"],
        )

        executor.execute(plan, on_progress=on_progress)

        assert len(progress_events) >= 2  # start + complete
        assert progress_events[0]["event"] == "step_start"
        assert progress_events[1]["event"] == "step_complete"


class TestExecutionResults:
    """Tests for execution result structure."""

    def test_result_contains_flow(self) -> None:
        """Test result contains flow name."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="my-flow",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )

        result = executor.execute(plan)

        assert result["flow"] == "my-flow"

    def test_result_contains_steps(self) -> None:
        """Test result contains executed steps."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )

        result = executor.execute(plan)

        assert "steps" in result
        assert "input/file" in result["steps"]

    def test_result_contains_artifacts(self) -> None:
        """Test result contains artifacts dict."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )

        result = executor.execute(plan)

        assert "artifacts" in result
        assert isinstance(result["artifacts"], dict)

    def test_result_contains_outputs(self) -> None:
        """Test result contains final outputs."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )

        result = executor.execute(plan)

        assert "outputs" in result
