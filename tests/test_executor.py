"""Tests for pipeline executor functionality.

Tests the PipelineExecutor class and runner dispatch.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.executor import (
    CancellationError,
    ExecutionError,
    MissingInputError,
    ModelNotInstalledError,
    PipelineExecutor,
    _get_friendly_missing_input_message,
)
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

    def test_executor_has_empty_executed_items(self) -> None:
        """Test that executed_items list is empty on init."""
        executor = PipelineExecutor()

        assert executor.executed_items == []


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

    def test_passthrough_missing_input_raises(self) -> None:
        """Test that missing required input raises MissingInputError."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test",
            steps=[
                PlanStep(uses="input/file", binds={}),
            ],
            outputs=["RasterPath"],
        )

        with pytest.raises(ExecutionError):
            executor.execute(plan)


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

        result = executor.execute(plan)

        assert "NDVIMap" in result["artifacts"]
        assert result["artifacts"]["NDVIMap"] is not None

    def test_python_runner_missing_entrypoint(self) -> None:
        executor = PipelineExecutor()
        from dta.dti.schemas import PlanStep, RegistryItem, Runner

        item = RegistryItem(
            id="test/missing", kind="algorithm", runner=Runner(type="python", entrypoint=""), inputs=[], outputs=[]
        )
        step = PlanStep(uses="test/missing", binds={})

        with pytest.raises(ExecutionError, match="missing entrypoint"):
            executor._run_python(step, item)

    def test_apply_preprocessors(self) -> None:
        executor = PipelineExecutor()
        from dta.dti.schemas import PreprocessorRef, RegistryItem, Runner

        # Mock preprocessor registry item
        item = RegistryItem(
            id="test/item",
            kind="algorithm",
            runner=Runner(type="python"),
            inputs=["RasterPath"],
            outputs=[],
            preprocessors=[PreprocessorRef(id="preprocessors/scale", apply_to="RasterPath")],
        )

        mock_preproc = RegistryItem(
            id="preprocessors/scale",
            kind="preprocessor",
            runner=Runner(type="python", entrypoint="dummy_entry.py"),
            inputs=[],
            outputs=[],
        )

        inputs = {"RasterPath": "/input.tif"}

        with patch("dta.dti.executor.get_item") as mock_get:
            mock_get.return_value = mock_preproc
            # Simulate failure in loading to test fallback behavior
            with patch("importlib.util.spec_from_file_location", return_value=None):
                result = executor._apply_preprocessors(item, inputs, "/out")

        # Should fallback to original value
        assert result["RasterPath"] == "/input.tif"

    def test_convert_tiffs_to_visualizations(self) -> None:
        executor = PipelineExecutor()
        # Create a tiny mock tiff
        import tempfile

        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
            tif_path = Path(tf.name)

        data = np.zeros((3, 2, 2), dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 2, 2)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=3,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        res = executor._convert_tiffs_to_visualizations([tif_path], "test")
        assert tif_path.stem in res

        tif_path.unlink()

    def test_model_not_installed_check(self) -> None:
        executor = PipelineExecutor()
        from dta.dti.schemas import PlanStep, RegistryItem, Runner

        step = PlanStep(uses="models/delineate-anything", binds={})
        item = RegistryItem(
            id="models/delineate-anything", kind="model", runner=Runner(type="python"), inputs=[], outputs=[]
        )

        with patch("dta.dti.models.model_manager.ModelManager.is_model_available", return_value=False):
            with pytest.raises(ModelNotInstalledError):
                executor._check_model_availability(step, item)


class TestProgressCallback:
    """Tests for execution progress callback."""

    def test_progress_callback_called(self) -> None:
        """Test that progress callback is called."""
        executor = PipelineExecutor()
        progress_events: list[dict[str, Any]] = []

        def on_progress(event: dict[str, Any]) -> None:
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

    def test_progress_callback_on_error(self) -> None:
        """Test that progress callback is called on error."""
        executor = PipelineExecutor()
        progress_events: list[dict[str, Any]] = []

        def on_progress(event: dict[str, Any]) -> None:
            progress_events.append(event)

        plan = ExecutionPlan(
            flow="error-test",
            steps=[
                PlanStep(uses="input/file", binds={}),  # Missing required binds
            ],
            outputs=["RasterPath"],
        )

        with pytest.raises(ExecutionError):
            executor.execute(plan, on_progress=on_progress)

        # Should have start + error events
        assert any(e["event"] == "step_start" for e in progress_events)
        assert any(e["event"] == "step_error" for e in progress_events)


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

    def test_outputs_colon_format(self) -> None:
        """Test outputs with 'publish: key' format."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["publish: RasterPath"],
        )

        result = executor.execute(plan)

        assert "RasterPath" in result["outputs"]


class TestCancellation:
    """Tests for execution cancellation."""

    def test_cancellation_raises(self) -> None:
        """Test that cancellation callback stops execution."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="cancel-test",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"}),
                PlanStep(uses="input/file-before", binds={"RasterPathBefore": "/test2.tif"}),
            ],
            outputs=["RasterPath"],
        )

        with pytest.raises(CancellationError):
            executor.execute(plan, is_cancelled=lambda: True)

    def test_no_cancellation_completes(self) -> None:
        """Test that execution completes without cancellation."""
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="no-cancel-test",
            steps=[PlanStep(uses="input/file", binds={"RasterPath": "/test.tif"})],
            outputs=["RasterPath"],
        )

        result = executor.execute(plan, is_cancelled=lambda: False)
        assert result["flow"] == "no-cancel-test"


class TestUnknownRunner:
    """Tests for unknown runner type."""

    def test_unknown_runner_type_raises(self) -> None:
        """Test that unknown runner type raises ExecutionError."""
        executor = PipelineExecutor()

        # Use model_construct to bypass Pydantic validation for the Literal type
        from dta.dti.schemas import RegistryItem, Runner

        mock_runner = Runner.model_construct(type="unknown_runner_type")
        mock_item = RegistryItem.model_construct(
            id="test/unknown",
            kind="algorithm",
            runner=mock_runner,
            inputs=[],
            outputs=[],
            preprocessors=[],
        )

        step = PlanStep(uses="test/unknown", binds={})

        with pytest.raises(ExecutionError, match="Unknown runner type"):
            executor._execute_step(step, mock_item)


class TestResolveVariables:
    """Tests for _resolve_variables."""

    def test_basic_resolution(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_variables("${MODEL_PATH}/model.bin", {"MODEL_PATH": "/models"})
        assert result == "/models/model.bin"

    def test_missing_variable_kept(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_variables("${MISSING}/file", {})
        assert result == "${MISSING}/file"

    def test_none_variable_kept(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_variables("${VAR}/file", {"VAR": None})
        assert result == "${VAR}/file"

    def test_multiple_variables(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_variables("${A}/${B}", {"A": "hello", "B": "world"})
        assert result == "hello/world"

    def test_no_variables(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_variables("plain string", {})
        assert result == "plain string"


class TestResolveArgsMap:
    """Tests for _resolve_args_map and _resolve_value."""

    def test_string_resolution(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_args_map({"path": "${DIR}/file"}, {"DIR": "/tmp"})
        assert result == {"path": "/tmp/file"}

    def test_list_resolution(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_value(["${A}", "${B}"], {"A": "1", "B": "2"})
        assert result == ["1", "2"]

    def test_dict_resolution(self) -> None:
        executor = PipelineExecutor()
        result = executor._resolve_value({"k": "${V}"}, {"V": "val"})
        assert result == {"k": "val"}

    def test_non_string_passthrough(self) -> None:
        executor = PipelineExecutor()
        assert executor._resolve_value(42, {}) == 42
        assert executor._resolve_value(True, {}) is True
        assert executor._resolve_value(None, {}) is None
        assert executor._resolve_value(3.14, {}) == 3.14


class TestParseToolRequest:
    """Tests for _parse_tool_request."""

    def test_json_in_markdown(self) -> None:
        executor = PipelineExecutor()
        text = 'Some text\n```json\n{"tool": "compute_image_similarity", "args": {"image1_key": "a"}}\n```\nMore text'
        result = executor._parse_tool_request(text)
        assert result is not None
        assert result["tool"] == "compute_image_similarity"

    def test_raw_json(self) -> None:
        executor = PipelineExecutor()
        text = 'I need to check: {"tool": "analyze_spatial_patterns", "args": {"image_key": "test"}}'
        result = executor._parse_tool_request(text)
        assert result is not None
        assert result["tool"] == "analyze_spatial_patterns"

    def test_no_tool_request(self) -> None:
        executor = PipelineExecutor()
        text = "This is just a plain text summary of the analysis results."
        result = executor._parse_tool_request(text)
        assert result is None

    def test_invalid_json_returns_none(self) -> None:
        executor = PipelineExecutor()
        text = '```json\n{"tool": broken}\n```'
        result = executor._parse_tool_request(text)
        assert result is None


class TestCleanSummaryText:
    """Tests for _clean_summary_text."""

    def test_removes_markdown_json(self) -> None:
        executor = PipelineExecutor()
        text = 'Summary.\n```json\n{"tool": "test", "args": {}}\n```\nMore text.'
        result = executor._clean_summary_text(text)
        assert "tool" not in result
        assert "Summary." in result
        assert "More text." in result

    def test_removes_raw_json(self) -> None:
        executor = PipelineExecutor()
        text = 'Before {"tool": "test", "args": {"k": "v"}} After'
        result = executor._clean_summary_text(text)
        assert "tool" not in result

    def test_collapses_whitespace(self) -> None:
        executor = PipelineExecutor()
        text = "Line1\n\n\n\n\nLine2"
        result = executor._clean_summary_text(text)
        assert "\n\n\n" not in result

    def test_strips(self) -> None:
        executor = PipelineExecutor()
        result = executor._clean_summary_text("  text  ")
        assert result == "text"


class TestBuildContextDescription:
    """Tests for _build_context_description."""

    def test_dict_with_visualizations(self) -> None:
        executor = PipelineExecutor()
        inputs = {"result": {"visualizations": {"rgb": "base64data"}}}
        desc = executor._build_context_description(inputs)
        assert "Visualizations" in desc

    def test_dict_with_output_files(self) -> None:
        executor = PipelineExecutor()
        inputs = {"result": {"output_files": ["/tmp/out.tif"]}}
        desc = executor._build_context_description(inputs)
        assert "Output files" in desc

    def test_dict_with_statistics(self) -> None:
        executor = PipelineExecutor()
        inputs = {"result": {"statistics": {"mean": 0.5}}}
        desc = executor._build_context_description(inputs)
        assert "statistics" in desc.lower()

    def test_dict_generic_keys(self) -> None:
        executor = PipelineExecutor()
        inputs = {"result": {"key1": "val1", "key2": "val2"}}
        desc = executor._build_context_description(inputs)
        assert "key1" in desc

    def test_raster_path(self) -> None:
        executor = PipelineExecutor()
        inputs = {"RasterPath": "/path/to/file.tif"}
        desc = executor._build_context_description(inputs)
        assert "Raster file" in desc

    def test_tiff_extension(self) -> None:
        executor = PipelineExecutor()
        inputs = {"RasterPath": "/path/to/file.tiff"}
        desc = executor._build_context_description(inputs)
        assert "Raster file" in desc

    def test_other_type(self) -> None:
        executor = PipelineExecutor()
        inputs = {"count": 42}
        desc = executor._build_context_description(inputs)
        assert "int" in desc


class TestFormatStatsForLLM:
    """Tests for _format_stats_for_llm."""

    def test_nested_statistics(self) -> None:
        executor = PipelineExecutor()
        inputs = {"NDVIMap": {"statistics": {"mean": 0.45, "std": 0.12}}}
        result = executor._format_stats_for_llm(inputs)
        assert "mean" in result
        assert "0.4500" in result

    def test_capitalized_statistics(self) -> None:
        executor = PipelineExecutor()
        inputs = {"result": {"Statistics": {"count": 1000}}}
        result = executor._format_stats_for_llm(inputs)
        assert "1,000" in result

    def test_top_level_statistics(self) -> None:
        executor = PipelineExecutor()
        inputs = {"statistics": {"min": 0.0, "max": 1.0}}
        result = executor._format_stats_for_llm(inputs)
        assert "min" in result

    def test_no_statistics(self) -> None:
        executor = PipelineExecutor()
        inputs = {"RasterPath": "/test.tif"}
        result = executor._format_stats_for_llm(inputs)
        assert result == "No statistics available."

    def test_string_stat_value(self) -> None:
        executor = PipelineExecutor()
        inputs = {"statistics": {"type": "ndvi"}}
        result = executor._format_stats_for_llm(inputs)
        assert "ndvi" in result

    def test_nested_dict_skipped(self) -> None:
        executor = PipelineExecutor()
        inputs = {"statistics": {"per_band": {"b1": 0.5}}}
        result = executor._format_stats_for_llm(inputs)
        assert "per_band" not in result  # nested dicts are skipped


class TestDetectAnalysisType:
    """Tests for _detect_analysis_type."""

    def test_ndvi(self) -> None:
        executor = PipelineExecutor()
        assert "NDVI" in executor._detect_analysis_type({"NDVIMap": {}})

    def test_ndsi(self) -> None:
        executor = PipelineExecutor()
        assert "NDSI" in executor._detect_analysis_type({"NDSIMap": {}})

    def test_snow(self) -> None:
        executor = PipelineExecutor()
        assert "Snow" in executor._detect_analysis_type({"SnowClassification": {}})

    def test_change(self) -> None:
        executor = PipelineExecutor()
        assert "change" in executor._detect_analysis_type({"ChangeMap": {}})

    def test_field_boundaries(self) -> None:
        executor = PipelineExecutor()
        assert "field" in executor._detect_analysis_type({"FieldBoundaries": {}}).lower()

    def test_features(self) -> None:
        executor = PipelineExecutor()
        assert "feature" in executor._detect_analysis_type({"Features": {}}).lower()

    def test_reconstruction(self) -> None:
        executor = PipelineExecutor()
        assert "reconstruction" in executor._detect_analysis_type({"Reconstruction": {}}).lower()

    def test_statistics(self) -> None:
        executor = PipelineExecutor()
        assert "statistics" in executor._detect_analysis_type({"Statistics": {}}).lower()

    def test_generic_fallback(self) -> None:
        executor = PipelineExecutor()
        assert executor._detect_analysis_type({"unknown": {}}) == "geospatial"


class TestAgentSummarize:
    """Tests for _agent_summarize with mocked LLM."""

    def test_basic_summarization(self) -> None:
        """Test basic summarization without tool calls."""
        executor = PipelineExecutor()

        mock_response = MagicMock()
        mock_response.text = "The vegetation analysis shows healthy coverage."
        mock_response.provider = "mock"

        mock_router = MagicMock()
        mock_router.generate.return_value = mock_response

        with patch("dta.dti.executor.get_llm_router", return_value=mock_router):
            result = executor._agent_summarize({"NDVIMap": {"statistics": {"mean": 0.5}}})

        assert "summary" in result
        assert "vegetation" in result["summary"].lower()
        assert result["llm_provider"] == "mock"

    def test_tool_call_then_response(self) -> None:
        """Test tool request followed by final response."""
        executor = PipelineExecutor()

        # First response: tool request
        tool_response = MagicMock()
        tool_response.text = (
            '```json\n{"tool": "compute_image_similarity", "args": {"image1_key": "a", "image2_key": "b"}}\n```'
        )

        # Second response: final interpretation
        final_response = MagicMock()
        final_response.text = "Based on the similarity metrics, the images are quite similar."
        final_response.provider = "mock"

        mock_router = MagicMock()
        mock_router.generate.side_effect = [tool_response, final_response]

        with (
            patch("dta.dti.executor.get_llm_router", return_value=mock_router),
            patch("dta.dti.executor.PipelineExecutor._execute_analysis_tool", return_value={"similarity": 0.95}),
        ):
            result = executor._agent_summarize({"result": {"visualizations": {"rgb": "base64"}}})

        assert "summary" in result
        assert "computed_metrics" in result

    def test_llm_failure_graceful_degradation(self) -> None:
        """Test graceful degradation when LLM fails."""
        executor = PipelineExecutor()

        mock_router = MagicMock()
        mock_router.generate.side_effect = Exception("LLM unavailable")

        with patch("dta.dti.executor.get_llm_router", return_value=mock_router):
            result = executor._agent_summarize({"NDVIMap": {}})

        assert "summary" in result
        assert "failed" in result["summary"].lower()


class TestGetInterpretationGuide:
    """Tests for _get_interpretation_guide."""

    def test_with_executed_items(self) -> None:
        executor = PipelineExecutor()
        from dta.dti.schemas import RegistryItem, Runner

        item = RegistryItem(
            id="algorithms/ndvi",
            kind="algorithm",
            runner=Runner(type="python"),
            inputs=["RasterPath"],
            outputs=["NDVIMap"],
            interpretation="NDVI values range from -1 to 1.",
        )
        executor.executed_items = [item]

        guide = executor._get_interpretation_guide()
        assert "NDVI" in guide

    def test_without_executed_items(self) -> None:
        executor = PipelineExecutor()
        executor.executed_items = []

        guide = executor._get_interpretation_guide()
        assert "remote sensing" in guide.lower()

    def test_skips_input_items(self) -> None:
        executor = PipelineExecutor()
        from dta.dti.schemas import RegistryItem, Runner

        item = RegistryItem(
            id="input/file",
            kind="input",
            runner=Runner(type="passthrough"),
            inputs=[],
            outputs=["RasterPath"],
            interpretation="This should be skipped.",
        )
        executor.executed_items = [item]

        guide = executor._get_interpretation_guide()
        assert "remote sensing" in guide.lower()  # Falls back to default


class TestFormatToolResult:
    """Tests for _format_tool_result."""

    def test_float_rounding(self) -> None:
        executor = PipelineExecutor()
        result = executor._format_tool_result({"score": 0.123456789})
        assert "0.123457" in result

    def test_nested_dict(self) -> None:
        executor = PipelineExecutor()
        result = executor._format_tool_result({"nested": {"val": 0.5}})
        assert "0.5" in result


class TestFriendlyMissingInputMessage:
    """Tests for _get_friendly_missing_input_message."""

    def test_input_file(self) -> None:
        msg = _get_friendly_missing_input_message("input/file", "RasterPath")
        assert "upload" in msg.lower()
        assert "GeoTIFF" in msg

    def test_input_file_before(self) -> None:
        msg = _get_friendly_missing_input_message("input/file-before", "RasterPathBefore")
        assert "before" in msg.lower()

    def test_input_file_after(self) -> None:
        msg = _get_friendly_missing_input_message("input/file-after", "RasterPathAfter")
        assert "after" in msg.lower()

    def test_raster_input_type(self) -> None:
        msg = _get_friendly_missing_input_message("input/custom", "RasterPath")
        assert "GeoTIFF" in msg

    def test_generic_fallback(self) -> None:
        msg = _get_friendly_missing_input_message("input/custom", "CustomData")
        assert "input" in msg.lower()


class TestExceptionClasses:
    """Tests for executor exception classes."""

    def test_missing_input_error(self) -> None:
        err = MissingInputError("input/file", "RasterPath")
        assert "input/file" in str(err)
        assert err.step_id == "input/file"
        assert err.input_type == "RasterPath"

    def test_missing_input_error_custom_message(self) -> None:
        err = MissingInputError("input/file", "RasterPath", "Custom message")
        assert str(err) == "Custom message"

    def test_model_not_installed_error(self) -> None:
        err = ModelNotInstalledError(
            step_id="models/test",
            model_id="test-model",
            model_name="Test Model",
            size_mb=500,
            description="A test model",
        )
        assert err.model_id == "test-model"
        assert err.size_mb == 500

        d = err.to_dict()
        assert d["error"] == "model_not_installed"
        assert d["model_required"]["id"] == "test-model"
        assert "500 MB" in d["message"]

    def test_cancellation_error(self) -> None:
        err = CancellationError("Cancelled")
        assert str(err) == "Cancelled"

    def test_execution_error(self) -> None:
        err = ExecutionError("Step failed")
        assert str(err) == "Step failed"
