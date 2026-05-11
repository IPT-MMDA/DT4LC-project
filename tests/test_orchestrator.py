"""Tests for orchestrator and data loader.

Tests orchestration flow, data loader inclusion, plan execution order,
and intent classification integration.

Tests that require LLM access are mocked to avoid CI failures.
Tests marked @pytest.mark.llm require live LLM providers.
"""

from unittest.mock import patch

import pytest

from dta.dti.coe.orchestrator import _inject_file_bindings, orchestrate
from dta.dti.schemas import Attachment, ChatRequest, ExecutionPlan, PlanStep


# Fixture for a mock attachment
@pytest.fixture
def mock_attachment() -> Attachment:
    """Create a mock attachment for testing."""
    return Attachment(id="test", filename="test.tif", mime_type="image/tiff", path="/tmp/test.tif")


class TestOrchestratorIntentClassification:
    """Tests for orchestrator intent classification."""

    @pytest.mark.llm
    def test_pipeline_intent_returns_plan(self, mock_attachment: Attachment) -> None:
        """Test that pipeline intent requests with attachments return a plan.

        Mocks the LLM-dependent classify_intent to avoid needing API keys.
        """
        mock_plan = ExecutionPlan(
            flow="ndvi",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": "/tmp/test.tif"}),
                PlanStep(uses="algorithms/ndvi", binds={}),
            ],
            outputs=["NDVIMap"],
        )

        with (
            patch("dta.dti.coe.orchestrator.classify_intent") as mock_classify,
            patch("dta.dti.coe.orchestrator.analyze"),
            patch("dta.dti.coe.orchestrator.plan", return_value=mock_plan),
            patch("dta.dti.coe.orchestrator.validate", return_value=mock_plan),
        ):
            from dta.dti.coe.intent_classifier import IntentType

            mock_classify.return_value = {
                "intent": IntentType.PIPELINE,
                "reason": "Has attachments and action keywords",
            }

            req = ChatRequest(prompt="calculate ndvi", attachments=[mock_attachment])
            result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline", f"Expected pipeline intent, got: {result.get('intent')}"
        assert "plan" in result, "Pipeline intent should include a plan"

    @pytest.mark.llm
    def test_conversation_intent_returns_response(self) -> None:
        """Test that conversation intent requests return a response."""
        with patch("dta.dti.coe.orchestrator.classify_intent") as mock_classify:
            from dta.dti.coe.intent_classifier import IntentType

            mock_classify.return_value = {
                "intent": IntentType.CONVERSATION,
                "reason": "Capability question",
                "response": "I can help you with geospatial analysis!",
            }

            req = ChatRequest(prompt="what can we do next?", attachments=[])
            result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "conversation", f"Expected conversation intent, got: {result.get('intent')}"
        assert "response" in result, "Conversation intent should include a response"

    def test_action_without_file_asks_for_upload(self) -> None:
        """Test that action requests without attachments ask for file upload."""
        req = ChatRequest(prompt="calculate ndvi", attachments=[])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "conversation", "Action without file should be conversation"
        assert "response" in result, "Should include helpful response asking for file"


class TestOrchestratorPlanValidationFailure:
    """Tests for orchestrator when plan validation fails."""

    def test_plan_validation_error(self, mock_attachment: Attachment) -> None:
        """Test that PlanError is handled gracefully."""
        from dta.dti.coe.plan_validator import PlanError

        mock_plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={})],
            outputs=["RasterPath"],
        )

        with (
            patch("dta.dti.coe.orchestrator.classify_intent") as mock_classify,
            patch("dta.dti.coe.orchestrator.analyze"),
            patch("dta.dti.coe.orchestrator.plan", return_value=mock_plan),
            patch("dta.dti.coe.orchestrator.validate", side_effect=PlanError("Invalid plan")),
        ):
            from dta.dti.coe.intent_classifier import IntentType

            mock_classify.return_value = {
                "intent": IntentType.PIPELINE,
                "reason": "test",
            }

            req = ChatRequest(prompt="calculate ndvi", attachments=[mock_attachment])
            result = orchestrate(req)

        assert result.get("ok") is False
        assert "error" in result
        assert "candidate" in result


class TestInjectFileBindings:
    """Tests for _inject_file_bindings helper."""

    def test_single_file_injection(self) -> None:
        """Test single file path injection."""
        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={})],
            outputs=["RasterPath"],
        )
        att = Attachment(id="t", filename="t.tif", mime_type="image/tiff", path="/tmp/t.tif")

        _inject_file_bindings(plan, [att])

        assert plan.steps[0].binds.get("RasterPath") == "/tmp/t.tif"

    def test_dual_file_injection(self) -> None:
        """Test before/after file injection for change detection."""
        plan = ExecutionPlan(
            flow="change",
            steps=[
                PlanStep(uses="input/file-before", binds={}),
                PlanStep(uses="input/file-after", binds={}),
                PlanStep(uses="algorithms/change-detection", binds={}),
            ],
            outputs=["ChangeMap"],
        )
        att1 = Attachment(id="t1", filename="before.tif", mime_type="image/tiff", path="/tmp/before.tif")
        att2 = Attachment(id="t2", filename="after.tif", mime_type="image/tiff", path="/tmp/after.tif")

        _inject_file_bindings(plan, [att1, att2])

        assert plan.steps[0].binds.get("RasterPathBefore") == "/tmp/before.tif"
        assert plan.steps[1].binds.get("RasterPathAfter") == "/tmp/after.tif"

    def test_attachment_without_path_skips(self) -> None:
        """Test that attachments without paths are skipped with warning."""
        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="input/file", binds={})],
            outputs=["RasterPath"],
        )
        att = Attachment(id="t", filename="t.tif", mime_type="image/tiff", path="")

        _inject_file_bindings(plan, [att])

        # Path is empty string (falsy), so RasterPath should NOT be set
        assert "RasterPath" not in plan.steps[0].binds

    def test_more_steps_than_attachments(self) -> None:
        """Test graceful handling when more input steps than attachments."""
        plan = ExecutionPlan(
            flow="test",
            steps=[
                PlanStep(uses="input/file-before", binds={}),
                PlanStep(uses="input/file-after", binds={}),
            ],
            outputs=["ChangeMap"],
        )
        att = Attachment(id="t", filename="before.tif", mime_type="image/tiff", path="/tmp/before.tif")

        # Only one attachment for two input steps - should not crash
        _inject_file_bindings(plan, [att])

        assert plan.steps[0].binds.get("RasterPathBefore") == "/tmp/before.tif"
        # Second step should not have binding
        assert "RasterPathAfter" not in plan.steps[1].binds


@pytest.mark.llm
class TestOrchestratorDataLoader:
    """Tests for orchestrator data loader inclusion.

    These tests require LLM access for context analysis.
    """

    def test_orchestration_includes_data_loader(self, mock_attachment: Attachment) -> None:
        """Test that orchestration includes input/file step."""
        req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[mock_attachment])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline", f"Expected pipeline intent, got: {result.get('intent')}"

        plan = result.get("plan", {})
        steps = [s.get("uses") for s in plan.get("steps", [])]

        assert len(steps) >= 2, f"Plan too short: {steps}"

        assert steps[0] == "input/file", f"First step should be input/file, got: {steps[0]}"

        processing_steps = [s for s in steps if "algorithms/" in s or "models/" in s]
        assert len(processing_steps) > 0, f"No processing steps found in: {steps}"

    def test_statistics_plan_includes_data_loader(self, mock_attachment: Attachment) -> None:
        """Test statistics request includes data loader."""
        req = ChatRequest(prompt="calculate statistics on kahovka", attachments=[mock_attachment])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline"

        steps = [s.get("uses") for s in result["plan"]["steps"]]
        assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"
        assert "algorithms/statistics" in steps, f"Should include statistics step: {steps}"

    def test_vegetation_analysis_includes_data_loader(self, mock_attachment: Attachment) -> None:
        """Test vegetation analysis includes data loader."""
        req = ChatRequest(prompt="analyze vegetation health", attachments=[mock_attachment])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline"

        steps = [s.get("uses") for s in result["plan"]["steps"]]
        assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"


@pytest.mark.llm
class TestOrchestratorPrompts:
    """Tests for various prompts.

    These tests require LLM access for context analysis.
    """

    def test_various_prompts_include_data_loader(self, mock_attachment: Attachment) -> None:
        """Test that various prompts all include appropriate data loader."""
        single_file_prompts = [
            "compute ndvi",
            "get statistics",
            "analyze land cover",
        ]

        for prompt in single_file_prompts:
            req = ChatRequest(prompt=prompt, attachments=[mock_attachment])
            result = orchestrate(req)

            assert result.get("ok"), f"Failed for '{prompt}': {result.get('error')}"
            assert result.get("intent") == "pipeline", f"Expected pipeline intent for '{prompt}'"

            steps = [s.get("uses") for s in result["plan"]["steps"]]
            assert steps[0] == "input/file", f"Prompt '{prompt}' missing data loader. Steps: {steps}"


@pytest.mark.llm
class TestOrchestratorChangeDetection:
    """Tests for change detection orchestration.

    These tests require LLM access for context analysis.
    """

    def test_change_detection_uses_dual_input(self) -> None:
        """Test that change detection prompts use dual file input."""
        att1 = Attachment(id="test1", filename="before.tif", mime_type="image/tiff", path="/tmp/before.tif")
        att2 = Attachment(id="test2", filename="after.tif", mime_type="image/tiff", path="/tmp/after.tif")
        change_prompts = [
            "detect changes in kahovka",
            "compare before and after images",
        ]

        for prompt in change_prompts:
            req = ChatRequest(prompt=prompt, attachments=[att1, att2])
            result = orchestrate(req)

            assert result.get("ok"), f"Failed for '{prompt}': {result.get('error')}"
            assert result.get("intent") == "pipeline", f"Expected pipeline intent for '{prompt}'"

            steps = [s.get("uses") for s in result["plan"]["steps"]]
            assert "input/file-before" in steps, f"Prompt '{prompt}' should use input/file-before. Steps: {steps}"
            assert "input/file-after" in steps, f"Prompt '{prompt}' should use input/file-after. Steps: {steps}"
            assert "algorithms/change-detection" in steps, (
                f"Prompt '{prompt}' should use change detection. Steps: {steps}"
            )


@pytest.mark.llm
class TestOrchestratorPlanOrder:
    """Tests for plan execution order.

    These tests require LLM access for context analysis.
    """

    def test_plan_execution_order(self, mock_attachment: Attachment) -> None:
        """Test that plan steps are in correct execution order."""
        req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[mock_attachment])
        result = orchestrate(req)

        assert result.get("ok")
        assert result.get("intent") == "pipeline"

        steps = [s.get("uses") for s in result["plan"]["steps"]]

        # Should be: input → processing → postprocessing
        assert steps[0].startswith("input/"), "First should be input"

        processing_idx = next((i for i, s in enumerate(steps) if "algorithms/" in s or "models/" in s), None)
        assert processing_idx is not None and processing_idx > 0, "Processing should come after input"

        postproc_idx = next((i for i, s in enumerate(steps) if "post-processing/" in s), None)
        if postproc_idx is not None:
            assert postproc_idx > processing_idx, "Postprocessing should come after processing"
