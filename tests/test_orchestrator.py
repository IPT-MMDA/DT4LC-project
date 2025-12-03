"""Tests for orchestrator and data loader.

Tests orchestration flow, data loader inclusion, plan execution order,
and intent classification integration.

Note: Some tests require LLM access (marked with @pytest.mark.llm).
These will fail if no LLM provider is available or rate limited.
Run with: pytest -m "not llm" to skip LLM-dependent tests.
"""

import pytest

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.schemas import Attachment, ChatRequest


# Fixture for a mock attachment
@pytest.fixture
def mock_attachment() -> Attachment:
    """Create a mock attachment for testing."""
    return Attachment(id="test", filename="test.tif", mime_type="image/tiff", path="/tmp/test.tif")


class TestOrchestratorIntentClassification:
    """Tests for orchestrator intent classification."""

    def test_pipeline_intent_returns_plan(self, mock_attachment: Attachment) -> None:
        """Test that pipeline intent requests with attachments return a plan."""
        req = ChatRequest(prompt="calculate ndvi", attachments=[mock_attachment])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline", f"Expected pipeline intent, got: {result.get('intent')}"
        assert "plan" in result, "Pipeline intent should include a plan"

    def test_conversation_intent_returns_response(self) -> None:
        """Test that conversation intent requests return a response."""
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
