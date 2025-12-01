"""Tests for orchestrator and data loader.

Tests orchestration flow, data loader inclusion, plan execution order,
and intent classification integration.
"""

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.schemas import ChatRequest


class TestOrchestratorIntentClassification:
    """Tests for orchestrator intent classification."""

    def test_pipeline_intent_returns_plan(self) -> None:
        """Test that pipeline intent requests return a plan."""
        req = ChatRequest(prompt="calculate ndvi", attachments=[])
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


class TestOrchestratorDataLoader:
    """Tests for orchestrator data loader inclusion."""

    def test_orchestration_includes_data_loader(self) -> None:
        """Test that orchestration includes input/file step."""
        req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline", f"Expected pipeline intent, got: {result.get('intent')}"

        plan = result.get("plan", {})
        steps = [s.get("uses") for s in plan.get("steps", [])]

        assert len(steps) >= 2, f"Plan too short: {steps}"

        assert steps[0] == "input/file", f"First step should be input/file, got: {steps[0]}"

        processing_steps = [s for s in steps if "algorithms/" in s or "models/" in s]
        assert len(processing_steps) > 0, f"No processing steps found in: {steps}"

    def test_statistics_plan_includes_data_loader(self) -> None:
        """Test statistics request includes data loader."""
        req = ChatRequest(prompt="calculate statistics on kahovka", attachments=[])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline"

        steps = [s.get("uses") for s in result["plan"]["steps"]]
        assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"
        assert "algorithms/statistics" in steps, f"Should include statistics step: {steps}"

    def test_vegetation_analysis_includes_data_loader(self) -> None:
        """Test vegetation analysis includes data loader."""
        req = ChatRequest(prompt="analyze vegetation health", attachments=[])
        result = orchestrate(req)

        assert result.get("ok"), f"Orchestration failed: {result.get('error')}"
        assert result.get("intent") == "pipeline"

        steps = [s.get("uses") for s in result["plan"]["steps"]]
        assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"


class TestOrchestratorPrompts:
    """Tests for various prompts."""

    def test_various_prompts_include_data_loader(self) -> None:
        """Test that various prompts all include appropriate data loader."""
        single_file_prompts = [
            "compute ndvi",
            "get statistics",
            "analyze land cover",
        ]

        for prompt in single_file_prompts:
            req = ChatRequest(prompt=prompt, attachments=[])
            result = orchestrate(req)

            assert result.get("ok"), f"Failed for '{prompt}': {result.get('error')}"
            assert result.get("intent") == "pipeline", f"Expected pipeline intent for '{prompt}'"

            steps = [s.get("uses") for s in result["plan"]["steps"]]
            assert steps[0] == "input/file", f"Prompt '{prompt}' missing data loader. Steps: {steps}"


class TestOrchestratorChangeDetection:
    """Tests for change detection orchestration."""

    def test_change_detection_uses_dual_input(self) -> None:
        """Test that change detection prompts use dual file input."""
        change_prompts = [
            "detect changes in kahovka",
            "compare before and after images",
        ]

        for prompt in change_prompts:
            req = ChatRequest(prompt=prompt, attachments=[])
            result = orchestrate(req)

            assert result.get("ok"), f"Failed for '{prompt}': {result.get('error')}"
            assert result.get("intent") == "pipeline", f"Expected pipeline intent for '{prompt}'"

            steps = [s.get("uses") for s in result["plan"]["steps"]]
            assert "input/file-before" in steps, f"Prompt '{prompt}' should use input/file-before. Steps: {steps}"
            assert "input/file-after" in steps, f"Prompt '{prompt}' should use input/file-after. Steps: {steps}"
            assert "algorithms/change-detection" in steps, (
                f"Prompt '{prompt}' should use change detection. Steps: {steps}"
            )


class TestOrchestratorPlanOrder:
    """Tests for plan execution order."""

    def test_plan_execution_order(self) -> None:
        """Test that plan steps are in correct execution order."""
        req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
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
