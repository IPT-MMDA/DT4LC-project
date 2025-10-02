"""Tests for planner data loader fix.

Ensures that both template and LLM planners always include
the input/file data loader step when needed.
"""

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.schemas import ChatRequest


def test_orchestration_includes_data_loader() -> None:
    """Test that orchestration includes input/file step."""
    req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
    result = orchestrate(req)

    assert result.get("ok"), f"Orchestration failed: {result.get('error')}"

    plan = result.get("plan", {})
    steps = [s.get("uses") for s in plan.get("steps", [])]

    # Should have at least 2 steps (data loader + processing)
    assert len(steps) >= 2, f"Plan too short: {steps}"

    # First step should be data loader
    assert steps[0] == "input/file", f"First step should be input/file, got: {steps[0]}"

    # Should include some processing step
    processing_steps = [s for s in steps if "algorithms/" in s or "models/" in s]
    assert len(processing_steps) > 0, f"No processing steps found in: {steps}"


def test_statistics_plan_includes_data_loader() -> None:
    """Test statistics request includes data loader."""
    req = ChatRequest(prompt="calculate statistics on kahovka", attachments=[])
    result = orchestrate(req)

    assert result.get("ok"), f"Orchestration failed: {result.get('error')}"

    steps = [s.get("uses") for s in result["plan"]["steps"]]
    assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"
    assert "algorithms/statistics" in steps, f"Should include statistics step: {steps}"


def test_vegetation_analysis_includes_data_loader() -> None:
    """Test vegetation analysis includes data loader."""
    req = ChatRequest(prompt="analyze vegetation changes", attachments=[])
    result = orchestrate(req)

    assert result.get("ok"), f"Orchestration failed: {result.get('error')}"

    steps = [s.get("uses") for s in result["plan"]["steps"]]
    assert steps[0] == "input/file", f"First step should be input/file, got: {steps}"


def test_various_prompts_include_data_loader() -> None:
    """Test that various prompts all include data loader."""
    test_prompts = [
        "compute ndvi",
        "get statistics",
        "analyze land cover",
        "detect changes in kahovka",
    ]

    for prompt in test_prompts:
        req = ChatRequest(prompt=prompt, attachments=[])
        result = orchestrate(req)

        assert result.get("ok"), f"Failed for '{prompt}': {result.get('error')}"

        steps = [s.get("uses") for s in result["plan"]["steps"]]
        assert steps[0] == "input/file", f"Prompt '{prompt}' missing data loader. Steps: {steps}"


def test_plan_execution_order() -> None:
    """Test that plan steps are in correct execution order."""
    req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
    result = orchestrate(req)

    assert result.get("ok")

    steps = [s.get("uses") for s in result["plan"]["steps"]]

    # Should be: input → processing → postprocessing
    assert steps[0].startswith("input/"), "First should be input"

    # Find processing step position
    processing_idx = next((i for i, s in enumerate(steps) if "algorithms/" in s or "models/" in s), None)
    assert processing_idx is not None and processing_idx > 0, "Processing should come after input"

    # Find postprocessing step position
    postproc_idx = next((i for i, s in enumerate(steps) if "post-processing/" in s), None)
    if postproc_idx is not None:
        assert postproc_idx > processing_idx, "Postprocessing should come after processing"
