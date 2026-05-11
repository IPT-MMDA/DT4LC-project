"""Tests for LLM planner logic."""

from unittest.mock import MagicMock, patch

import pytest

from dta.dti.coe.context_agent import ContextUnderstanding
from dta.dti.coe.llm_planner import _parse_plan_json, _validate_plan, plan_with_llm
from dta.dti.executor import PlanStep
from dta.dti.schemas import Registry, RegistryItem, Runner


def test_parse_plan_json() -> None:
    """Test parsing JSON plan from LLM response."""
    # Valid JSON without markdown
    valid_json = '{"steps": [{"uses": "test"}], "reasoning": "test"}'
    res = _parse_plan_json(valid_json)
    assert len(res["steps"]) == 1

    # Valid JSON with markdown
    markdown_json = (
        "```json\n" + '{"steps": [{"uses": "comp-1", "binds": {"A": 1}}], "reasoning": "Markdown test"}' + "\n```"
    )
    res = _parse_plan_json(markdown_json)
    assert res["steps"][0]["uses"] == "comp-1"
    assert res["steps"][0]["binds"] == {"A": 1}

    # Missing steps field
    with pytest.raises(ValueError, match="Missing 'steps' field"):
        _parse_plan_json('{"reasoning": "bad"}')

    # Steps not a list
    with pytest.raises(ValueError, match="'steps' must be a list"):
        _parse_plan_json('{"steps": "notalist"}')

    # Step missing uses
    with pytest.raises(ValueError, match="missing 'uses' field"):
        _parse_plan_json('{"steps": [{"bad": "step"}]}')

    # Invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON"):
        _parse_plan_json("{bad json")


def test_validate_plan() -> None:
    """Test plan validation."""
    reg = Registry(
        version="1.0",
        types=[],
        instances=[
            RegistryItem(id="good/component", kind="algorithm", runner=Runner(type="python"), inputs=[], outputs=[])
        ],
    )

    # Empty plan
    with pytest.raises(ValueError, match="Plan has no steps"):
        _validate_plan([], reg)

    # Valid plan
    steps = [PlanStep(uses="good/component")]
    _validate_plan(steps, reg)  # Should not raise

    # Invalid plan
    steps = [PlanStep(uses="bad/component")]
    with pytest.raises(ValueError, match="not found in registry"):
        _validate_plan(steps, reg)


@patch("dta.dti.coe.llm_planner.get_llm_router")
def test_plan_with_llm(mock_get_router: MagicMock) -> None:
    """Test planning flow."""
    ctx = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[])
    reg = Registry(
        version="1.0",
        types=[],
        instances=[
            RegistryItem(id="test/component", kind="algorithm", runner=Runner(type="python"), inputs=[], outputs=[])
        ],
    )

    mock_router = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"steps": [{"uses": "test/component"}]}'
    mock_response.provider = "fake"
    mock_router.generate.return_value = mock_response
    mock_get_router.return_value = mock_router

    plan = plan_with_llm(ctx, reg)
    assert plan.flow == "test"
    assert len(plan.steps) == 1
    assert plan.steps[0].uses == "test/component"


@patch("dta.dti.coe.llm_planner.get_llm_router")
def test_plan_with_llm_failure(mock_get_router: MagicMock) -> None:
    """Test planning error handling."""
    ctx = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[])
    reg = Registry(version="1.0", types=[], instances=[])

    mock_router = MagicMock()
    mock_router.generate.side_effect = Exception("Generation failed")
    mock_get_router.return_value = mock_router

    with pytest.raises(Exception, match="LLM planner failed: Generation failed"):
        plan_with_llm(ctx, reg)
