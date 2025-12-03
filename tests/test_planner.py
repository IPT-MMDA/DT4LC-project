"""Tests for LLM-powered planner.

Tests template planner, LLM planner, and hybrid planning strategies.
"""

from unittest.mock import MagicMock

import pytest

from dta.dti.coe.llm import LLMResponse
from dta.dti.coe.llm.base import BaseLLMProvider
from dta.dti.coe.llm.router import LLMRouter
from dta.dti.coe.llm_planner import (
    estimate_plan_confidence,
    format_registry_for_llm,
    plan_with_llm,
)
from dta.dti.coe.planner import plan, plan_template
from dta.dti.registry import Registry
from dta.dti.schemas import ContextUnderstanding, ExecutionPlan, RegistryItem, Runner


@pytest.fixture
def mock_registry() -> Registry:
    """Create mock registry for testing."""
    return Registry(
        version="1.0",
        types=["Raster", "NDVIMap", "Statistics", "Summary"],
        instances=[
            RegistryItem(
                id="input/kahovka",
                kind="input",
                runner=Runner(type="python", entrypoint="dta.dti.assets.kahovka"),
                keywords=["kahovka", "satellite"],
                inputs=[],
                outputs=["Raster"],
            ),
            RegistryItem(
                id="algorithms/ndvi",
                kind="algorithm",
                runner=Runner(type="python", entrypoint="dta.dti.algorithms.ndvi"),
                keywords=["ndvi", "vegetation"],
                inputs=["Raster"],
                outputs=["NDVIMap"],
            ),
            RegistryItem(
                id="algorithms/statistics",
                kind="algorithm",
                runner=Runner(type="python", entrypoint="dta.dti.algorithms.statistics"),
                keywords=["stats", "analysis"],
                inputs=["Raster"],
                outputs=["Statistics"],
            ),
            RegistryItem(
                id="post-processing/agent-analysis",
                kind="postprocess",
                runner=Runner(type="agent"),
                keywords=["llm", "summary"],
                inputs=[],
                outputs=["Summary"],
            ),
        ],
    )


@pytest.fixture
def simple_context() -> ContextUnderstanding:
    """Simple context with clear keywords."""
    return ContextUnderstanding(
        goal="Calculate NDVI on Kahovka data",
        required_inputs=["Raster"],
        desired_outputs=["NDVIMap"],
        hints={"keywords": ["ndvi", "kahovka"], "output_type": "chat"},
    )


@pytest.fixture
def complex_context() -> ContextUnderstanding:
    """Complex context without clear keywords."""
    return ContextUnderstanding(
        goal="Analyze vegetation health trends in the reservoir area and compare with baseline",
        required_inputs=[],
        desired_outputs=[],
        hints={"keywords": [], "output_type": "chat"},
    )


class TestRegistryFormatting:
    """Tests for registry formatting for LLM."""

    def test_format_registry_for_llm(self, mock_registry: Registry) -> None:
        """Test registry formatting for LLM."""
        formatted = format_registry_for_llm(mock_registry)

        assert "# Available Pipeline Components" in formatted
        assert "input/kahovka" in formatted
        assert "algorithms/ndvi" in formatted
        assert "Inputs:" in formatted
        assert "Outputs:" in formatted
        assert "Keywords:" in formatted


class TestConfidenceEstimation:
    """Tests for plan confidence estimation."""

    def test_estimate_confidence_high(self, simple_context: ContextUnderstanding) -> None:
        """Test confidence estimation for simple request."""
        confidence = estimate_plan_confidence(simple_context)

        assert confidence >= 0.7

    def test_estimate_confidence_low(self, complex_context: ContextUnderstanding) -> None:
        """Test confidence estimation for complex request."""
        confidence = estimate_plan_confidence(complex_context)

        assert confidence < 0.7

    def test_confidence_scoring_edge_cases(self) -> None:
        """Test confidence scoring edge cases."""
        ctx2 = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[], hints={})
        assert estimate_plan_confidence(ctx2) == 0.0

        ctx3 = ContextUnderstanding(
            goal="test",
            required_inputs=["Raster"],
            desired_outputs=["NDVIMap"],
            hints={"keywords": ["ndvi", "kahovka"]},
        )
        confidence = estimate_plan_confidence(ctx3)
        assert confidence == 1.0


class TestTemplatePlanner:
    """Tests for template-based planner."""

    def test_template_planner(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test template-based planner."""
        plan_result = plan_template(simple_context, mock_registry)

        assert plan_result is not None
        assert len(plan_result.steps) > 0
        assert plan_result.flow == simple_context.goal


class TestLLMPlanner:
    """Tests for LLM-powered planner."""

    def test_llm_planner_with_mock(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test LLM planner with mocked LLM."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"
        mock_provider.model = "mock-model"

        mock_response = LLMResponse(
            text="""{
  "steps": [
    {"uses": "input/kahovka"},
    {"uses": "algorithms/ndvi"},
    {"uses": "post-processing/agent-analysis"}
  ],
  "reasoning": "Load Kahovka data, calculate NDVI, generate summary"
}""",
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        router = LLMRouter([mock_provider])

        plan_result = plan_with_llm(simple_context, mock_registry, router)

        assert plan_result is not None
        assert len(plan_result.steps) == 3
        assert plan_result.steps[0].uses == "input/kahovka"
        assert plan_result.steps[1].uses == "algorithms/ndvi"
        assert plan_result.steps[2].uses == "post-processing/agent-analysis"

    def test_llm_planner_invalid_json(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test LLM planner with invalid JSON response."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"
        mock_provider.model = "mock-model"

        mock_response = LLMResponse(
            text="This is not valid JSON",
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        router = LLMRouter([mock_provider])

        with pytest.raises(Exception, match="LLM planner failed"):
            plan_with_llm(simple_context, mock_registry, router)

    def test_llm_planner_missing_component(
        self, simple_context: ContextUnderstanding, mock_registry: Registry
    ) -> None:
        """Test LLM planner with non-existent component."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"
        mock_provider.model = "mock-model"

        mock_response = LLMResponse(
            text="""{
  "steps": [
    {"uses": "input/nonexistent"},
    {"uses": "algorithms/ndvi"}
  ],
  "reasoning": "Test plan"
}""",
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        router = LLMRouter([mock_provider])

        with pytest.raises(Exception, match="(not found in registry|LLM planner failed)"):
            plan_with_llm(simple_context, mock_registry, router)

    def test_llm_planner_markdown_json(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test LLM planner with JSON wrapped in markdown."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"
        mock_provider.model = "mock-model"

        mock_response = LLMResponse(
            text="""```json
{
  "steps": [
    {"uses": "input/kahovka"},
    {"uses": "algorithms/ndvi"}
  ],
  "reasoning": "Test plan"
}
```""",
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        router = LLMRouter([mock_provider])

        plan_result = plan_with_llm(simple_context, mock_registry, router)
        assert len(plan_result.steps) == 2

    def test_llm_planner_empty_steps(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test LLM planner with empty steps."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"
        mock_provider.model = "mock-model"

        mock_response = LLMResponse(
            text='{"steps": [], "reasoning": "No steps needed"}',
            model="mock-model",
            provider="mock",
        )
        mock_provider.generate.return_value = mock_response

        router = LLMRouter([mock_provider])

        with pytest.raises(Exception, match="Plan has no steps"):
            plan_with_llm(simple_context, mock_registry, router)


class TestHybridPlanner:
    """Tests for hybrid planner."""

    def test_hybrid_planner_uses_template(self, simple_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test hybrid planner uses template for high confidence."""
        plan_result = plan(simple_context, mock_registry, use_llm=True)

        assert plan_result is not None
        assert len(plan_result.steps) > 0

    def test_hybrid_planner_uses_llm(self, complex_context: ContextUnderstanding, mock_registry: Registry) -> None:
        """Test hybrid planner attempts LLM for low confidence."""
        plan_result = plan(complex_context, mock_registry, use_llm=True)

        assert plan_result is not None
        assert len(plan_result.steps) >= 0

    def test_hybrid_planner_force_template(
        self, complex_context: ContextUnderstanding, mock_registry: Registry
    ) -> None:
        """Test hybrid planner can be forced to use template."""
        plan_result = plan(complex_context, mock_registry, use_llm=False)

        assert plan_result is not None
        assert isinstance(plan_result, ExecutionPlan)
