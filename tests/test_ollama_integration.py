"""Integration tests for Ollama LLM provider.

Tests the Ollama integration with improved prompts to ensure correct JSON formatting.
"""

import pytest

from dta.dti.coe.context_agent import analyze
from dta.dti.schemas import ChatRequest


@pytest.mark.asyncio
async def test_ollama_context_analysis_format() -> None:
    """Test that Ollama returns correctly formatted arrays (not objects)."""
    req = ChatRequest(prompt="calculate ndvi on kahovka data", attachments=[])
    registry_types = ["Raster", "Features", "NDVIMap", "ChangeMap", "Statistics", "Insights"]

    result = analyze(req, registry_types)

    # Verify structure
    assert hasattr(result, "goal")
    assert hasattr(result, "desired_outputs")
    assert hasattr(result, "required_inputs")
    assert hasattr(result, "hints")

    # Critical: desired_outputs and required_inputs must be lists of strings
    assert isinstance(result.desired_outputs, list)
    assert isinstance(result.required_inputs, list)

    # All items must be strings (not dicts or objects)
    for output in result.desired_outputs:
        assert isinstance(output, str), f"desired_outputs contains non-string: {output}"

    for inp in result.required_inputs:
        assert isinstance(inp, str), f"required_inputs contains non-string: {inp}"

    # Verify goal is a string
    assert isinstance(result.goal, str)
    assert len(result.goal) > 0


@pytest.mark.asyncio
async def test_ollama_various_prompts() -> None:
    """Test Ollama format consistency across different prompts."""
    registry_types = ["Raster", "Features", "NDVIMap", "ChangeMap", "Statistics"]

    test_cases = [
        "calculate ndvi",
        "analyze vegetation changes",
        "load raster and compute statistics",
        "detect land cover changes",
    ]

    for prompt in test_cases:
        req = ChatRequest(prompt=prompt, attachments=[])
        result = analyze(req, registry_types)

        # All outputs must be string arrays
        assert all(isinstance(x, str) for x in result.desired_outputs), f"Failed for prompt: {prompt}"
        assert all(isinstance(x, str) for x in result.required_inputs), f"Failed for prompt: {prompt}"


def test_ollama_available() -> None:
    """Test that Ollama provider is available and configured."""
    from dta.dti.coe.llm.config import create_router_from_env

    router = create_router_from_env()

    # Should have at least one provider
    assert len(router.providers) > 0

    # At least one provider should be available
    available = [p for p in router.providers if p.is_available()]
    assert len(available) > 0, "No LLM providers available"

    # Check if Ollama is configured
    ollama_providers = [p for p in router.providers if p.name == "ollama"]
    if ollama_providers:
        assert ollama_providers[0].is_available(), "Ollama configured but not available"
