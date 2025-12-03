"""Insights generation module.

Uses LLMs to generate natural language insights from analysis results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dta.dti.coe.llm import get_llm_router

if TYPE_CHECKING:
    from dta.dti.coe.llm import LLMRouter

logger = logging.getLogger(__name__)


class InsightGenerator:
    """Generate natural language insights from analysis results."""

    def __init__(self, llm_router: LLMRouter | None = None) -> None:
        """Initialize insight generator.

        Args:
            llm_router: Optional LLM router for insight generation
        """
        self.llm_router = llm_router

    def _get_router(self) -> LLMRouter:
        """Get or create LLM router."""
        if self.llm_router is None:
            self.llm_router = get_llm_router()
        return self.llm_router

    def generate_ndvi_insights(self, ndvi_data: dict[str, Any]) -> str:
        """Generate insights from NDVI analysis.

        Args:
            ndvi_data: NDVI analysis results with statistics

        Returns:
            Natural language insights
        """
        try:
            router = self._get_router()

            # Extract key statistics
            stats = ndvi_data.get("statistics", {})
            mean_ndvi = stats.get("mean", 0)
            std_ndvi = stats.get("std", 0)
            min_ndvi = stats.get("min", -1)
            max_ndvi = stats.get("max", 1)

            # Build prompt
            from dta.dti.coe.llm import LLMMessage

            system_prompt = """You are an expert in remote sensing and vegetation analysis.
Provide concise, actionable insights from NDVI (Normalized Difference Vegetation Index) data.

NDVI interpretation:
- -1.0 to 0.0: Water, bare soil, or non-vegetated surfaces
- 0.0 to 0.2: Very sparse vegetation or stressed crops
- 0.2 to 0.4: Sparse vegetation, early growth, or degraded areas
- 0.4 to 0.6: Moderate vegetation health
- 0.6 to 0.8: Healthy, dense vegetation
- 0.8 to 1.0: Very dense, healthy vegetation (forests, crops at peak)

Keep your response to 3-5 sentences. Focus on practical interpretation."""

            user_prompt = f"""Analyze this NDVI data and provide insights:

**Statistics:**
- Mean NDVI: {mean_ndvi:.3f}
- Standard Deviation: {std_ndvi:.3f}
- Range: {min_ndvi:.3f} to {max_ndvi:.3f}

What does this tell us about the vegetation health and land cover?"""

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            response = router.generate(messages, temperature=0.3, max_tokens=300)
            return response.text

        except Exception as e:
            logger.warning(f"Failed to generate LLM insights: {e}")
            # Fallback to template-based insights
            return self._template_ndvi_insights(ndvi_data)

    def generate_change_insights(self, change_data: dict[str, Any]) -> str:
        """Generate insights from change detection analysis.

        Args:
            change_data: Change detection results with statistics

        Returns:
            Natural language insights
        """
        try:
            router = self._get_router()

            # Extract statistics
            stats = change_data.get("statistics", {})
            mean_change = stats.get("mean_change", 0)
            total_decrease = abs(stats.get("total_decrease", 0))
            total_increase = stats.get("total_increase", 0)

            # Build prompt
            from dta.dti.coe.llm import LLMMessage

            system_prompt = """You are an expert in environmental change detection and land cover analysis.
Provide concise insights from NDVI change detection data.

Change interpretation:
- Negative values: Vegetation decrease (deforestation, degradation, drought)
- Positive values: Vegetation increase (reforestation, crop growth, recovery)
- Near-zero: Stable conditions

Focus on the magnitude and implications of changes. Keep it to 3-5 sentences."""

            user_prompt = f"""Analyze this NDVI change data:

**Statistics:**
- Mean Change: {mean_change:.3f}
- Total Decrease: {total_decrease:.2f}
- Total Increase: {total_increase:.2f}

What changes occurred and what might have caused them?"""

            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]

            response = router.generate(messages, temperature=0.3, max_tokens=300)
            return response.text

        except Exception as e:
            logger.warning(f"Failed to generate LLM insights: {e}")
            return self._template_change_insights(change_data)

    def _template_ndvi_insights(self, ndvi_data: dict[str, Any]) -> str:
        """Fallback template-based NDVI insights."""
        stats = ndvi_data.get("statistics", {})
        mean_ndvi = stats.get("mean", 0)

        if mean_ndvi < 0.2:
            health = "sparse or stressed vegetation"
        elif mean_ndvi < 0.4:
            health = "moderate vegetation with room for improvement"
        elif mean_ndvi < 0.6:
            health = "healthy vegetation"
        else:
            health = "dense, thriving vegetation"

        return (
            f"The mean NDVI of {mean_ndvi:.3f} indicates {health}. "
            f"Values range from {stats.get('min', -1):.3f} to {stats.get('max', 1):.3f}, "
            f"showing variability across the area."
        )

    def _template_change_insights(self, change_data: dict[str, Any]) -> str:
        """Fallback template-based change insights."""
        stats = change_data.get("statistics", {})
        mean_change = stats.get("mean_change", 0)
        total_decrease = abs(stats.get("total_decrease", 0))
        total_increase = stats.get("total_increase", 0)

        if abs(mean_change) < 0.05:
            trend = "relatively stable"
        elif mean_change < 0:
            trend = "declining vegetation health"
        else:
            trend = "improving vegetation health"

        return (
            f"The area shows {trend} with a mean change of {mean_change:.3f}. "
            f"Total decrease: {total_decrease:.2f}, Total increase: {total_increase:.2f}."
        )


def format_statistics(stats: dict[str, Any]) -> str:
    """Format statistics as human-readable text.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted string
    """
    lines = ["**Statistics:**\n"]

    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"- {key.replace('_', ' ').title()}: {value:.4f}")
        elif isinstance(value, int):
            lines.append(f"- {key.replace('_', ' ').title()}: {value:,}")
        else:
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)
