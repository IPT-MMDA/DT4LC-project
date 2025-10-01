"""Post-processing module for DTA.

Transforms raw algorithm outputs into rich, user-friendly formats:
- Visualizations (PNG, charts)
- GeoJSON for web maps
- LLM-powered insights
- Statistical summaries
"""

from .insights import InsightGenerator, format_statistics
from .visualization import Visualizer

__all__ = [
    "Visualizer",
    "InsightGenerator",
    "format_statistics",
]
