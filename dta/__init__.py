"""Digital Twin Aggregator (DTA).

Provides runtime services used by the orchestrator:
- PipelineExecutor: runs a simple step graph
- DataAssetManager: resolves logical asset names to paths
- ModelRegistry: resolves logical model names to callables
- PostProcessor: handles visualization and summaries
"""

from .assets import DataAssetManager
from .executor import PipelineExecutor
from .models import ModelRegistry
from .post import PostProcessor
from .algorithms import Algorithms

__all__ = [
    "DataAssetManager",
    "PipelineExecutor",
    "ModelRegistry",
    "PostProcessor",
    "Algorithms",
]
