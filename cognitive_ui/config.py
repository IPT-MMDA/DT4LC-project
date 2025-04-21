"""
Configuration module for the Cognitive Digital Twin UI.

This module contains configuration constants and settings for the application.
"""

from pathlib import Path

# --- LLM Settings ---
MAX_TOKENS = 1000
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-001"
DEFAULT_GEMINI_MAX_TOKENS: int | None = None

# --- Visualization ---
VIZ_NO_DATA: float = -9999  # No data value for visualizations
VIZ_NO_DATA_FLOAT = 0.0001
VIZ_PERCENTILES = (0.1, 99.9)

# --- Data Settings ---
DEFAULT_HISTORICAL_TIMESTAMP = "2017-01-01"

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent

RESOURCES_PATH = ROOT_DIR / "resources"
RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

CACHE_DIR = RESOURCES_PATH / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- UI Settings ---
UI_MAX_TEXT_LENGTH = {
    "interpretation": 800,
    "causal": 800,
    "interventions": 1000,
    "query": 1200,
    "synthesis": 800,
    "uncertainty": 800,
}
UI_TABS = {
    "DATASET_ANALYSIS": "🛰️ Dataset Analysis",
    "CHANGE_ANALYSIS": "📊 Change Analysis",
    "PROBLEM_SOLVING": "❓ Problem Solving & Queries",
}

# --- Query Templates ---
SYNTHESIS_QUERY: str = (
    "How might the observed environmental patterns affect the hydrological cycle, agricultural productivity, "
    "and climate resilience in this region?"
)
UNCERTAINTY_QUERY: str = (
    "What is the confidence level in the identified land cover changes, and what alternative explanations "
    "should be considered?"
)
