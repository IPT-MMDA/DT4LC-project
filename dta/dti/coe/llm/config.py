"""LLM Configuration Management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .router import LLMRouter


def load_llm_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load LLM configuration from YAML file.

    Args:
        config_path: Path to config file (default: dta/config/llm.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        from dta.config import ROOT_DIR

        config_path = ROOT_DIR / "dta" / "config" / "llm.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        # Return default config
        return get_default_config()

    with config_path.open() as f:
        return yaml.safe_load(f)


def get_default_config() -> dict[str, Any]:
    """Get default LLM configuration.

    Priority order:
    1. Gemini (if API key available)
    2. Ollama llama3.2 (if available)

    Returns:
        Default configuration dict
    """
    providers = []

    # Add Gemini if API key present
    if os.environ.get("GEMINI_API_KEY"):
        providers.append({"type": "gemini", "model": "gemini-2.0-flash-exp"})

    # Add Ollama as fallback
    providers.append(
        {
            "type": "ollama",
            "model": "llama3.2",
            "base_url": "http://localhost:11434",
        }
    )

    return {
        "providers": providers,
        "strategy": "fallback",  # Try in order, fallback on failure
    }


def create_router_from_env() -> LLMRouter:
    """Create LLM router from environment/config.

    Checks for config file, falls back to environment-based defaults.

    Returns:
        Configured LLM router
    """
    config = load_llm_config()
    return LLMRouter.from_config(config)


def save_llm_config(config: dict[str, Any], config_path: Path | str | None = None) -> None:
    """Save LLM configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save to (default: dta/config/llm.yaml)
    """
    if config_path is None:
        from dta.config import ROOT_DIR

        config_path = ROOT_DIR / "dta" / "config" / "llm.yaml"

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False)
