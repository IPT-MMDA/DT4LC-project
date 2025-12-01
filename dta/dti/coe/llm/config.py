"""LLM Configuration Management."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import yaml

from .router import LLMRouter

# Track model rotation state
_model_rotation_index: dict[str, int] = {}


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


def _is_provider_enabled(provider: str) -> bool:
    """Check if a provider is enabled via environment variable.

    Args:
        provider: Provider name (gemini, groq, ollama)

    Returns:
        True if enabled (default: True)
    """
    env_var = f"LLM_ENABLE_{provider.upper()}"
    value = os.environ.get(env_var, "true").lower()
    return value in ("true", "1", "yes", "on")


def _get_provider_order() -> list[str]:
    """Get provider priority order from environment.

    Returns:
        List of provider names in priority order
    """
    order_str = os.environ.get("LLM_PROVIDER_ORDER", "gemini,groq,ollama")
    return [p.strip().lower() for p in order_str.split(",") if p.strip()]


def _get_models_list(env_var: str, default: str) -> list[str]:
    """Parse comma-separated model list from environment variable.

    Args:
        env_var: Environment variable name
        default: Default value if not set

    Returns:
        List of model names
    """
    models_str = os.environ.get(env_var, default)
    return [m.strip() for m in models_str.split(",") if m.strip()]


def _get_next_model(provider: str, models: list[str]) -> str:
    """Get next model in rotation for a provider.

    Uses round-robin rotation across models.

    Args:
        provider: Provider name (gemini, groq, ollama)
        models: List of available models

    Returns:
        Next model to use
    """
    global _model_rotation_index

    if len(models) == 1:
        return models[0]

    # Initialize index if needed
    if provider not in _model_rotation_index:
        _model_rotation_index[provider] = random.randint(0, len(models) - 1)

    # Get current model and advance index
    idx = _model_rotation_index[provider]
    model = models[idx]
    _model_rotation_index[provider] = (idx + 1) % len(models)

    return model


def _get_provider_config(provider: str) -> dict[str, Any] | None:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (gemini, groq, ollama)

    Returns:
        Provider config dict or None if not available
    """
    if provider == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            models = _get_models_list("GEMINI_MODELS", "gemini-2.0-flash-exp")
            model = _get_next_model("gemini", models)
            return {"type": "gemini", "model": model}
    elif provider == "groq":
        if os.environ.get("GROQ_API_KEY"):
            model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
            return {"type": "groq", "model": model}
    elif provider == "ollama":
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        return {
            "type": "ollama",
            "model": ollama_model,
            "base_url": ollama_url,
        }
    elif provider == "apertus":
        # Apertus requires explicit opt-in (heavy local model)
        if os.environ.get("APERTUS_ENABLED", "false").lower() in ("true", "1", "yes", "on"):
            model = os.environ.get("APERTUS_MODEL", "swiss-ai/Apertus-8B")
            device = os.environ.get("APERTUS_DEVICE", "0")  # GPU device index or "cpu"
            dtype = os.environ.get("APERTUS_DTYPE", "bfloat16")
            return {
                "type": "apertus",
                "model": model,
                "device": int(device) if device.isdigit() else device,
                "dtype": dtype,
            }
    return None


def get_default_config() -> dict[str, Any]:
    """Get LLM configuration from environment.

    Respects the following environment variables:
    - LLM_ENABLE_GEMINI: Enable/disable Gemini (default: true)
    - LLM_ENABLE_GROQ: Enable/disable Groq (default: true)
    - LLM_ENABLE_OLLAMA: Enable/disable Ollama (default: true)
    - LLM_PROVIDER_ORDER: Priority order (default: gemini,groq,ollama)
    - LLM_STRATEGY: Routing strategy (default: fallback)

    Returns:
        Configuration dict with enabled providers in priority order
    """
    providers = []
    provider_order = _get_provider_order()

    for provider in provider_order:
        # Skip if provider is disabled
        if not _is_provider_enabled(provider):
            continue

        # Get provider config (checks API keys, etc.)
        config = _get_provider_config(provider)
        if config:
            providers.append(config)

    # Get routing strategy
    strategy = os.environ.get("LLM_STRATEGY", "fallback")

    return {
        "providers": providers,
        "strategy": strategy,
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
