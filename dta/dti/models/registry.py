"""Model Registry for DTA.

Provides unified interface for loading and managing models.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BaseModel(Protocol):
    """Protocol for all models.

    All model wrappers must implement this interface.
    """

    @property
    def name(self) -> str:
        """Model name."""
        ...

    @property
    def version(self) -> str:
        """Model version."""
        ...

    @property
    def required_inputs(self) -> list[str]:
        """Required input types."""
        ...

    @property
    def outputs(self) -> list[str]:
        """Output types."""
        ...

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run prediction.

        Args:
            inputs: Input data

        Returns:
            Prediction results
        """
        ...

    def is_available(self) -> bool:
        """Check if model is available for use.

        Returns:
            True if model can be loaded
        """
        ...


class ModelRegistry:
    """Registry for managing multiple models."""

    def __init__(self) -> None:
        """Initialize model registry."""
        self._models: dict[str, BaseModel] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        model: BaseModel,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a model.

        Args:
            model: Model instance
            metadata: Optional metadata (GPU requirements, memory, etc.)
        """
        model_id = f"{model.name}:{model.version}"
        self._models[model_id] = model

        if metadata:
            self._metadata[model_id] = metadata
        else:
            self._metadata[model_id] = {}

        logger.info(f"Registered model: {model_id}")

    def get(self, model_id: str) -> BaseModel:
        """Get model by ID.

        Args:
            model_id: Model identifier (name:version)

        Returns:
            Model instance

        Raises:
            KeyError: If model not found
        """
        if model_id not in self._models:
            # Try without version
            matches = [mid for mid in self._models.keys() if mid.startswith(f"{model_id}:")]
            if matches:
                model_id = matches[0]  # Use first match
            else:
                raise KeyError(f"Model not found: {model_id}")

        return self._models[model_id]

    def list_available(self) -> list[str]:
        """List all available models.

        Returns:
            List of model IDs
        """
        return [model_id for model_id, model in self._models.items() if model.is_available()]

    def list_all(self) -> list[str]:
        """List all registered models (including unavailable).

        Returns:
            List of model IDs
        """
        return list(self._models.keys())

    def check_requirements(self, model_id: str) -> dict[str, Any]:
        """Get model requirements.

        Args:
            model_id: Model identifier

        Returns:
            Requirements dict
        """
        model = self.get(model_id)
        metadata = self._metadata.get(model_id, {})

        # Get missing requirements if available
        missing_requirements: list[str] = []
        if hasattr(model, "get_missing_requirements"):
            missing_requirements = model.get_missing_requirements()

        return {
            "model_id": model.name,
            "name": metadata.get("display_name", model.name),
            "version": model.version,
            "description": metadata.get("description", ""),
            "author": metadata.get("author", ""),
            "source_url": metadata.get("source_url", ""),
            "inputs": model.required_inputs,
            "outputs": model.outputs,
            "available": model.is_available(),
            "missing_requirements": missing_requirements,
            "gpu_required": metadata.get("gpu_required", False),
            "memory_mb": metadata.get("memory_mb", 0),
            "latency_ms": metadata.get("latency_ms", 0),
        }

    def get_metadata(self, model_id: str) -> dict[str, Any]:
        """Get model metadata.

        Args:
            model_id: Model identifier

        Returns:
            Metadata dict
        """
        return self._metadata.get(model_id, {})


# Global registry
_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry.

    Returns:
        Model registry instance
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
        _initialize_default_models()
    return _model_registry


def _initialize_default_models() -> None:
    """Initialize default models.

    Note: Models (Prithvi, Delineate-Anything) are now managed through ModelManager
    and downloaded on-demand. They are executed via the registry.yaml configuration
    and the pipeline executor. See:
    - dta/dti/models/model_manager.py (download management)
    - dta/dti/models/third_party/ (inference wrappers)
    - dta/registry.yaml (pipeline configuration)
    """
    # No default models to register - all models are managed via ModelManager
    # and executed via registry.yaml configuration
    pass
