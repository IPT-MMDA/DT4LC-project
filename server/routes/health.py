"""Health, capabilities, models, and metrics endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from dta.dti.metrics import get_metrics_collector
from dta.dti.models.registry import get_model_registry
from dta.dti.registry import load_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health")  # type: ignore[misc]
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"ok": True, "service": "DT4LC", "version": "1.0.0"}


@router.get("/capabilities")  # type: ignore[misc]
async def list_capabilities() -> JSONResponse:
    """List all available components from the registry.

    Returns models, algorithms, and other registered components.
    """
    try:
        registry = load_registry()
        return JSONResponse(
            {
                "version": registry.version,
                "types": registry.types,
                "instances": [item.model_dump() for item in registry.instances],
                "count": len(registry.instances),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load registry: {e}") from e


@router.get("/models")  # type: ignore[misc]
async def list_models() -> JSONResponse:
    """List all registered models from the model registry.

    Returns model information including requirements, availability,
    descriptions, author info, and source URLs.
    """
    try:
        registry = get_model_registry()
        models = []

        # List ALL models from Python registry
        for model_id in registry.list_all():
            req = registry.check_requirements(model_id)
            models.append(req)

        # Also include hosted models from YAML registry (models with integration field)
        try:
            yaml_registry = load_registry()
            for item in yaml_registry.instances:
                if item.kind == "model" and item.integration:
                    models.append(
                        {
                            "model_id": item.id,
                            "name": item.id.split("/")[-1].replace("-", " ").title(),
                            "description": item.description or "",
                            "author": item.metadata.get("author", ""),
                            "source_url": item.integration.url,
                            "available": item.integration.status == "active",
                            "missing_requirements": item.integration.requires
                            if item.integration.status == "planned"
                            else [],
                            "gpu_required": False,
                            "integration_type": item.integration.type,
                            "integration_status": item.integration.status,
                            "keywords": item.keywords,
                            "hosting": item.metadata.get("hosting", "external"),
                            "team": item.metadata.get("team", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to load hosted models from YAML registry: {e}")

        return JSONResponse({"models": models, "count": len(models)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}") from e


@router.get("/metrics")  # type: ignore[misc]
async def get_metrics() -> JSONResponse:
    """Get system metrics including execution and LLM stats."""
    try:
        collector = get_metrics_collector()
        stats = collector.get_stats()

        return JSONResponse(
            {
                "total_executions": stats.total_executions,
                "successful_executions": stats.successful_executions,
                "failed_executions": stats.failed_executions,
                "average_duration_seconds": stats.avg_execution_time,
                "total_llm_calls": stats.total_llm_calls,
                "total_llm_tokens": stats.total_llm_tokens,
                "total_llm_cost": stats.total_llm_cost,
                "llm_by_provider": stats.llm_by_provider,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}") from e
