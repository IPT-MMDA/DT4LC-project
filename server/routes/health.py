"""Health, capabilities, models, and metrics endpoints."""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from dta.config import CACHE_PATH, MODELS_PATH, UPLOADS_PATH
from dta.dti.coe.llm.config import get_default_config
from dta.dti.coe.llm.router import LLMRouter
from dta.dti.data_sources.gee_common import is_initialized
from dta.dti.metrics import get_metrics_collector
from dta.dti.models.registry import get_model_registry
from dta.dti.registry import load_registry
from server.schemas import (
    DiskEntry,
    DiskUsage,
    GEEStatus,
    HealthResponse,
    LLMProviderStatus,
    ModelEntry,
    ModelsInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["health"])

_LOCALHOST_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _dir_size_bytes(p: Path) -> int:
    """Recursively sum file sizes using os.scandir for minimal syscall overhead."""
    if not p.exists():
        return 0
    total = 0
    with os.scandir(p) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _dir_size_bytes(Path(entry.path))
    return total


def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b //= 1024
    return f"{b:.1f} TB"


def _collect_llm_providers() -> list[LLMProviderStatus]:
    try:
        cfg = get_default_config()
        router = LLMRouter.from_config(cfg)
        return [LLMProviderStatus(name=p.name, model=p.model, available=p.is_available()) for p in router.providers]
    except (ImportError, ValueError, OSError) as exc:
        logger.warning("Could not collect LLM provider info: %s", exc)
        return [LLMProviderStatus(error=str(exc))]


def _collect_gee_status() -> GEEStatus:
    sa_configured = bool(os.environ.get("GEE_SERVICE_ACCOUNT_KEY"))
    if not is_initialized():
        return GEEStatus(initialized=False, service_account_configured=sa_configured)

    try:
        import ee

        ee.Number(1).getInfo()
        return GEEStatus(initialized=True, service_account_configured=sa_configured)
    except Exception as exc:
        logger.warning("GEE connection check failed: %s", exc)
        return GEEStatus(initialized=False, service_account_configured=sa_configured, error=str(exc))


def _collect_models_info() -> ModelsInfo:
    try:
        registry = get_model_registry()
        all_ids = registry.list_all()
        entries = [ModelEntry(id=mid, available=registry.get(mid).is_available()) for mid in all_ids]

        try:
            yaml_reg = load_registry()
            hosted = [i for i in yaml_reg.instances if i.kind == "model" and i.integration]
            entries += [
                ModelEntry(id=i.id, available=i.integration.status == "active" if i.integration else False)
                for i in hosted
            ]
            hosted_available = sum(1 for h in hosted if h.integration and h.integration.status == "active")
        except Exception:
            hosted = []
            hosted_available = 0

        return ModelsInfo(
            total=len(all_ids) + len(hosted),
            available=len(registry.list_available()) + hosted_available,
            models=entries,
        )
    except Exception as exc:
        return ModelsInfo(error=str(exc))


def _collect_disk_usage() -> DiskUsage:
    try:
        exports_dir = CACHE_PATH / "gee_exports"
        entries = {
            "uploads": UPLOADS_PATH,
            "cache": CACHE_PATH,
            "models": MODELS_PATH,
            "exports": exports_dir,
        }
        disk_entries = {
            key: DiskEntry(bytes=_dir_size_bytes(p), human=_fmt_bytes(_dir_size_bytes(p)))
            for key, p in entries.items()
        }
        return DiskUsage(**disk_entries)
    except Exception as exc:
        return DiskUsage(error=str(exc))


@router.get("/health", response_model=HealthResponse)  # type: ignore[misc]
async def health(
    request: Request,
    detailed: bool = Query(False, description="Return extended diagnostics (localhost only)"),
) -> HealthResponse:
    """Health check endpoint.

    Without ?detailed=true returns a simple liveness response.
    With ?detailed=true returns internal diagnostics; restricted to localhost.
    """
    if not detailed:
        return HealthResponse()

    client_host = request.client.host if request.client else ""
    if client_host not in _LOCALHOST_HOSTS:
        raise HTTPException(status_code=403, detail="Detailed diagnostics are available from localhost only")

    return HealthResponse(
        llm_providers=_collect_llm_providers(),
        gee=_collect_gee_status(),
        models=_collect_models_info(),
        disk=_collect_disk_usage(),
    )


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
