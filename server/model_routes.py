"""Model management API routes.

Provides endpoints for listing, downloading, and deleting ML models.
Downloads are non-blocking and run in background threads. Listing and status are open;
download and cancel need a role ``MODEL_DOWNLOAD_ON_DEMAND`` allows, and delete needs an
operator. Otherwise: 403 ``MODEL_GATED`` with ``details: {model_id}``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from dta.dti.executor import ModelGatedError, ModelNotInstalledError
from dta.dti.models import ModelStatus, get_model_manager

from .schemas import (
    ErrorDetail,
    ErrorResponse,
    MLModelActionResponse,
    MLModelDetailResponse,
    MLModelDownloadResponse,
    MLModelsListResponse,
)
from .sessions import ROLES, CurrentSession, Session, can_download_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ml-models", tags=["models"])

GATED = {403: {"model": ErrorResponse, "description": "MODEL_GATED"}}


def _gated(model_id: str, message: str) -> JSONResponse:
    error = ErrorDetail(code="MODEL_GATED", message=message, details={"model_id": model_id})
    return JSONResponse(ErrorResponse(error=error).model_dump(mode="json"), status_code=403)


def _download_refusal(model_info: dict[str, Any], session: Session) -> JSONResponse | None:
    """403 ``MODEL_GATED`` unless ``MODEL_DOWNLOAD_ON_DEMAND`` lets the caller's role download,
    worded as a job's refusal (``server.jobs``): it names the role, unless nobody may download."""
    if can_download_models(session.role):
        return None
    args = ("", model_info["id"], model_info["name"], model_info["size_mb"], "")
    downloads_off = not any(can_download_models(r) for r in ROLES)
    error = ModelNotInstalledError(*args) if downloads_off else ModelGatedError(*args, role=session.role)
    return _gated(model_info["id"], str(error))


@router.get(  # type: ignore[misc]
    "",
    response_model=MLModelsListResponse,
    summary="List ML models",
)
async def list_models() -> MLModelsListResponse:
    """List downloadable ML models and cache usage."""
    try:
        manager = get_model_manager()
        models = manager.list_models()
        total_size_mb = sum(m["size_mb"] for m in models if m["status"] == ModelStatus.AVAILABLE.value)

        return MLModelsListResponse(
            models=models,
            cache_dir=str(manager.cache_dir),
            total_installed_mb=total_size_mb,
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}") from e


@router.get(  # type: ignore[misc]
    "/{model_id}",
    response_model=MLModelDetailResponse,
    summary="Get ML model details",
)
async def get_model(model_id: str) -> MLModelDetailResponse:
    """Return metadata and download progress for one model."""
    try:
        manager = get_model_manager()
        model_info = manager.get_model_info(model_id)

        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        return MLModelDetailResponse.model_validate(model_info)  # type: ignore[no-any-return]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model: {e}") from e


@router.post(  # type: ignore[misc]
    "/{model_id}/download",
    status_code=202,
    response_model=MLModelDownloadResponse,
    summary="Download ML model",
    responses=GATED,
)
async def start_download(model_id: str, session: CurrentSession) -> MLModelDownloadResponse | JSONResponse:
    """Start a background download of the model weights (403 ``MODEL_GATED`` for a role that may not)."""
    try:
        manager = get_model_manager()

        model_info = manager.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        if refusal := _download_refusal(model_info, session):
            return refusal

        current_status = manager.get_model_status(model_id)
        if current_status == ModelStatus.DOWNLOADING:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already downloading")
        if current_status == ModelStatus.AVAILABLE:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already installed")

        progress = manager.start_download(model_id)
        logger.info(f"Started download for model {model_id}")

        return MLModelDownloadResponse(
            model_id=model_id,
            status=progress.status.value,
            message=f"Download started for {model_info['name']}",
            size_mb=model_info["size_mb"],
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to start download for {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start download: {e}") from e


@router.post(  # type: ignore[misc]
    "/{model_id}/cancel",
    response_model=MLModelActionResponse,
    summary="Cancel model download",
    responses=GATED,
)
async def cancel_download(model_id: str, session: CurrentSession) -> MLModelActionResponse | JSONResponse:
    """Cancel an in-progress model download (403 ``MODEL_GATED`` for a role that may not download)."""
    try:
        manager = get_model_manager()

        model_info = manager.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        if refusal := _download_refusal(model_info, session):
            return refusal

        if manager.get_model_status(model_id) != ModelStatus.DOWNLOADING:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not currently downloading")

        cancelled = manager.cancel_download(model_id)
        if not cancelled:
            raise HTTPException(status_code=400, detail="Failed to cancel download")

        return MLModelActionResponse.model_validate(
            {
                "model_id": model_id,
                "status": "cancelling",
                "message": "Download cancellation requested",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel download for {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel download: {e}") from e


@router.delete(  # type: ignore[misc]
    "/{model_id}",
    response_model=MLModelActionResponse,
    summary="Delete ML model",
    responses=GATED,
)
async def delete_model(model_id: str, session: CurrentSession) -> MLModelActionResponse | JSONResponse:
    """Remove an installed model from the local cache (operators only; else 403 ``MODEL_GATED``)."""
    try:
        manager = get_model_manager()

        model_info = manager.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        if not session.operator:
            return _gated(model_id, f"Only an operator can delete {model_info['name']} from this server.")

        if manager.get_model_status(model_id) != ModelStatus.AVAILABLE:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not installed")

        deleted = manager.delete_model(model_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete model")

        logger.info(f"Deleted model {model_id}")
        return MLModelActionResponse.model_validate(
            {
                "model_id": model_id,
                "status": "deleted",
                "message": f"Model '{model_info['name']}' has been deleted",
                "freed_mb": model_info["size_mb"],
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}") from e
