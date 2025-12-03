"""Model management API routes.

Provides endpoints for listing, downloading, and deleting ML models.
Downloads are non-blocking and run in background threads.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from dta.dti.models import ModelStatus, get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ml-models", tags=["models"])


@router.get("")
async def list_models() -> JSONResponse:
    """List all available ML models with their current status.

    Returns models that can be downloaded and used for inference,
    including download progress for models currently being downloaded.
    """
    try:
        manager = get_model_manager()
        models = manager.list_models()

        # Calculate total size of installed models
        total_size_mb = sum(m["size_mb"] for m in models if m["status"] == ModelStatus.AVAILABLE.value)

        return JSONResponse(
            {
                "models": models,
                "cache_dir": str(manager.cache_dir),
                "total_installed_mb": total_size_mb,
            }
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}") from e


@router.get("/{model_id}")
async def get_model(model_id: str) -> JSONResponse:
    """Get detailed information for a specific model.

    Includes download progress if the model is currently being downloaded.

    Args:
        model_id: Model identifier (e.g., "prithvi-eo-v1-100m")
    """
    try:
        manager = get_model_manager()
        model_info = manager.get_model_info(model_id)

        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        return JSONResponse(model_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model: {e}") from e


@router.post("/{model_id}/download")
async def start_download(model_id: str) -> JSONResponse:
    """Start downloading a model in the background.

    The download runs asynchronously and doesn't block.
    Poll GET /v1/ml-models/{model_id} to track progress.

    Args:
        model_id: Model identifier to download

    Returns:
        Initial download status with model info
    """
    try:
        manager = get_model_manager()

        # Check if model exists
        model_info = manager.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Check if already downloading or available
        current_status = manager.get_model_status(model_id)
        if current_status == ModelStatus.DOWNLOADING:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already downloading")
        if current_status == ModelStatus.AVAILABLE:
            raise HTTPException(status_code=409, detail=f"Model '{model_id}' is already installed")

        # Start background download
        progress = manager.start_download(model_id)

        logger.info(f"Started download for model {model_id}")

        return JSONResponse(
            {
                "model_id": model_id,
                "status": progress.status.value,
                "message": f"Download started for {model_info['name']}",
                "size_mb": model_info["size_mb"],
            },
            status_code=202,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to start download for {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start download: {e}") from e


@router.post("/{model_id}/cancel")
async def cancel_download(model_id: str) -> JSONResponse:
    """Cancel an active download.

    Args:
        model_id: Model identifier

    Returns:
        Cancellation status
    """
    try:
        manager = get_model_manager()

        # Check if model exists
        if manager.get_model_info(model_id) is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Check if actually downloading
        if manager.get_model_status(model_id) != ModelStatus.DOWNLOADING:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not currently downloading")

        cancelled = manager.cancel_download(model_id)

        if cancelled:
            return JSONResponse(
                {
                    "model_id": model_id,
                    "status": "cancelling",
                    "message": "Download cancellation requested",
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel download")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel download for {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel download: {e}") from e


@router.delete("/{model_id}")
async def delete_model(model_id: str) -> JSONResponse:
    """Delete a downloaded model to free up space.

    Args:
        model_id: Model identifier to delete

    Returns:
        Deletion status
    """
    try:
        manager = get_model_manager()

        # Check if model exists
        model_info = manager.get_model_info(model_id)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Check if installed
        if manager.get_model_status(model_id) != ModelStatus.AVAILABLE:
            raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not installed")

        deleted = manager.delete_model(model_id)

        if deleted:
            logger.info(f"Deleted model {model_id}")
            return JSONResponse(
                {
                    "model_id": model_id,
                    "status": "deleted",
                    "message": f"Model '{model_info['name']}' has been deleted",
                    "freed_mb": model_info["size_mb"],
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e}") from e
