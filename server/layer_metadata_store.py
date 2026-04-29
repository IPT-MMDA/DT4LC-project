"""Layer metadata storage for GEE layers.

Stores metadata about GEE layers loaded on the map so they can be
exported for AI analysis later.
"""

import json
import logging
from typing import Any

from dta.config import CACHE_PATH

logger = logging.getLogger(__name__)

# Storage path for layer metadata
METADATA_DIR = CACHE_PATH / "layer_metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def save_layer_metadata(layer_id: str, metadata: dict[str, Any]) -> None:
    """Save layer metadata to disk.

    Args:
        layer_id: Unique layer identifier
        metadata: Layer metadata dictionary
    """
    try:
        metadata_file = METADATA_DIR / f"{layer_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for layer {layer_id}")
    except Exception as e:
        logger.error(f"Failed to save metadata for layer {layer_id}: {e}")
        raise


def get_layer_metadata(layer_id: str) -> dict[str, Any] | None:
    """Retrieve layer metadata from disk.

    Args:
        layer_id: Unique layer identifier

    Returns:
        Layer metadata dictionary or None if not found
    """
    try:
        metadata_file = METADATA_DIR / f"{layer_id}.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata not found for layer {layer_id}")
            return None

        with open(metadata_file) as f:
            metadata = json.load(f)
        logger.info(f"Retrieved metadata for layer {layer_id}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to retrieve metadata for layer {layer_id}: {e}")
        return None


def delete_layer_metadata(layer_id: str) -> bool:
    """Delete layer metadata from disk.

    Args:
        layer_id: Unique layer identifier

    Returns:
        True if deleted, False if not found
    """
    try:
        metadata_file = METADATA_DIR / f"{layer_id}.json"
        if not metadata_file.exists():
            return False

        metadata_file.unlink()
        logger.info(f"Deleted metadata for layer {layer_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete metadata for layer {layer_id}: {e}")
        return False


def list_all_layers() -> list[dict[str, Any]]:
    """List all stored layer metadata.

    Returns:
        List of layer metadata dictionaries
    """
    try:
        layers = []
        for metadata_file in METADATA_DIR.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    layers.append(metadata)
            except Exception as e:
                logger.error(f"Failed to read {metadata_file}: {e}")
                continue

        logger.info(f"Found {len(layers)} stored layers")
        return layers
    except Exception as e:
        logger.error(f"Failed to list layers: {e}")
        return []
