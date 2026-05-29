"""Shared utility functions for the server package."""

import json
from typing import Any

import numpy as np

from dta.config import UPLOADS_PATH

# Upload directory from centralized config (resources/.cache/uploads/)
UPLOAD_DIR = UPLOADS_PATH

# Maximum file upload size (200 MB)
MAX_UPLOAD_SIZE = 200 * 1024 * 1024


def sse_frame(payload: dict[str, Any]) -> bytes:
    """Format payload as Server-Sent Event frame.

    Args:
        payload: Data to encode as JSON

    Returns:
        Bytes with SSE format: "data: <json>\\n\\n"
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def apply_colormap(data: np.ndarray, colormap_name: str) -> np.ndarray:
    """Apply colormap to single-band data, returning RGB."""
    import matplotlib.pyplot as plt

    # Get the colormap
    cmap = plt.get_cmap(colormap_name)

    # Normalize data to 0-1 range
    data_normalized = data / 255.0

    # Apply colormap (returns RGBA)
    rgba = cmap(data_normalized)

    # Convert to RGB uint8 (drop alpha channel)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    return rgb


def detect_data_type(filename: str) -> str:
    """Detect data type from filename."""
    filename_lower = filename.lower()
    if "ndvi" in filename_lower:
        return "ndvi"
    elif "ndwi" in filename_lower:
        return "ndwi"
    elif "ndsi" in filename_lower:
        return "ndsi"
    elif "lulc" in filename_lower or "land" in filename_lower:
        return "lulc"
    elif "change" in filename_lower:
        return "change"
    return "generic"
