__all__ = [
    "ROOT_DIR",
    "DTA_PATH",
    "REGISTRY_PATH",
    "RESOURCES_PATH",
    "CACHE_PATH",
    "TEMP_PATH",
    "MODELS_PATH",
    "UPLOADS_PATH",
]

import os
from pathlib import Path

__current_file_path = Path(__file__)

ROOT_DIR = __current_file_path.parent.parent.parent

DTA_PATH = ROOT_DIR / "dta"
REGISTRY_PATH = DTA_PATH / "registry.yaml"

# All cache, models, temp files consolidated under resources/
RESOURCES_PATH = ROOT_DIR / "resources"
RESOURCES_PATH.mkdir(parents=True, exist_ok=True)

# Main cache directory - resources/.cache/
CACHE_PATH = RESOURCES_PATH / ".cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Temporary files - resources/.cache/tmp/
TEMP_PATH = CACHE_PATH / "tmp"
TEMP_PATH.mkdir(parents=True, exist_ok=True)

# ML models cache - resources/.cache/models/
# Can be overridden via DT4LC_MODEL_CACHE env var for Docker volumes
MODELS_PATH = Path(os.environ.get("DT4LC_MODEL_CACHE", str(CACHE_PATH / "models")))
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# File uploads - resources/.cache/uploads/
# Can be overridden via DT4LC_UPLOADS_PATH env var for Docker volumes
UPLOADS_PATH = Path(os.environ.get("DT4LC_UPLOADS_PATH", str(CACHE_PATH / "uploads")))
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

# Deprecated: Keep for backwards compatibility but point to new location
THIRD_PARTY_MODELS_PATH = MODELS_PATH
