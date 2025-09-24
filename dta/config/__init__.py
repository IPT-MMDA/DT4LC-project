__all__ = ["ROOT_DIR", "DTA_PATH", "REGISTRY_PATH"]

from pathlib import Path

__current_file_path = Path(__file__)

ROOT_DIR = __current_file_path.parent.parent.parent

DTA_PATH = ROOT_DIR / "dta"
REGISTRY_PATH = DTA_PATH / "registry.yaml"

CACHE_PATH = ROOT_DIR / ".cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

TEMP_PATH = ROOT_DIR / ".temp"
TEMP_PATH.mkdir(parents=True, exist_ok=True)

THIRD_PARTY_MODELS_PATH = ROOT_DIR / "dta/dti/models/third_party"
THIRD_PARTY_MODELS_PATH.mkdir(parents=True, exist_ok=True)
