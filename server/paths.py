"""Allowlist for server files that routes read from a client-supplied path."""

from pathlib import Path
import tempfile

from fastapi import HTTPException

from dta.config import CACHE_PATH, OUTPUTS_PATH, TEMP_PATH

from .utils import UPLOAD_DIR

# Delineate-Anything working directory
DELINEATE_TEMP_DIR = Path(tempfile.gettempdir()) / "dt4lc_delineate"
# Same as dta.dti.data_sources.gee_export.EXPORT_DIR, which imports ``ee`` at module load
GEE_EXPORT_DIR = CACHE_PATH / "gee_exports"


def resolve_served_path(path: str) -> Path:
    """Resolve ``path`` and return it if it is a regular file under an allowed root.

    Raises ``HTTPException`` 403 if the resolved path is outside the roots
    (uploads, job outputs, temp, GEE exports, Delineate temp dir) and 404 if it
    is missing or not a regular file.
    """
    # OUTPUTS_PATH holds job artifacts; TEMP_PATH model outputs written by the
    # executor and Delineate-Anything. Read at call time so tests can monkeypatch them.
    allowed_dirs = [DELINEATE_TEMP_DIR, UPLOAD_DIR, TEMP_PATH, OUTPUTS_PATH, GEE_EXPORT_DIR]

    resolved = Path(path).resolve()
    if not any(resolved.is_relative_to(allowed_dir.resolve()) for allowed_dir in allowed_dirs):
        raise HTTPException(status_code=403, detail="Access denied: path not in allowed directories")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    return resolved
