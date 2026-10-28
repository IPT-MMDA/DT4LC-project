"""Allowlist for server files that routes read from a client-supplied path."""

from collections.abc import Iterable
from pathlib import Path
import tempfile

from fastapi import HTTPException

from dta.config import CACHE_PATH, OUTPUTS_PATH, TEMP_PATH

from .jobs import get_job_queue
from .sessions import Session
from .uploads import read_owner
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


async def ensure_served_owner(resolved: Path, session: Session) -> None:
    """404 unless ``session`` may read ``resolved`` (a path from :func:`resolve_served_path`).

    Job artifacts (``OUTPUTS_PATH/<job_id>/...``) belong to the job's owner, uploads and
    their previews to the uploader; operators read everything. Other allowed roots (temp,
    Delineate, GEE exports) and ``OUTPUTS_PATH/run-*`` (pipelines run outside the job
    queue) have no owner and stay readable as before.
    """
    if session.operator:
        return
    outputs = OUTPUTS_PATH.resolve()
    if resolved.is_relative_to(outputs):
        job_id = resolved.relative_to(outputs).parts[0]
        if job_id.startswith("run-"):
            return
        owner = await get_job_queue().get_owner(job_id)
    elif resolved.is_relative_to(UPLOAD_DIR.resolve()):
        owner = read_owner(resolved)
    else:
        return
    if not session.can_see(owner):
        raise HTTPException(status_code=404, detail="File not found")


ATTACHMENT_REFUSED = "This attachment isn't a file you can use here. Attach files with + or the sample."
ATTACHMENT_NOT_FOUND = "Attachment not found."


async def ensure_attachments_allowed(paths: Iterable[str | None], session: Session) -> None:
    """403 or 404 unless ``session`` may use every attachment path in ``paths`` (empty ones are skipped).

    The rule of :func:`resolve_served_path` and :func:`ensure_served_owner`, with messages that never
    echo the path: 403 outside the roots, 404 missing, not a file, or another owner's file.
    """
    for attached in paths:
        if not attached:
            continue
        try:
            await ensure_served_owner(resolve_served_path(attached), session)
        except HTTPException as e:
            detail = ATTACHMENT_REFUSED if e.status_code == 403 else ATTACHMENT_NOT_FOUND
            raise HTTPException(status_code=e.status_code, detail=detail) from None
