"""Upload ownership: a ``<stem>.owner.json`` sidecar next to each upload.

The sidecar sits beside the GeoTIFF and its ``<stem>.png`` preview, so it shares their
lifetime and needs neither a schema nor a migration, and works with ``JOB_STORAGE=memory``.
An upload without a sidecar (from before v0.6.0) has no owner: operators only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .sessions import Session

logger = logging.getLogger(__name__)

_SUFFIX = ".owner.json"


def owner_file(path: Path) -> Path:
    """The sidecar of an upload or its preview: ``abc_x.tif`` and ``abc_x.png`` -> ``abc_x.owner.json``."""
    return path.with_name(path.with_suffix("").name + _SUFFIX)


def write_owner(path: Path, session: Session) -> None:
    record = {"owner": session.id, "role": session.role, "created_at": datetime.now().isoformat()}
    owner_file(path).write_text(json.dumps(record), encoding="utf-8")


def _read(path: Path) -> dict[str, str] | None:
    try:
        record = json.loads(owner_file(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def read_owner(path: Path) -> str | None:
    """The session id that uploaded ``path`` (or the upload ``path`` previews), ``None`` if unknown."""
    record = _read(path)
    owner = record.get("owner") if record else None
    return owner if isinstance(owner, str) else None


def cleanup_guest_uploads(upload_dir: Path, guest_hours: float) -> int:
    """Delete guest uploads older than ``guest_hours``: the GeoTIFF, its preview and the sidecar. Blocking."""
    cutoff = datetime.now() - timedelta(hours=guest_hours)
    removed = 0
    sidecars = [f for f in upload_dir.iterdir() if f.name.endswith(_SUFFIX)]
    by_stem: dict[str, list[Path]] = {}
    for f in upload_dir.iterdir():
        if f.is_file() and not f.name.endswith(_SUFFIX):
            by_stem.setdefault(f.with_suffix("").name, []).append(f)
    for sidecar in sidecars:
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            expired = record.get("role") == "guest" and datetime.fromisoformat(record["created_at"]) < cutoff
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if not expired:
            continue
        for f in by_stem.get(sidecar.name[: -len(_SUFFIX)], []):
            f.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.info(f"Cleaned up {removed} expired guest uploads")
    return removed
