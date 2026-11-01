"""Operator overrides of environment settings, written by ``PUT /v1/settings``.

The overrides file (``DT4LC_SETTINGS_PATH``, default ``CACHE_PATH / "settings.json"``) is a
flat JSON object from environment variable name to string value, e.g.
``{"LLM_PROVIDER_ORDER": "groq,gemini", "GUEST_ANALYSES_PER_HOUR": "3"}``. Only the names in
:data:`OVERRIDABLE` are read or written; none of them is a secret. Readers call
:func:`getenv` instead of ``os.environ.get``: an override wins over the environment, which
wins over the reader's default. ``.env`` is never written.

The file is re-read when its modification time or size changes, so a restart (or a hand
edit) is picked up; writes are atomic (a temporary file in the same directory, then
``os.replace``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import overload

from . import CACHE_PATH

logger = logging.getLogger(__name__)

LLM_PROVIDERS = ("gemini", "groq", "ollama", "apertus", "anthropic", "mistral", "openai")

OVERRIDABLE = frozenset(
    {
        "LLM_PROVIDER_ORDER",
        *(f"LLM_ENABLE_{name.upper()}" for name in LLM_PROVIDERS),
        "GEMINI_MODELS",
        "GROQ_MODEL",
        "OLLAMA_MODEL",
        "APERTUS_MODEL",
        "ANTHROPIC_MODELS",
        "MISTRAL_MODELS",
        "OPENAI_MODELS",
        "MODEL_DOWNLOAD_ON_DEMAND",
        "ALLOWED_EMAILS",
        "ALLOWED_DOMAINS",
        "OPERATOR_EMAILS",
        "GUEST_ANALYSES_PER_HOUR",
        "GUEST_SESSION_HOURS",
    }
)

_lock = threading.Lock()
# (path, (mtime_ns, size) or None when the file is missing, overrides)
_cache: tuple[Path, tuple[int, int] | None, dict[str, str]] | None = None


def settings_path() -> Path:
    return Path(os.environ.get("DT4LC_SETTINGS_PATH", "").strip() or CACHE_PATH / "settings.json")


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _read(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning(f"Ignoring the settings overrides file {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        logger.warning(f"Ignoring the settings overrides file {path}: not a JSON object")
        return {}
    ignored = sorted(k for k, v in data.items() if k not in OVERRIDABLE or not isinstance(v, str))
    if ignored:
        logger.warning(f"Ignoring unknown or non-string keys in {path}: {', '.join(map(str, ignored))}")
    return {k: v for k, v in data.items() if k in OVERRIDABLE and isinstance(v, str)}


def overrides() -> dict[str, str]:
    """The current overrides (a copy), re-read when the file changed."""
    global _cache
    path = settings_path()
    stamp = _stamp(path)
    with _lock:
        if _cache is None or _cache[0] != path or _cache[1] != stamp:
            _cache = (path, stamp, _read(path) if stamp is not None else {})
        return dict(_cache[2])


@overload
def getenv(name: str, default: str) -> str: ...
@overload
def getenv(name: str, default: None = None) -> str | None: ...
def getenv(name: str, default: str | None = None) -> str | None:
    """The override of ``name`` if there is one, else ``os.environ.get(name, default)``."""
    value = overrides().get(name) if name in OVERRIDABLE else None
    if value is not None:
        return value
    return os.environ.get(name, default) if default is not None else os.environ.get(name)


def source(name: str) -> str:
    """Where :func:`getenv` takes ``name`` from: ``override``, ``env`` or ``default``."""
    if name in overrides():
        return "override"
    return "env" if name in os.environ else "default"


def write_overrides(changes: dict[str, str | None]) -> dict[str, str]:
    """Set (a string) or remove (``None``) overrides and write the file atomically; returns them all.

    Raises:
        KeyError: a name not in :data:`OVERRIDABLE`.
    """
    global _cache
    unknown = sorted(set(changes) - OVERRIDABLE)
    if unknown:
        raise KeyError(f"not overridable: {', '.join(unknown)}")
    path = settings_path()
    with _lock:
        current = _read(path)
        for name, value in changes.items():
            if value is None:
                current.pop(name, None)
            else:
                current[name] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dict(sorted(current.items())), f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _cache = (path, _stamp(path), dict(current))
        return dict(current)
