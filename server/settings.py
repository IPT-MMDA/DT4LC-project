"""Operator settings: the ``GET /v1/settings`` document, applying a ``PUT``, the assistant probe.

Writable values are overrides of environment variables (:mod:`dta.config.overrides`): the
readers (LLM router config, roles, guest quota, model-download policy) call its ``getenv``,
so a write applies at once; assistant changes also rebuild the shared LLM router and drop
the cached assistant status. API keys are reported as present or absent, never their values.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import threading
import time
from typing import Any

from dta import __version__
from dta.config import CACHE_PATH, MODELS_PATH, OUTPUTS_PATH, TEMP_PATH
from dta.config.overrides import getenv, source, write_overrides
from dta.dti.coe.llm import LLMMessage, get_llm_router, reset_llm_router
from dta.dti.coe.llm.base import BaseLLMProvider
from dta.dti.coe.llm.config import _get_models_list, _get_provider_order, _is_provider_enabled
from dta.dti.coe.llm.router import _KEY_PATTERN
from dta.dti.data_sources.gee_common import is_configured as ee_is_configured
from dta.dti.models import model_states

from .capabilities import deployment
from .github_oauth import sign_in_configured
from .jobs import JobStatus, get_job_queue
from .routes.health import _collect_assistant_status, invalidate_assistant_status
from .sample import SAMPLE_DIR, SAMPLE_FILES, SAMPLE_NAME
from .schemas import SettingsUpdate
from .sessions import guest_analyses_per_hour, guest_session_hours
from .utils import UPLOAD_DIR

TEST_PATH = "/v1/settings/assistant/test"
PRE_DOWNLOAD_PATH = "/v1/ml-models/{id}/download"
DOWNLOAD_POLICIES = ["all", "members", "operators", "none"]
BOUNDS = {"guest_analyses_per_hour": {"min": 0, "max": 1000}, "guest_session_hours": {"min": 1, "max": 720}}

# Environment variables whose values never leave the server; probe errors are scrubbed of them.
SECRET_ENV = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "GPT_SECRET",
    "GITHUB_CLIENT_SECRET",
    "SESSION_SECRET",
)


@dataclass(frozen=True)
class ProviderSpec:
    models_env: str
    default_model: tuple[str, str]  # (module under dta.dti.coe.llm, constant) or ("", literal)
    multiple: bool
    key_env: tuple[str, ...]  # any one set counts as a key; empty: no key needed


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec("GEMINI_MODELS", ("gemini", "DEFAULT_GEMINI_MODEL"), True, ("GEMINI_API_KEY",)),
    "groq": ProviderSpec("GROQ_MODEL", ("", "llama-3.3-70b-versatile"), False, ("GROQ_API_KEY",)),
    "ollama": ProviderSpec("OLLAMA_MODEL", ("", "llama3.2"), False, ()),
    "apertus": ProviderSpec("APERTUS_MODEL", ("", "swiss-ai/Apertus-8B"), False, ()),
    "anthropic": ProviderSpec(
        "ANTHROPIC_MODELS", ("anthropic", "DEFAULT_ANTHROPIC_MODEL"), True, ("ANTHROPIC_API_KEY",)
    ),
    "mistral": ProviderSpec("MISTRAL_MODELS", ("mistral", "DEFAULT_MISTRAL_MODEL"), True, ("MISTRAL_API_KEY",)),
    "openai": ProviderSpec(
        "OPENAI_MODELS", ("openai", "DEFAULT_OPENAI_MODEL"), True, ("OPENAI_API_KEY", "GPT_SECRET")
    ),
}

# Field name in PUT /v1/settings -> environment variable it overrides.
ACCESS_ENV = {
    "allowed_emails": "ALLOWED_EMAILS",
    "allowed_domains": "ALLOWED_DOMAINS",
    "operator_emails": "OPERATOR_EMAILS",
    "guest_analyses_per_hour": "GUEST_ANALYSES_PER_HOUR",
    "guest_session_hours": "GUEST_SESSION_HOURS",
}


def _enable_env(name: str) -> str:
    return f"LLM_ENABLE_{name.upper()}"


def _default_model(spec: ProviderSpec) -> str:
    module, value = spec.default_model
    return str(getattr(importlib.import_module(f"dta.dti.coe.llm.{module}"), value)) if module else value


def _list(name: str) -> list[str]:
    """A comma-separated setting as an ordered, de-duplicated, lower-cased list."""
    items = (item.strip().lower() for item in getenv(name, "").split(","))
    return list(dict.fromkeys(item for item in items if item))


# --- Storage ------------------------------------------------------------------

# Entries one walk visits at most; storage numbers are reused for STORAGE_TTL_S.
WALK_CAP = 100_000
STORAGE_TTL_S = 10.0
_storage_lock = threading.Lock()
_storage_cache: dict[Path, tuple[float, dict[str, Any]]] = {}


def dir_usage(root: Path, cap: int = WALK_CAP) -> dict[str, Any]:
    """``{bytes, files, truncated}`` of the regular files under ``root`` (symlinks not followed)."""
    total = files = seen = 0
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    seen += 1
                    if seen > cap:
                        return {"bytes": total, "files": files, "truncated": True}
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue
    return {"bytes": total, "files": files, "truncated": False}


def _cached_usage(root: Path) -> dict[str, Any]:
    now = time.monotonic()
    with _storage_lock:
        hit = _storage_cache.get(root)
        if hit is not None and now - hit[0] < STORAGE_TTL_S:
            return hit[1]
    usage = dir_usage(root)
    with _storage_lock:
        _storage_cache[root] = (now, usage)
    return usage


def server_paths() -> dict[str, Path]:
    return {
        "cache": CACHE_PATH,
        "outputs": OUTPUTS_PATH,
        "temp": TEMP_PATH,
        "uploads": UPLOAD_DIR,
        "models": MODELS_PATH,
    }


def _uploads() -> dict[str, Any]:
    names = [p.name for p in UPLOAD_DIR.iterdir() if p.is_file()] if UPLOAD_DIR.is_dir() else []
    uploaded = [n for n in names if not n.endswith((".owner.json", ".png"))]
    return {"path": str(UPLOAD_DIR), "files": len(uploaded), "bytes": _cached_usage(UPLOAD_DIR)["bytes"]}


# --- The document -------------------------------------------------------------


def _router_names() -> set[str]:
    try:
        return {p.name for p in get_llm_router().providers}
    except Exception:
        return set()


def _assistant() -> dict[str, Any]:
    order = _get_provider_order()
    in_router = _router_names()
    providers = {}
    for name in [*order, *(n for n in PROVIDERS if n not in order)]:
        spec = PROVIDERS.get(name)
        if spec is None:  # an unknown name in LLM_PROVIDER_ORDER (the router skips it too)
            continue
        providers[name] = {
            "enabled": _is_provider_enabled(name),
            "models": _get_models_list(spec.models_env, _default_model(spec)),
            "multiple_models": spec.multiple,
            "key_required": bool(spec.key_env),
            "key_present": any(os.environ.get(env) for env in spec.key_env),
            "available": name in in_router,
            "editable": {
                "enabled": True,
                "models": True,
                "multiple_models": False,
                "key_required": False,
                "key_present": False,
                "available": False,
            },
            "source": {"enabled": source(_enable_env(name)), "models": source(spec.models_env)},
        }
    status = _collect_assistant_status()
    return {
        "provider_order": order,
        "providers": providers,
        "connected": status.connected,
        "provider": status.provider,
        "model": status.model,
        "test_path": TEST_PATH,
        "editable": {
            "provider_order": True,
            "providers": True,
            "connected": False,
            "provider": False,
            "model": False,
            "test_path": False,
        },
        "source": {"provider_order": source("LLM_PROVIDER_ORDER")},
    }


def _models() -> dict[str, Any]:
    policy = getenv("MODEL_DOWNLOAD_ON_DEMAND", "").strip().lower()
    return {
        "items": model_states(),
        "download_policy": policy if policy in DOWNLOAD_POLICIES else "members",
        "download_policy_options": DOWNLOAD_POLICIES,
        "pre_download": {"method": "POST", "path": PRE_DOWNLOAD_PATH},
        "editable": {
            "items": False,
            "download_policy": True,
            "download_policy_options": False,
            "pre_download": False,
        },
        "source": {"download_policy": source("MODEL_DOWNLOAD_ON_DEMAND")},
    }


def _access() -> dict[str, Any]:
    return {
        "allowed_emails": _list("ALLOWED_EMAILS"),
        "allowed_domains": [d.lstrip("@") for d in _list("ALLOWED_DOMAINS")],
        "operator_emails": _list("OPERATOR_EMAILS"),
        "guest_analyses_per_hour": guest_analyses_per_hour(),
        "guest_session_hours": guest_session_hours(),
        "bounds": BOUNDS,
        "sign_in": {"configured": sign_in_configured()},
        "editable": {**dict.fromkeys(ACCESS_ENV, True), "bounds": False, "sign_in": False},
        "source": {field: source(env) for field, env in ACCESS_ENV.items()},
    }


def _data_sources() -> dict[str, Any]:
    key = os.environ.get("GEE_SERVICE_ACCOUNT_KEY", "")
    return {
        "earth_engine": {
            "configured": ee_is_configured(),
            "project_id_present": bool(os.environ.get("GEE_PROJECT_ID")),
            "key_file_present": bool(key) and Path(key).is_file(),
        },
        "uploads": _uploads(),
        "sample": {
            "name": SAMPLE_NAME,
            "bundled": all((SAMPLE_DIR / f).is_file() for f in SAMPLE_FILES),
            "path": str(SAMPLE_DIR),
        },
        "editable": {"earth_engine": False, "uploads": False, "sample": False},
    }


def _server() -> dict[str, Any]:
    paths = server_paths()
    stats = get_job_queue().get_stats()
    by_status = {s.value: 0 for s in JobStatus} | stats["by_status"]
    return {
        "version": __version__,
        "deployment": deployment(),
        "paths": {key: str(p) for key, p in paths.items()},
        "storage": {key: _cached_usage(p) for key, p in paths.items()},
        "jobs": {
            "total": stats["total_jobs"],
            "by_status": by_status,
            "queue_size": stats["queue_size"],
            "workers": stats["workers"],
        },
        "editable": {"version": False, "deployment": False, "paths": False, "storage": False, "jobs": False},
    }


def settings_document() -> dict[str, Any]:
    """``GET /v1/settings``. Blocking (file checks, the cached assistant probe, directory walks)."""
    return {
        "assistant": _assistant(),
        "models": _models(),
        "access": _access(),
        "data_sources": _data_sources(),
        "server": _server(),
    }


# --- PUT ----------------------------------------------------------------------


def _set(changes: dict[str, str | None], fields: set[str], field: str, env: str, value: Any) -> None:
    """Record ``field`` if the request set it: ``None`` drops the override, a list joins with commas."""
    if field not in fields:
        return
    if value is None or isinstance(value, str):
        changes[env] = value
    elif isinstance(value, bool):
        changes[env] = "true" if value else "false"
    elif isinstance(value, list):
        changes[env] = ",".join(value)
    else:
        changes[env] = str(value)


def override_changes(update: SettingsUpdate) -> dict[str, str | None]:
    """The environment overrides a validated ``PUT`` sets (a string) or drops (``None``)."""
    changes: dict[str, str | None] = {}
    if update.assistant is not None:
        a = update.assistant
        _set(changes, a.model_fields_set, "provider_order", "LLM_PROVIDER_ORDER", a.provider_order)
        for name, p in a.providers.items():
            _set(changes, p.model_fields_set, "enabled", _enable_env(name), p.enabled)
            _set(changes, p.model_fields_set, "models", PROVIDERS[name].models_env, p.models)
    if update.models is not None:
        m = update.models
        _set(changes, m.model_fields_set, "download_policy", "MODEL_DOWNLOAD_ON_DEMAND", m.download_policy)
    if update.access is not None:
        for field, env in ACCESS_ENV.items():
            _set(changes, update.access.model_fields_set, field, env, getattr(update.access, field))
    return changes


def apply_update(update: SettingsUpdate) -> None:
    """Write the overrides of ``update``; rebuild the LLM router if the assistant changed. Blocking."""
    changes = override_changes(update)
    if not changes:
        return
    write_overrides(changes)
    if update.assistant is not None:
        reset_llm_router()
        invalidate_assistant_status()


# --- Assistant probe ----------------------------------------------------------

PROBE_TIMEOUT_S = 20.0
_PROBE = [LLMMessage(role="user", content="Reply with the single word OK.")]


def redact(text: str) -> str:
    """``text`` without the values of :data:`SECRET_ENV` or anything shaped like an API key."""
    for name in SECRET_ENV:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "***")
    return _KEY_PATTERN.sub("***", text)[:300]


def _probe(provider: BaseLLMProvider) -> dict[str, Any]:
    start = time.perf_counter()
    error: str | None = None
    try:
        if not provider.is_available():
            error = "not available"
        else:
            provider.generate(_PROBE, temperature=0.0)
    except Exception as exc:
        error = redact(f"{type(exc).__name__}: {exc}")
    ms = round((time.perf_counter() - start) * 1000)
    return {"name": provider.name, "model": provider.model, "ok": error is None, "latency_ms": ms, "error": error}


async def probe_assistant() -> dict[str, Any]:
    """Probe every provider of the shared router concurrently, each bounded by :data:`PROBE_TIMEOUT_S`."""
    try:
        providers = list(get_llm_router().providers)
    except Exception as exc:
        return {"connected": False, "providers": [], "error": redact(str(exc))}

    async def one(provider: BaseLLMProvider) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(asyncio.to_thread(_probe, provider), PROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            ms = round(PROBE_TIMEOUT_S * 1000)
            error = f"no answer in {PROBE_TIMEOUT_S:.0f} s"
            return {"name": provider.name, "model": provider.model, "ok": False, "latency_ms": ms, "error": error}

    results = list(await asyncio.gather(*(one(p) for p in providers)))
    return {"connected": any(r["ok"] for r in results), "providers": results}
