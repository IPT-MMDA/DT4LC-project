"""The capabilities document (``GET /v1/capabilities``): what this server can do, for this caller.

Six tiers in display order: ``ask``, ``composed``, ``builtin``, ``models``, ``data``, ``coming``.
Built-in and coming items come from the registry entries that carry a ``ui:`` block; the
Ask and Composed copy lives here because those tiers are not registry entries.

Item shapes per tier:
- ask: ``{id, prompt}``
- composed: ``{id, title, description, recipe, badge, needs_assistant, needs_label, prompt}``
- builtin: ``{id, group, title, formula, description, inputs, outputs, time_estimate,
  runs_without_assistant, prompt, model: {id, title, short}|null, status, status_label}``;
  ``status`` is ``available`` or, for a step that needs a model, the model's status.
- models: ``{id, title, description, size_bytes, size_label, download_time, licence,
  licence_url, used_by, status, status_label}``; ``status`` is one of :data:`MODEL_STATUSES`.
- data: ``{id, title, detail, status, status_label}``; ``status`` is ``always``,
  ``not_bundled`` (sample missing), ``configured`` or ``not_configured``.
- coming: ``{id, title, description}`` (the whole registry description).
"""

from __future__ import annotations

import functools
import os
from typing import Any, Literal

from dta.config import RESOURCES_PATH
from dta.dti.registry import load_registry
from dta.dti.schemas import UI, RegistryItem

from .schemas import AssistantStatus
from .sessions import Role, Session, can_download_models, guest_analyses_per_hour, guest_session_hours, session_info
from .utils import MAX_UPLOAD_SIZE

DEPLOYMENT_LABELS = {"demo": "public demo", "local": "local server"}
ROLE_LABELS: dict[Role, str] = {"guest": "guest access", "member": "member account", "operator": "operator"}

# Provider model ids -> display names; an unknown id is shown as it is.
MODEL_DISPLAY_NAMES = {
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama3.2": "Llama 3.2",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "mistral-medium-latest": "Mistral Medium",
    "mistral-large-latest": "Mistral Large",
    "gpt-4o-mini": "GPT-4o mini",
}

# Registry band roles (config.required_bands) -> the names on the cards.
BAND_ROLE_LABELS = {"blue": "Blue", "green": "Green", "red": "Red", "nir": "NIR", "swir": "SWIR1"}

BUILTIN_GROUPS = (
    {"id": "spectral", "title": "Spectral indices"},
    {"id": "change", "title": "Change, classification, statistics"},
    {"id": "models", "title": "Model-based"},
)

MODEL_STATUSES = ("installed", "downloads_on_first_use", "not_available_for_guests", "not_installed")
RUNS_WITHOUT_ASSISTANT = "Runs without the assistant"

OFFLINE_BANNER = (
    "The assistant is offline on this server, so tiers 1 and 2 are unavailable. "
    "Built-in analyses run as usual; answers are limited to the catalog."
)

ASK_OFFLINE_LABEL = "Assistant offline — limited answers"
ASK_PROMPTS = (
    {"id": "ndwi-vs-ndvi", "prompt": "Why is NDWI better than NDVI for water?"},
    {"id": "explain-last", "prompt": "Explain the last result in two sentences"},
    {"id": "landsat", "prompt": "What would you do differently with Landsat?"},
)

# The sample has no B7, so the band-math example is NDMI (B5, B6), never NBR.
COMPOSED_ITEMS = (
    {
        "id": "any-index",
        "title": "Any spectral index",
        "description": "Band math for indices not in the catalog, from the bands the image has.",
        "recipe": "NDMI = (B5 − B6)/(B5 + B6) · NBR · SAVI · NDBI · custom",
        "prompt": "Map moisture change with NDMI between the two dates",
    },
    {
        "id": "masked-classification",
        "title": "Masked classification",
        "description": "Mask → classify → statistics, so classes are reported only where they matter.",
        "recipe": "ndwi_change → mask → lulc → stats",
        "prompt": "Classify land cover on the flooded fields",
    },
    {
        "id": "multi-date",
        "title": "Multi-date comparison",
        "description": "The same index on three or more dates, as a time series and a difference stack.",
        "recipe": "index(t₁…tₙ) → Δ per step → table",
        "prompt": "Track water weekly through June",
    },
    {
        "id": "parameter-changes",
        "title": "Parameter changes",
        "description": "Re-run a step with a different threshold, band set or cloud limit.",
        "recipe": "threshold 0.2 → 0.1 · cloud ≤10 → ≤25 %",
        "prompt": "Redo the water map with a 0.1 threshold",
    },
)

SAMPLE_DIR = RESOURCES_PATH / "kahovka_data"
SAMPLE_DETAIL = "kahovka_data · HLS L30 · 2023-06-01 → 06-09"
EE_TITLE = "Earth Engine · Sentinel-2 / Landsat / MODIS"
EE_DETAIL = "Fetch by region, dates, bands and cloud filter from the Map"


def deployment() -> Literal["demo", "local"]:
    """``DT4LC_DEPLOYMENT``: ``demo`` (the public demo) or ``local`` (default; any other value)."""
    value = os.environ.get("DT4LC_DEPLOYMENT", "").strip().lower()
    return "demo" if value == "demo" else "local"


def model_display_name(model_id: str | None) -> str | None:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id) if model_id else None


def size_label(size_bytes: int) -> str:
    return f"{round(size_bytes / (1024 * 1024))} MB"


def duration_label(seconds: int) -> str:
    return f"~{seconds} s" if seconds < 90 else f"~{round(seconds / 60)} min"


@functools.cache
def _ui_items() -> tuple[tuple[RegistryItem, ...], tuple[RegistryItem, ...]]:
    """Registry entries with a ``ui:`` block: (built-in by group then order, coming by order)."""
    items = [i for i in load_registry().instances if i.ui]
    group_order = {g["id"]: n for n, g in enumerate(BUILTIN_GROUPS)}
    builtin = sorted(
        (i for i in items if _ui(i).tier == "builtin"),
        key=lambda i: (group_order.get(_ui(i).group or "", len(group_order)), _ui(i).order),
    )
    coming = sorted((i for i in items if _ui(i).tier == "coming"), key=lambda i: _ui(i).order)
    return tuple(builtin), tuple(coming)


def _ui(item: RegistryItem) -> UI:
    assert item.ui is not None
    return item.ui


def _model_status(state: dict[str, Any], role: Role) -> tuple[str, str]:
    """(status, label) of one model for ``role``; see :data:`MODEL_STATUSES`."""
    if state["installed"]:
        return "installed", "Installed on this server"
    if can_download_models(role):
        size = size_label(state["size_bytes"])
        return "downloads_on_first_use", (
            f"Downloads on first use · {size} · {duration_label(state['est_download_s'])} · {state['licence']}"
        )
    if role == "guest" and can_download_models("member"):
        return "not_available_for_guests", "Not available for guests — ask an operator"
    return "not_installed", "Not installed on this server — ask an operator"


def _whole(text: str | None) -> str | None:
    """A registry description as one paragraph (all of it, newlines folded)."""
    return " ".join(text.split()) if text else None


def build_capabilities(
    session: Session, assistant: AssistantStatus, states: list[dict[str, Any]], ee_configured: bool
) -> dict[str, Any]:
    """The capabilities document for ``session``.

    Args:
        session: the caller (its role decides the model statuses and the session block).
        assistant: the cached ``/v1/health`` assistant status.
        states: ``dta.dti.models.model_states()``.
        ee_configured: ``gee_common.is_configured()``.
    """
    role = session.role
    connected = assistant.connected
    display = (model_display_name(assistant.model) or assistant.provider) if connected else None
    builtin_items, coming_items = _ui_items()

    # Models: status per model, ordered like the model-based built-in cards.
    step_ui = {i.model_id: _ui(i) for i in builtin_items if i.model_id}
    model_order = {i.model_id: n for n, i in enumerate(builtin_items) if i.model_id}
    models: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda s: model_order.get(s["id"], len(model_order))):
        status, label = _model_status(state, role)
        ui_model = step_ui[state["id"]].model if state["id"] in step_ui else None
        models.append(
            {
                "id": state["id"],
                "title": ui_model.title if ui_model else state["display_name"],
                "description": ui_model.description if ui_model else None,
                "size_bytes": state["size_bytes"],
                "size_label": size_label(state["size_bytes"]),
                "download_time": duration_label(state["est_download_s"]),
                "licence": state["licence"],
                "licence_url": state["licence_url"],
                "used_by": state["steps"],
                "status": status,
                "status_label": label,
            }
        )
    model_by_id = {m["id"]: m for m in models}

    builtin: list[dict[str, Any]] = []
    for item in builtin_items:
        ui = _ui(item)
        model = model_by_id.get(item.model_id or "")
        status, label = ("available", RUNS_WITHOUT_ASSISTANT)
        if model and model["status"] != "installed":
            status, label = model["status"], model["status_label"]
        elif model:
            status = "installed"
        bands = item.config.get("required_bands", [])
        builtin.append(
            {
                "id": item.id,
                "group": ui.group,
                "title": ui.title,
                "formula": ui.formula,
                "description": ui.description,
                "inputs": ui.inputs or [BAND_ROLE_LABELS.get(b, b) for b in bands],
                "outputs": ui.outputs,
                "time_estimate": ui.time,
                "runs_without_assistant": True,
                "prompt": ui.prompt,
                "model": (
                    {"id": item.model_id, "title": model["title"], "short": ui.model.short if ui.model else None}
                    if model
                    else None
                ),
                "status": status,
                "status_label": label,
            }
        )

    # Summary line.
    installed = [m["title"] for m in models if m["status"] == "installed"]
    if installed:
        models_part = f"{' and '.join(installed)} installed"
    elif any(m["status"] == "downloads_on_first_use" for m in models):
        models_part = "models download on first use"
    else:
        models_part = "no models installed"
    deploy = deployment()
    parts = {
        "deployment": DEPLOYMENT_LABELS[deploy],
        "role": ROLE_LABELS[role],
        "assistant": f"{display} connected" if connected else "assistant offline",
        "models": models_part,
        "earth_engine": "Earth Engine configured" if ee_configured else "no Earth Engine",
    }
    shown = [k for k in parts if k != "deployment" or deploy == "demo"]
    line = " · ".join(parts[k] for k in shown)

    sample_ok = SAMPLE_DIR.is_dir() and any(SAMPLE_DIR.glob("*.tif"))
    per_hour, kept = guest_analyses_per_hour(), guest_session_hours()
    composed = [
        {**c, "badge": "Composed", "needs_assistant": True, "needs_label": "Needs the assistant"}
        for c in COMPOSED_ITEMS
    ]
    tiers: list[dict[str, Any]] = [
        {
            "id": "ask",
            "number": 1,
            "title": "Ask",
            "subtitle": "Questions, clarifications, explanations of methods and results, follow-ups.",
            "status": "connected" if connected else "offline",
            "status_label": f"Assistant · {display} · connected" if connected else ASK_OFFLINE_LABEL,
            "needs_assistant": True,
            "items": [dict(p) for p in ASK_PROMPTS],
        },
        {
            "id": "composed",
            "number": 2,
            "title": "Composed analyses",
            "subtitle": "The assistant combines the built-in steps into new pipelines.",
            "status": "available" if connected else "offline",
            "status_label": "Needs the assistant",
            "needs_assistant": True,
            "badge": "Composed",
            "items": composed,
        },
        {
            "id": "builtin",
            "number": 3,
            "title": "Built-in analyses",
            "subtitle": "Run without the assistant — the catalog every installation has.",
            "status": "available",
            "status_label": RUNS_WITHOUT_ASSISTANT,
            "needs_assistant": False,
            "groups": [dict(g) for g in BUILTIN_GROUPS],
            "items": builtin,
        },
        {
            "id": "models",
            "number": 4,
            "title": "Models",
            "subtitle": "Downloaded weights that some built-in analyses need.",
            "status": "available",
            "needs_assistant": False,
            "items": models,
        },
        {
            "id": "data",
            "number": 5,
            "title": "Data sources",
            "subtitle": "Where imagery can come from on this server.",
            "status": "available",
            "needs_assistant": False,
            "note": f"Guests: up to {per_hour} analyses per hour, results kept {kept} h in this browser, "
            "no account needed.",
            "items": [
                {
                    "id": "uploads",
                    "title": "Uploads",
                    "detail": f"GeoTIFF, up to {MAX_UPLOAD_SIZE // (1024 * 1024)} MB per file · any projection",
                    "status": "always",
                    "status_label": "Always",
                },
                {
                    "id": "sample",
                    "title": "Bundled sample",
                    "detail": SAMPLE_DETAIL,
                    "status": "always" if sample_ok else "not_bundled",
                    "status_label": "Always" if sample_ok else "Not on this server",
                },
                {
                    "id": "earth-engine",
                    "title": EE_TITLE,
                    "detail": EE_DETAIL,
                    "status": "configured" if ee_configured else "not_configured",
                    "status_label": "Configured" if ee_configured else "Not configured",
                },
            ],
        },
        {
            "id": "coming",
            "number": 6,
            "title": "Coming",
            "status": "coming",
            "needs_assistant": False,
            "collapsed": True,
            "items": [{"id": i.id, "title": _ui(i).title, "description": _whole(i.description)} for i in coming_items],
        },
    ]
    example_prompts = [
        {"tier": t["id"], "id": item["id"], "prompt": item["prompt"]}
        for t in tiers
        for item in t["items"]
        if item.get("prompt")
    ]
    return {
        "summary": {"deployment": deploy, "parts": parts, "line": line},
        "assistant": {
            "connected": connected,
            "provider": assistant.provider,
            "model": assistant.model,
            "display": display,
        },
        "offline": not connected,
        "banner": None if connected else OFFLINE_BANNER,
        "tiers": tiers,
        "guest_limits": {"analyses_per_hour": per_hour, "kept_hours": kept},
        "session": session_info(session),
        "example_prompts": example_prompts,
    }
