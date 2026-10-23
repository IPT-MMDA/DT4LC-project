"""Job event stream: event names, payloads, SSE frames and the failure table.

``GET /v1/jobs/{id}/events`` streams one job's event log as Server-Sent Events.
Every event is stored on the job as ``{"seq": int, "type": str, "data": dict}``
(``seq`` starts at 1 and increases by one per event) and sent as::

    id: <seq>
    event: <type>
    data: <data as JSON>

Every ``data`` carries ``t`` (Unix time in seconds, 3 decimals) besides the
fields below; optional fields are left out when unknown.

- ``intent`` ``{text}``: one sentence of what will happen, written from the
  registry triggers and the template planner at submission (no LLM call).
- ``proposal`` ``{text, plan}``: planning composed a plan that is not a single
  built-in analysis (``plan.composed``); the job waits at status ``proposal``
  for ``POST /v1/jobs/{id}/confirm``, which continues the same log with
  ``planning`` ready. ``proposal`` is not terminal: the stream stays open.
- ``planning`` ``{phase: "started"}`` when planning starts, then
  ``{phase: "ready", steps: [{index, id, name, kind}], composed}`` when the
  plan is ready (``index`` is 1-based, ``kind`` is the registry kind). When a
  model's weights must be downloaded first, a ``{index, id: "prepare/<model id>",
  name: "Preparing model · <model>", kind: "prepare", model_id}`` row precedes
  the step that needs it, and the later rows' indices shift by one.
- ``step_started`` ``{index, id, name}``.
- ``step_progress`` ``{index, fraction?, message?, bytes?, total_bytes?, eta_s?}``.
- ``step_finished`` ``{index, id, elapsed_s}``.
- ``done`` ``{elapsed_s, summary, followups}``: fetch ``GET /v1/jobs/{id}`` for the result
  (which carries the same ``followups``, full prompts for the chips under it, :func:`followups`).
- ``failed`` ``{code, message, fixes, step_index?, raw?}``, see :func:`classify_failure`.
- ``cancelled`` ``{at_step?, reason?}``; ``reason: "expired"`` for a proposal never confirmed.

``done``, ``failed`` and ``cancelled`` are terminal: nothing follows them.
"""

from __future__ import annotations

from datetime import date
import functools
import json
import re
from typing import Any

from dta.config import RESOURCES_PATH
from dta.dti.coe.llm.router import _KEY_PATTERN
from dta.dti.data_sources.gee_region import EarthEngineNotConfiguredError
from dta.dti.executor import InputBandsError, MissingInputError, ModelGatedError, ModelNotInstalledError
from dta.dti.models import AVAILABLE_MODELS
from dta.dti.schemas import ExecutionPlan, PlanStep, Registry, RegistryItem

EVENT_TYPES = (
    "intent",
    "proposal",
    "planning",
    "step_started",
    "step_progress",
    "step_finished",
    "done",
    "failed",
    "cancelled",
)
TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})

KEEPALIVE_FRAME = b": keep-alive\n\n"

_MAX_MESSAGE = 500
_MAX_RAW = 2000


def sse_event_frame(event: dict[str, Any]) -> bytes:
    """One stored event as an SSE frame: ``id``, ``event`` and one ``data`` line."""
    data = json.dumps(event["data"], ensure_ascii=False, separators=(",", ":"))
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n".encode()


# --- Step names ----------------------------------------------------------------


def _base_label(item: RegistryItem) -> str:
    if item.kind == "postprocess":
        return "Interpretation"
    if item.display_name:
        return re.sub(r"\s+(Calculation|\(multi-index\))$", "", item.display_name)
    return item.id.rsplit("/", 1)[-1].replace("-", " ").capitalize()


def step_name(step: PlanStep, item: RegistryItem, filenames: dict[str, str] | None = None) -> str:
    """Display name of a plan step, e.g. ``"NDVI"``, ``"Read hlsl_20230601.tif"``, ``"Change Detection · NDWI"``,
    ``"Fetch HLS L30 · 2023-05-29 → 06-04"`` (an Earth Engine fetch names itself in its config).

    ``filenames`` maps a bound path to the name the user uploaded it as.
    """
    if item.kind == "input" and step.config.get("name"):
        return str(step.config["name"])
    if item.kind == "input":
        path = next(iter(step.binds.values()), "")
        if not path:
            return "Read input file"
        return f"Read {(filenames or {}).get(path) or path.replace(chr(92), '/').rsplit('/', 1)[-1]}"
    if step.config.get("name"):
        return str(step.config["name"])
    label = _base_label(item)
    if step.binds.get("IndexType"):
        label = f"{label} · {step.binds['IndexType'].upper()}"
    return label


def _item(registry: Registry, step_id: str) -> RegistryItem | None:
    return next((i for i in registry.instances if i.id == step_id), None)


def step_skeleton(
    plan: ExecutionPlan,
    registry: Registry,
    filenames: dict[str, str] | None = None,
    prepare: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """``[{index, id, name, kind}]`` for the ``planning`` ready event; index is 1-based.

    ``prepare`` maps a plan step's 1-based index to the model whose weights are downloaded
    before it (:meth:`PipelineExecutor.models_to_prepare`); each adds a ``kind: "prepare"``
    row, so a row's index equals the plan index only without preparation rows.
    """
    rows: list[dict[str, Any]] = []
    for plan_index, step in enumerate(plan.steps, start=1):
        model_id = (prepare or {}).get(plan_index)
        if model_id:
            rows.append(
                {
                    "index": len(rows) + 1,
                    "id": f"prepare/{model_id}",
                    "name": f"Preparing model · {AVAILABLE_MODELS[model_id].name}",
                    "kind": "prepare",
                    "model_id": model_id,
                }
            )
        item = _item(registry, step.uses)
        rows.append(
            {
                "index": len(rows) + 1,
                "id": step.uses,
                "name": step_name(step, item, filenames) if item else step.uses,
                "kind": item.kind if item else "algorithm",
            }
        )
    return rows


def format_elapsed(seconds: float) -> str:
    """``"<1 s"``, ``"48 s"``, ``"1.2 min"``."""
    if seconds < 1:
        return "<1 s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"


def done_summary(skeleton: list[dict[str, Any]], elapsed_s: float) -> str:
    """DESIGN §4.5 one-liner, e.g. ``"3 steps · 48 s · NDWI (×2) → Change Detection"``.

    Counts every plan step; the chain lists the algorithm and model steps,
    with consecutive repeats collapsed to ``(×n)``.
    """
    names = [row["name"] for row in skeleton if row["kind"] in ("algorithm", "model")]
    chain: list[list[Any]] = []
    for name in names:
        if chain and chain[-1][0] == name:
            chain[-1][1] += 1
        else:
            chain.append([name, 1])
    parts = [f"{len(skeleton)} step{'s' if len(skeleton) != 1 else ''}", format_elapsed(elapsed_s)]
    if chain:
        parts.append(" → ".join(name if n == 1 else f"{name} (×{n})" for name, n in chain))
    return " · ".join(parts)


# Follow-ups after an analysis whose registry item lists none.
DEFAULT_FOLLOWUPS = ["What else can you do with this data?"]


def followups(plan: ExecutionPlan | None, registry: Registry) -> list[str]:
    """Follow-up prompts for the chips under a result: ``ui.followups`` of the plan's last analysis step.

    Per-variant lists are chosen by the step's ``IndexType`` bind (change detection) or its
    config ``name`` (band math), lower-cased, else ``default``. :data:`DEFAULT_FOLLOWUPS`
    when the item has none; no plan (a conversation reply) has none.
    """
    if plan is None:
        return []
    analysis = [(s, i) for s in plan.steps if (i := _item(registry, s.uses)) and i.kind in ("algorithm", "model")]
    if not analysis:
        return list(DEFAULT_FOLLOWUPS)
    step, item = analysis[-1]
    configured = item.ui.followups if item.ui else []
    if isinstance(configured, dict):
        variant = str(step.binds.get("IndexType") or step.config.get("name") or "").lower()
        configured = configured.get(variant) or configured.get("default", [])
    return list(configured or DEFAULT_FOLLOWUPS)


# --- Intent sentence -----------------------------------------------------------

# What a change in each index maps ("...to map where water changed").
_CHANGE_SUBJECT = {"ndvi": "vegetation", "ndwi": "water", "ndsi": "snow and ice"}


@functools.cache
def _sample_names() -> frozenset[str]:
    return frozenset(p.name for p in (RESOURCES_PATH / "kahovka_data").glob("*.tif"))


def _day_month(dates: list[str]) -> list[str]:
    parsed = []
    for d in dates:
        try:
            parsed.append(date.fromisoformat(d))
        except ValueError:
            return []
    with_year = len({d.year for d in parsed}) > 1
    return [f"{d.day} {d:%B}" + (f" {d.year}" if with_year else "") for d in parsed]


def intent_text(
    plan: ExecutionPlan, registry: Registry, attachments: list[dict[str, Any]], region: dict[str, Any] | None = None
) -> str:
    """Plain-language sentence for the ``intent`` event, from a deterministic (template) plan and the attachments.

    Dates come from the upload metadata (``attachment["metadata"]["date"]``) when present. Without
    attachments, a ``region`` (the job's region echo) is the source, with its start and end dates.
    """
    names = [str(a.get("filename") or "") for a in attachments]
    if names and all(n in _sample_names() for n in names):
        source = "the sample"
    elif len(names) == 1:
        source = names[0]
    elif names:
        source = f"{', '.join(names[:-1])} and {names[-1]}"
    else:
        source = ""
    dates = [d for a in attachments if (d := (a.get("metadata") or {}).get("date"))]
    days = _day_month(dates) if len(dates) == len(attachments) else []
    if region and not attachments:
        source = region.get("name") or "the region"
        # A change analysis fetches a composite per date; a single-date one a composite over the range.
        change = any(s.uses == "input/file-before" for s in plan.steps)
        days = _day_month([region["start_date"], region["end_date"]]) if change else []
    for_days = f" for {' and '.join(days)}" if days else ""
    from_source = f" from {source}" if source else ""

    main = None
    for step in plan.steps:
        item = _item(registry, step.uses)
        if item and item.kind in ("algorithm", "model"):
            main = (step, item)
            break
    if main is None:
        return "I'll work out which analysis fits this request" + (f" on {source}." if source else ".")
    step, item = main
    index_type = step.binds.get("IndexType")
    if index_type:
        subject = _CHANGE_SUBJECT.get(index_type.lower(), "what")
        where = "what changed" if subject == "what" else f"where {subject} changed"
        return f"I'll compute {index_type.upper()}{for_days}{from_source}, then difference them to map {where}."
    name = step_name(step, item)
    if item.kind == "model":
        return f"I'll run {name}" + (f" on {source}." if source else ".")
    return f"I'll compute {name}{for_days}{from_source}."


# --- Failures ------------------------------------------------------------------
#
# A fix is a button the UI renders under a failed analysis:
#   {"label": str, "kind": "prompt" | "retry" | "upload" | "open_map" | "sample", "prompt"?: str, "params"?: dict}
#   prompt:   put ``prompt`` in the composer (or send it) as a new request.
#   retry:    resubmit the same prompt and attachments, with ``params`` merged into the request.
#   upload:   open the file picker, then retry with the new file.
#   open_map: open the map to pick a region.
#   sample:   attach the bundled Kakhovka sample (both dates) instead of the region and resend the prompt.

_ASK_CAPABILITIES = {"label": "See what I can do", "kind": "prompt", "prompt": "What can you do?"}
_BUILT_IN = {
    "label": "Use a built-in analysis instead",
    "kind": "prompt",
    "prompt": "Which built-in analyses can I run on this file?",
}
_UPLOAD_OWN = {"label": "Use my own image", "kind": "upload"}
_RETRY = {"label": "Retry", "kind": "retry"}

# (code, exception types in the cause chain, message pattern, stage, fixes): the first matching row wins.
# ``stage`` limits a row to "planning" or "running" failures. Later stories add their codes here
# (GUEST_QUOTA, ...).
FAILURE_TABLE: list[tuple[str, tuple[type[BaseException], ...], str | None, str | None, list[dict[str, Any]]]] = [
    (
        "INPUT_MISSING",
        (MissingInputError,),
        r"Missing required input|I need (a|an) .*(file|image)",
        None,
        [
            {"label": "Upload a GeoTIFF", "kind": "upload"},
            {"label": "Pick a region on the map", "kind": "open_map"},
        ],
    ),
    (
        "EE_NOT_CONFIGURED",
        (EarthEngineNotConfiguredError,),
        r"Earth Engine (isn't configured on|refused) this server",
        None,
        [_UPLOAD_OWN, {"label": "Try the Kakhovka sample", "kind": "sample"}],
    ),
    (
        "BAND_MISSING",
        (InputBandsError,),
        r"needs \w+ HLS bands",
        None,
        [
            _UPLOAD_OWN,
            {"label": "Run NDWI change instead", "kind": "prompt", "prompt": "Map NDWI change between the two dates"},
        ],
    ),
    (
        "BAND_MISSING",
        (),
        r"uses NBR = .*, which needs bands? ",
        None,
        [
            {
                "label": "Map moisture change with NDMI instead",
                "kind": "prompt",
                "prompt": "Map moisture change with NDMI between the two dates",
            },
            _UPLOAD_OWN,
        ],
    ),
    (
        "BAND_MISSING",
        (),
        r"Band \w+ is not in this file|requires band|, which needs bands? ",
        None,
        [
            _UPLOAD_OWN,
            {
                "label": "What can this file do?",
                "kind": "prompt",
                "prompt": "Which analyses can I run with the bands in this file?",
            },
        ],
    ),
    (
        "MODEL_GATED",
        (ModelGatedError,),
        r"isn't available for \w+s — ask an operator",
        None,
        [_ASK_CAPABILITIES, _BUILT_IN],
    ),
    (
        "MODEL_NOT_INSTALLED",
        (ModelNotInstalledError,),
        r"isn't installed on this server",
        None,
        [_ASK_CAPABILITIES, _BUILT_IN],
    ),
    ("INTERRUPTED", (), r"Server restarted while job was in progress", None, [_RETRY]),
    ("PLAN_FAILED", (), r"^I can't map this request", "planning", [_ASK_CAPABILITIES, _BUILT_IN]),
    ("PLAN_FAILED", (), None, "planning", [_ASK_CAPABILITIES, _RETRY]),
    ("INTERNAL", (), None, None, [_RETRY]),
]


def redact(text: str) -> str:
    """Strip API-key-like tokens, as the LLM router does before logging or raising."""
    return _KEY_PATTERN.sub("***", text)


def _plain_message(text: str) -> str:
    """The error without the executor's ``Step <id> failed:`` / validator's ``Step <id>:`` prefixes."""
    return re.sub(r"^(?:Step \S+(?: failed)?: )+", "", text).strip()


def classify_failure(
    error: BaseException | str, stage: str = "running", step_index: int | None = None
) -> dict[str, Any]:
    """``failed`` event data for an error: ``{code, message, fixes, step_index?, raw}``.

    Args:
        error: The exception (its ``__cause__`` chain is searched too) or the stored error text.
        stage: ``"planning"`` before the plan is ready, else ``"running"``.
        step_index: 1-based index of the step that failed, if any.
    """
    chain: list[BaseException] = []
    exc = error if isinstance(error, BaseException) else None
    while exc is not None and exc not in chain:
        chain.append(exc)
        exc = exc.__cause__
    raw = redact(str(error))

    def matches(row: tuple[str, tuple[type[BaseException], ...], str | None, str | None, Any]) -> bool:
        _, types, pattern, row_stage, _ = row
        if row_stage is not None and row_stage != stage:
            return False
        by_type = bool(types) and any(isinstance(e, types) for e in chain)
        by_text = pattern is not None and re.search(pattern, raw) is not None
        return by_type or by_text or (not types and pattern is None)

    code, _, _, _, fixes = next(row for row in FAILURE_TABLE if matches(row))
    data: dict[str, Any] = {
        "code": code,
        "message": _plain_message(raw)[:_MAX_MESSAGE] or "The analysis failed.",
        "fixes": [dict(f) for f in fixes],
    }
    if step_index is not None:
        data["step_index"] = step_index
    data["raw"] = raw[:_MAX_RAW]
    return data
