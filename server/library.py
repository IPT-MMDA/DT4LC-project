"""Library rows and result-detail exports of a job: the stored ``card`` and the statistics CSV.

``card`` is computed once, when a job reaches a terminal state, and stored with the job, so
``GET /v1/jobs`` shows a Library row without loading the result::

    {title, analysis: {id, label} | null, kind, thumbnail_url, duration_s, error: {code, message} | null}

- ``title``: deterministic, from the plan's main analysis step (its last algorithm or model step):
  "Water change · NDWI difference", "Moisture change · NDMI difference", "NDVI · 2023-06-01",
  "Raster statistics · 2023-06-01", "Land cover · …", "Field boundaries · …". A job that never got
  a plan (it failed at planning, or was answered as a conversation) is titled with a short form of
  its prompt.
- ``analysis``: the main step's registry id and display label ("Change detection", "Band math",
  "Statistics", "Delineate-Anything", …). A job that failed at planning names the analysis its
  ``intent`` sentence announced (the template plan), if any.
- ``kind``: ``change``, ``index``, ``classification``, ``fields``, ``statistics``, ``image``, or
  ``composed`` for a composed plan; ``null`` without an analysis.
- ``thumbnail_url``: the main step's first map-only PNG (``*_bare``), for completed jobs.
- ``duration_s``: ``completed_at - started_at``; ``null`` if the job never started.
- ``error``: the ``failed`` event's code and message.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from dta.dti.algorithms.change_detection import INDEX_CONFIG
from dta.dti.coe.named_indices import NAMED_INDICES
from dta.dti.data_sources.gee_region import date_range
from dta.dti.raster import _acquisition_date
from dta.dti.schemas import ExecutionPlan, PlanStep, Registry, RegistryItem

if TYPE_CHECKING:
    from .jobs import Job

# Labels where the design's wording differs from the registry's ``ui.title``.
_LABELS = {
    "algorithms/statistics": "Statistics",
    "algorithms/band-math": "Band math",
    "algorithms/band-math-change": "Band math",
}
_KINDS = {
    "algorithms/change-detection": "change",
    "algorithms/band-math-change": "change",
    "algorithms/ndvi": "index",
    "algorithms/evi": "index",
    "algorithms/ndwi": "index",
    "algorithms/ndsi": "index",
    "algorithms/band-math": "index",
    "algorithms/lulc-classifier": "classification",
    "algorithms/snow-classifier": "classification",
    "algorithms/statistics": "statistics",
    "models/delineate-anything": "fields",
    "models/prithvi-reconstruction": "image",
}
# Title of a single-date analysis, before " · <date>" (an index is titled by its name).
_SINGLE_TITLES = {
    "algorithms/statistics": "Raster statistics",
    "algorithms/lulc-classifier": "Land cover",
    "algorithms/snow-classifier": "Snow cover",
    "models/delineate-anything": "Field boundaries",
    "models/prithvi-reconstruction": "Prithvi reconstruction",
}
_SHORT_PROMPT = 60


def analysis_step(plan: ExecutionPlan | None, registry: Registry) -> tuple[PlanStep, RegistryItem] | None:
    """The plan's main analysis step: its last algorithm or model step, with its registry item."""
    items = {i.id: i for i in registry.instances}
    found = None
    for step in plan.steps if plan else []:
        item = items.get(step.uses)
        if item and item.kind in ("algorithm", "model"):
            found = (step, item)
    return found


def analysis_label(item: RegistryItem) -> str:
    """``"Change detection"``, ``"Band math"``, ``"Statistics"``, ``"Delineate-Anything"``, ``"NDVI"``, …"""
    if item.id in _LABELS:
        return _LABELS[item.id]
    if item.ui and item.ui.model:
        return item.ui.model.title
    if item.ui and item.ui.title:
        return item.ui.title
    return item.display_name or item.id.rsplit("/", 1)[-1]


def short_prompt(prompt: str) -> str:
    """The prompt's first line, at most 60 characters (cut at a word, with "…"), first letter upper case."""
    text = " ".join((prompt.strip().splitlines() or [""])[0].split())
    if len(text) > _SHORT_PROMPT:
        text = text[: _SHORT_PROMPT - 1].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"
    return text[:1].upper() + text[1:] if text else "Analysis"


def _input_date(job: Job, plan: ExecutionPlan) -> str | None:
    """The analysed input's date: an Earth Engine fetch's window, else the bound file's upload metadata
    date or the date in its name."""
    attachments = list(job.attachments) + list((job.context or {}).get("previous_attachments", []))
    for step in plan.steps:
        if step.config.get("start_date") and step.config.get("end_date"):
            return date_range(str(step.config["start_date"]), str(step.config["end_date"]))
        path = step.binds.get("RasterPath")
        if path:
            meta = next((a.get("metadata") or {} for a in attachments if a.get("path") == path), {})
            return meta.get("date") or _acquisition_date({}, Path(str(path)).name)
    return None


def _title(job: Job, step: PlanStep, item: RegistryItem) -> str:
    if item.id == "algorithms/change-detection":
        index = str(step.binds.get("IndexType") or "ndvi").lower()
        subject = INDEX_CONFIG.get(index, {}).get("subject")
        return f"{subject} change · {index.upper()} difference" if subject else f"{index.upper()} change · difference"
    name = str(step.config.get("name") or "")
    if item.id == "algorithms/band-math-change":
        named = next((n for n in NAMED_INDICES if n.name == name.upper()), None)
        if named:
            return f"{named.title} change · {named.name} difference"
        return f"{name or 'Band math'} change · difference"
    if item.id in _SINGLE_TITLES:
        head = _SINGLE_TITLES[item.id]
    elif item.id == "algorithms/band-math":
        head = name or "Band math"
    else:
        head = analysis_label(item)
    date = _input_date(job, job.plan) if job.plan else None
    return f"{head} · {date}" if date else head


def _thumbnail_url(job: Job, item: RegistryItem) -> str | None:
    artifacts = ((job.result or {}).get("execution") or {}).get("artifacts") or {}
    artifact = artifacts.get(item.outputs[0]) if item.outputs else None
    visualizations = artifact.get("visualizations") if isinstance(artifact, dict) else None
    for key, ref in (visualizations or {}).items():
        if key.endswith("_bare") and isinstance(ref, dict) and ref.get("url"):
            return str(ref["url"])
    return None


def card(job: Job, registry: Registry, planned: ExecutionPlan | None = None) -> dict[str, Any]:
    """The job's Library card (module docstring). ``planned`` is the plan its ``intent`` sentence
    came from, used for ``analysis`` and ``kind`` when the job has no plan."""
    main = analysis_step(job.plan, registry)
    named = main or analysis_step(planned, registry)
    plan = job.plan or planned
    composed = bool(plan and plan.composed)
    failed = next((e["data"] for e in reversed(job.events) if e["type"] == "failed"), None)
    duration = (job.completed_at - job.started_at).total_seconds() if job.started_at and job.completed_at else None
    return {
        "title": _title(job, *main) if main else short_prompt(job.prompt),
        "analysis": {"id": named[1].id, "label": analysis_label(named[1])} if named else None,
        "kind": ("composed" if composed else _KINDS.get(named[1].id)) if named else None,
        "thumbnail_url": _thumbnail_url(job, main[1]) if main and job.result else None,
        "duration_s": round(duration, 1) if duration is not None else None,
        "error": {"code": failed["code"], "message": failed["message"]} if failed else None,
    }


# --- Statistics CSV --------------------------------------------------------------------------

_BAND_COLUMNS = ("min", "max", "mean", "std", "median")
_PERCENTILES = ("p5", "p25", "p75", "p95")


def slug(text: str) -> str:
    """``"Water change · NDWI difference"`` -> ``"water-change-ndwi-difference"``."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _rows(artifact: dict[str, Any]) -> list[list[Any]] | None:
    """Header and rows of an analysis artifact's statistics, or ``None`` if it has none."""
    classes = artifact.get("class_statistics")
    if isinstance(classes, dict) and classes:
        with_change = any("mean_change" in c for c in classes.values())
        header = ["class", "area_km2", "percent", "pixels"] + (["mean_change"] if with_change else [])
        rows = [
            [c.get("name", name), c.get("area_km2"), c.get("percent"), c.get("pixels")]
            + ([c.get("mean_change")] if with_change else [])
            for name, c in classes.items()
        ]
        return [header, *rows]
    bands = artifact.get("bands")
    if isinstance(bands, dict) and bands:
        header = ["band", *_BAND_COLUMNS, *_PERCENTILES, "valid_pixels", "nodata_pixels"]
        rows = [
            [band]
            + [s.get(k) for k in _BAND_COLUMNS]
            + [(s.get("percentiles") or {}).get(p) for p in _PERCENTILES]
            + [s.get("valid_pixels"), s.get("nodata_pixels")]
            for band, s in bands.items()
        ]
        return [header, *rows]
    stats = artifact.get("statistics")
    if isinstance(stats, dict) and "mean" in stats:
        keys = [k for k, v in stats.items() if isinstance(v, int | float | type(None))]
        name = artifact.get("name") or str(artifact.get("index_type") or "index").upper()
        return [["index", *keys], [name, *(stats[k] for k in keys)]]
    return None


def statistics_csv(job: Job, registry: Registry) -> str | None:
    """The CSV of the job's main analysis statistics, or ``None`` if it has none:

    - the class table (``class,area_km2,percent,pixels[,mean_change]``) for change, LULC and snow;
    - one row per band (``band,min,max,mean,std,median,p5,p25,p75,p95,valid_pixels,nodata_pixels``)
      for raster statistics;
    - one row of the index statistics (``index,min,max,mean,…``) for a single index.
    """
    main = analysis_step(job.plan, registry)
    artifacts = ((job.result or {}).get("execution") or {}).get("artifacts") or {}
    artifact = artifacts.get(main[1].outputs[0]) if main and main[1].outputs else None
    rows = _rows(artifact) if isinstance(artifact, dict) else None
    if rows is None:
        return None
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()
