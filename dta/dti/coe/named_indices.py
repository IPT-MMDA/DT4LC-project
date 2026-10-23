"""Named spectral indices no built-in analysis computes: NDMI, NBR, SAVI, NDBI.

A prompt that names one ("moisture", "NDMI", "burn severity", ...) is composed
into a band-math plan from this table, without an LLM, and offered to the user
as a proposal. If an attached file lacks a band the index needs, the request is
refused before any planning, with the reason (the committed sample has B2–B6,
so NBR, which needs B7, is refused there).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import re

from dta.config import RESOURCES_PATH
from dta.dti.coe.triggers import phrase_in
from dta.dti.raster import _band_summary, _code_family
from dta.dti.schemas import Attachment, ExecutionPlan, PlanStep, Registry, RegistryItem


@dataclass(frozen=True)
class NamedIndex:
    name: str  # "NDMI"
    title: str  # sentence start of a refusal: "Burn severity uses NBR = ..."
    algorithm: str  # "I don't have a <algorithm> algorithm." (one file)
    change_algorithm: str  # the same for two dates
    formula: str  # over role placeholders, e.g. "({nir} - {swir1}) / ({nir} + {swir1})"
    words: tuple[str, ...]  # whole words or phrases in a prompt that ask for it


NAMED_INDICES: tuple[NamedIndex, ...] = (
    NamedIndex(
        "NDMI",
        "Moisture",
        "moisture",
        "moisture-change",
        "({nir} - {swir1}) / ({nir} + {swir1})",
        ("ndmi", "moisture"),
    ),
    NamedIndex(
        "NBR",
        "Burn severity",
        "burn-severity",
        "burn-severity",
        "({nir} - {swir2}) / ({nir} + {swir2})",
        ("nbr", "burn severity", "burn", "burned", "burnt"),
    ),
    NamedIndex(
        "SAVI",
        "Soil-adjusted vegetation",
        "soil-adjusted vegetation",
        "soil-adjusted vegetation change",
        "1.5 * ({nir} - {red}) / ({nir} + {red} + 0.5)",
        ("savi", "soil-adjusted", "soil adjusted"),
    ),
    NamedIndex(
        "NDBI",
        "Built-up",
        "built-up",
        "built-up change",
        "({swir1} - {nir}) / ({swir1} + {nir})",
        ("ndbi", "built-up"),
    ),
)

ROLE_LABELS = {"red": "Red", "nir": "NIR", "swir1": "SWIR1", "swir2": "SWIR2"}
# Band code of each role per sensor family (dta.dti.raster._code_family); other files read as Landsat.
BAND_CODES = {
    "landsat": {"red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7"},
    "sentinel2": {"red": "B4", "nir": "B8", "swir1": "B11", "swir2": "B12"},
}
# Indices with a built-in analysis: a prompt naming one is not a named-index request ("moisture with NDWI").
_BUILT_IN = ("ndvi", "evi", "ndwi", "ndsi")


# "a 0.1 threshold", "threshold 0.1", "threshold of 0.1", "threshold = .1"
_THRESHOLD = re.compile(r"(?<![\w.])(\d*\.?\d+)\s+threshold\b|\bthreshold\s*(?:of|at|=|:)?\s*(\d*\.?\d+)", re.I)


def prompt_threshold(prompt: str) -> float | None:
    """The change threshold a prompt asks for ("with a 0.1 threshold"), or ``None``; positive only."""
    m = _THRESHOLD.search(prompt)
    value = float(m.group(1) or m.group(2)) if m else 0.0
    return value if value > 0 else None


@dataclass(frozen=True)
class NamedIndexRequest:
    """A prompt that asks for a named index: the composed plan (file inputs bound) and either the
    proposal text or, when a file lacks a band, the refusal (``missing``)."""

    index: NamedIndex
    plan: ExecutionPlan
    text: str
    missing: str | None = None


def match_named_index(prompt: str) -> NamedIndex | None:
    """The index a prompt names: by acronym first, else by its words unless a built-in index is named."""
    for index in NAMED_INDICES:
        if phrase_in(index.name, prompt):
            return index
    if any(phrase_in(b, prompt) for b in _BUILT_IN):
        return None
    return next((i for i in NAMED_INDICES if any(phrase_in(w, prompt) for w in i.words)), None)


def display_formula(index: NamedIndex) -> str:
    """``"(NIR − SWIR1)/(NIR + SWIR1)"``: the formula over role names, as DESIGN writes it."""
    return index.formula.format(**ROLE_LABELS).replace(" / ", "/").replace(" - ", " − ").replace(" * ", " × ")


def _roles(index: NamedIndex) -> list[str]:
    """Roles in order of first use in the formula."""
    order = sorted((index.formula.index(f"{{{r}}}"), r) for r in ROLE_LABELS if f"{{{r}}}" in index.formula)
    return [r for _, r in order]


@functools.cache
def _sample_names() -> frozenset[str]:
    return frozenset(p.name for p in (RESOURCES_PATH / "kahovka_data").glob("*.tif"))


def _bands(path: str) -> dict[str, int] | None:
    """Band names a formula can use on ``path`` (codes and roles), or ``None`` if it cannot be read."""
    import rasterio

    from dta.dti.algorithms.band_math import band_names

    try:
        with rasterio.open(path) as src:
            return band_names(src)
    except rasterio.errors.RasterioIOError:
        return None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def named_index_request(prompt: str, attachments: list[Attachment], reg: Registry) -> NamedIndexRequest | None:
    """The composed plan for a prompt that names an index in :data:`NAMED_INDICES`, or ``None``.

    Two or more files make a change plan (``algorithms/band-math-change``, both dates and their
    difference), one file a single-date plan (``algorithms/band-math``). Without a file whose bands
    can be read nothing is composed: the bands cannot be checked.
    """
    index = match_named_index(prompt)
    files = [a for a in attachments if a.path]
    if index is None or not files:
        return None
    from dta.dti.algorithms.band_math import DEFAULT_THRESHOLD, ROLES
    from dta.dti.coe.orchestrator import _inject_file_bindings

    available = [_bands(str(a.path)) for a in files]
    codes_of = [{n for n in names if n not in ROLES} for names in available if names is not None]
    if not codes_of:
        return None
    family = _code_family({c[1:].lower() for c in codes_of[0]})
    codes = BAND_CODES.get(family or "landsat", BAND_CODES["landsat"])
    roles = _roles(index)
    formula = index.formula.format(**codes)

    change = len(files) >= 2
    if change:
        steps = [
            PlanStep(uses="input/file-before"),
            PlanStep(uses="input/file-after"),
            PlanStep(
                uses="algorithms/band-math-change",
                config={
                    "formula": formula,
                    "name": index.name,
                    "threshold": prompt_threshold(prompt) or DEFAULT_THRESHOLD,
                },
            ),
        ]
    else:
        steps = [
            PlanStep(uses="input/file"),
            PlanStep(uses="algorithms/band-math", config={"formula": formula, "name": index.name}),
        ]
    steps += [PlanStep(uses=it.id) for it in reg.instances if it.kind == "postprocess"][:1]
    plan = ExecutionPlan(flow=prompt, steps=steps, outputs=["publish: chat"], composed=True)
    _inject_file_bindings(plan, files[:2] if change else files[:1], prompt)

    text = proposal_text_for(index, [codes[r] for r in roles], change)
    for att, names in zip(files, available, strict=True):
        if names is None:
            continue  # a missing or unreadable file fails in its own step
        lacking = [r for r in roles if codes[r] not in names]
        if lacking:
            file_codes: list[str | None] = [n for n in names if n not in ROLES]
            if all(a.filename in _sample_names() for a in files):
                subject = "The sample has"
            elif len(files) == 1:
                subject = "This file has"
            else:
                subject = f"{att.filename} has"
            needs = _join([f"{codes[r]} ({ROLE_LABELS[r]})" for r in lacking])
            missing = (
                f"{index.title} uses {index.name} = {display_formula(index)}, which needs "
                f"band{'s' if len(lacking) > 1 else ''} {needs}. {subject} {_band_summary(file_codes)}, "
                "so I can't compute it here."
            )
            return NamedIndexRequest(index, plan, text, missing)
    return NamedIndexRequest(index, plan, text)


def proposal_text_for(index: NamedIndex, band_codes: list[str], change: bool) -> str:
    """DESIGN §4.2: "I don't have a moisture-change algorithm. I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1)
    from B5 and B6 for both dates and difference them — run that?"."""
    algorithm = index.change_algorithm if change else index.algorithm
    dates = " for both dates and difference them" if change else ""
    return (
        f"I don't have a {algorithm} algorithm. I can compute {index.name} = {display_formula(index)} "
        f"from {_join(band_codes)}{dates} — run that?"
    )


def _label(item: RegistryItem) -> str:
    return item.display_name or item.id.rsplit("/", 1)[-1]


def proposal_text(plan: ExecutionPlan, reg: Registry) -> str:
    """The proposal sentence for a composed plan, built from the plan, never from LLM prose.

    Band math with a named index's formula gets the DESIGN wording; other band math names its
    formula and bands; any other combination lists its analysis steps.
    """
    from dta.dti.algorithms.band_math import validate_formula

    items = {it.id: it for it in reg.instances}
    for step in plan.steps:
        if not step.uses.startswith("algorithms/band-math"):
            continue
        formula = str(step.config.get("formula", "")).strip()
        name = str(step.config.get("name") or formula)
        change = step.uses == "algorithms/band-math-change"
        for index in NAMED_INDICES:
            for codes in BAND_CODES.values():
                if index.name == name.upper() and " ".join(formula.split()) == index.formula.format(**codes):
                    return proposal_text_for(index, [codes[r] for r in _roles(index)], change)
        try:
            bands = validate_formula(formula)
        except ValueError:
            bands = []
        what = formula if name == formula else f"{name} = {formula}"
        source = f" from {_join(bands)}" if bands else ""
        dates = " for both dates and difference them" if change else ""
        return f"No built-in analysis matches this. I can compute {what}{source}{dates} — run that?"
    names = [
        _label(items[s.uses]) for s in plan.steps if s.uses in items and items[s.uses].kind in ("algorithm", "model")
    ]
    return (
        f"No single built-in analysis matches this. I can run {_join(names) if names else 'these steps'} — run that?"
    )
