"""Execution plan validation.

Validates that execution plans have satisfied dependencies and valid runners.
"""

from typing import Any

from dta.dti.exceptions import ResourceError
from dta.dti.registry import get_item
from dta.dti.schemas import ExecutionPlan, Registry
from dta.dti.validation import PlanValidator

BAND_MATH_ENTRYPOINT = "dta.dti.algorithms.band_math"
EE_FETCH_ENTRYPOINT = "dta.dti.data_sources.gee_region"


class PlanError(Exception):
    """Raised when plan validation fails."""


def validate(plan: ExecutionPlan, reg: Registry) -> ExecutionPlan:
    """Validate execution plan dependencies and runner configuration.

    Raises:
        PlanError: If step dependencies are unmet, runner is misconfigured,
            or the plan exceeds resource limits.
    """
    try:
        PlanValidator(reg).check_resources(plan)
    except ResourceError as e:
        raise PlanError(str(e)) from e

    have_types: set[str] = set()
    bound: dict[str, str] = {}  # output type -> value bound on an input step (a raster path)
    for step in plan.steps:
        it = get_item(reg, step.uses)
        # check inputs are satisfied by previous outputs, binds, or empty
        for inp in it.inputs:
            # Input is satisfied if:
            # 1. A previous step produces this type (in have_types)
            # 2. A literal value is provided in step.binds (e.g., IndexType="ndvi")
            if inp not in have_types and inp not in step.binds:
                raise PlanError(f"Step {it.id} requires {inp}, not available yet.")
        # accumulate outputs
        for out in it.outputs:
            have_types.add(out)
            if out in step.binds:
                bound[out] = step.binds[out]

        # minimal runner sanity
        if it.runner and it.runner.type == "python" and not it.runner.entrypoint:
            raise PlanError(f"{it.id} missing python entrypoint")

        # Step config: only band math and the Earth Engine fetch take one; a formula must be safe before anything runs.
        if it.runner and it.runner.entrypoint == BAND_MATH_ENTRYPOINT:
            _check_band_math(step.config.get("formula", ""), [bound[i] for i in it.inputs if i in bound], it.id)
        elif it.runner and it.runner.entrypoint == EE_FETCH_ENTRYPOINT:
            _check_ee_fetch(step.config, it.id)
        elif step.config:
            raise PlanError(f"Step {it.id} takes no config, got {sorted(step.config)}.")

    return plan


def _check_band_math(formula: str, raster_paths: list[str], step_id: str) -> None:
    """PlanError unless ``formula`` is safe band math whose bands are in every existing raster given."""
    import rasterio

    from dta.dti.algorithms.band_math import band_names, validate_formula

    try:
        validate_formula(formula)
        for raster_path in raster_paths:
            try:
                with rasterio.open(raster_path) as src:
                    available = band_names(src)
            except rasterio.errors.RasterioIOError:
                continue  # a missing or unreadable file fails in its own step
            validate_formula(formula, available)
    except ValueError as e:
        raise PlanError(f"Step {step_id}: {e}") from e


def _check_ee_fetch(config: dict[str, Any], step_id: str) -> None:
    """PlanError unless ``config`` is an Earth Engine fetch config (region bounds, dataset, dates, bands)."""
    from dta.dti.data_sources.gee_region import check_config

    try:
        check_config(config)
    except ValueError as e:
        raise PlanError(f"Step {step_id}: {e}") from e
