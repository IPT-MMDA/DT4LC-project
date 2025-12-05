"""Execution plan validation.

Validates that execution plans have satisfied dependencies and valid runners.
"""

from dta.dti.registry import get_item
from dta.dti.schemas import ExecutionPlan, Registry


class PlanError(Exception):
    """Raised when plan validation fails."""


def validate(plan: ExecutionPlan, reg: Registry) -> ExecutionPlan:
    """Validate execution plan dependencies and runner configuration.

    Raises:
        PlanError: If step dependencies are unmet or runner is misconfigured.
    """
    have_types: set[str] = set()
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

        # minimal runner sanity
        if it.runner and it.runner.type == "python" and not it.runner.entrypoint:
            raise PlanError(f"{it.id} missing python entrypoint")

    return plan
