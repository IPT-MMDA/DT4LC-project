from dta.dti.registry import get_item
from dta.dti.schemas import ExecutionPlan, Registry


class PlanError(Exception): ...


def validate(plan: ExecutionPlan, reg: Registry) -> ExecutionPlan:
    have_types: set[str] = set()
    for step in plan.steps:
        it = get_item(reg, step.uses)
        # check inputs are satisfied by previous outputs (or empty)
        for inp in it.inputs:
            if inp not in have_types:
                raise PlanError(f"Step {it.id} requires {inp}, not available yet.")
        # accumulate outputs
        for out in it.outputs:
            have_types.add(out)

        # minimal runner sanity
        if it.runner.type == "python" and not it.runner.entrypoint:
            raise PlanError(f"{it.id} missing python entrypoint")

    return plan
