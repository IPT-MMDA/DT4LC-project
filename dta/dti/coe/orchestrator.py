from typing import Any

from dta.dti.registry import load_registry
from dta.dti.schemas import ChatRequest

from .context_agent import analyze
from .decision_agent import PlanError, validate
from .planner_agent import plan


def orchestrate(req: ChatRequest) -> dict[str, Any]:
    reg = load_registry()
    ctx = analyze(req, registry_types=reg.types)
    candidate = plan(ctx, reg)
    try:
        final_plan = validate(candidate, reg)
        return {"ok": True, "plan": final_plan.model_dump()}
    except PlanError as e:
        return {"ok": False, "error": str(e), "candidate": candidate.model_dump()}
