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

    # Inject file paths from attachments into input/file step binds
    if req.attachments:
        for step in candidate.steps:
            if step.uses == "input/file":
                # Use the first attachment's path as RasterPath
                if req.attachments[0].path:
                    step.binds["RasterPath"] = req.attachments[0].path
                    import logging

                    logging.info(f"Injected RasterPath: {req.attachments[0].path}")
                else:
                    import logging

                    logging.warning("Attachment has no path!")
                break
    else:
        import logging

        logging.warning("No attachments provided - input/file step will have no RasterPath bind")

    try:
        final_plan = validate(candidate, reg)
        return {"ok": True, "plan": final_plan.model_dump()}
    except PlanError as e:
        return {"ok": False, "error": str(e), "candidate": candidate.model_dump()}
