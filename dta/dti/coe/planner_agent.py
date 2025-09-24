from dta.dti.registry import find_items_by_keywords, find_items_producing
from dta.dti.schemas import ContextUnderstanding, ExecutionPlan, PlanStep, Registry


def plan(ctx: ContextUnderstanding, reg: Registry) -> ExecutionPlan:
    steps: list[PlanStep] = []
    # 1) ensure inputs exist; if user attached raster, use "input/file"
    for need in ctx.required_inputs or []:
        # naive: prefer passthrough inputs that produce need
        for it in reg.instances:
            if it.kind == "input" and need in it.outputs:
                steps.append(PlanStep(uses=it.id))
                break

    # 2) choose a chain to reach desired outputs
    kw_ranked = find_items_by_keywords(reg, ctx.hints.get("keywords", []))
    for want in ctx.desired_outputs or []:
        candidates = [i for i in kw_ranked if want in i.outputs] or find_items_producing(reg, want)
        if not candidates:
            continue
        chosen = candidates[0]
        steps.append(PlanStep(uses=chosen.id))

    # 3) optional postprocess if LLM summary is desired
    for it in kw_ranked:
        if it.kind == "postprocess":
            steps.append(PlanStep(uses=it.id))
            break

    return ExecutionPlan(steps=steps, outputs=["publish: chat"])
