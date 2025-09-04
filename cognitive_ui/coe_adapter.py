from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from orchestrator import orchestrate_request
from dta import DataAssetManager, ModelRegistry, PipelineExecutor, PostProcessor


def run_orchestrated_flow(
    prompt: str,
    *,
    selected_area: Dict[str, Any] | None = None,
    attached_files: list[str] | None = None,
    options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full orchestrated flow in-process and return plan + result.

    This is used by the Streamlit UI to demonstrate the architecture without
    requiring the external API server.
    """

    project_root = Path(__file__).resolve().parents[1]
    plan = orchestrate_request(
        prompt,
        extras={
            "selected_area": selected_area,
            "attached_files": attached_files or [],
            "options": options or {},
        },
    )

    assets = DataAssetManager(project_root=project_root)
    models = ModelRegistry()
    post = PostProcessor()
    executor = PipelineExecutor(assets=assets, models=models, post=post)
    progress: list[dict[str, Any]] = []

    def on_step(evt: dict[str, Any]) -> None:
        progress.append(evt)

    result = executor.run(plan.flow, plan.steps, on_step=on_step)
    result["progress"] = progress

    return {"plan": {"flow": plan.flow, "steps": [asdict(s) for s in plan.steps], "notes": plan.notes}, "result": result}
