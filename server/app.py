from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from orchestrator import orchestrate_request
from dta import DataAssetManager, ModelRegistry, PipelineExecutor, PostProcessor


def build_app(project_root: Path) -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from fastapi.responses import JSONResponse
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError("FastAPI is not installed. Install with `[api]` extras.") from exc

    app = FastAPI(title="DT4LC Orchestration API", version="0.1.0")

    assets = DataAssetManager(project_root=project_root)
    models = ModelRegistry()
    post = PostProcessor()
    executor = PipelineExecutor(assets=assets, models=models, post=post)

    def run_flow(req: Dict[str, Any]) -> Any:
        prompt = req.get("prompt", "")
        plan = orchestrate_request(prompt, extras={
            "selected_area": req.get("selected_area"),
            "attached_files": req.get("attached_files", []) or [],
            "options": req.get("options", {}) or {},
        })
        result = executor.run(plan.flow, plan.steps)
        payload = {
            "plan": {"flow": plan.flow, "steps": [asdict(s) for s in plan.steps], "notes": plan.notes},
            "result": {k: ("<ndarray>" if k == "artifacts" and isinstance(v, dict) and "WMS1" in v else v) for k, v in result.items()},
        }
        return JSONResponse(payload)

    # Register route without decorator to keep typing strict
    app.add_api_route("/flow", run_flow, methods=["POST"])  # pragma: no cover

    return app


def app_factory() -> Any:
    return build_app(Path(__file__).resolve().parents[1])
