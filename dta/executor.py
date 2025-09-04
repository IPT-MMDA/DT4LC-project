from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# First-party imports (package modules)
from .algorithms import Algorithms
from .assets import DataAssetManager
from .models import ModelRegistry
from .post import PostProcessor


@dataclass
class PipelineExecutor:
    """Very small synchronous executor for the planned steps.

    The executor passes named artifacts between steps in a simple dict. This is
    intentionally minimal to keep the first version transparent.
    """

    assets: DataAssetManager
    models: ModelRegistry
    post: PostProcessor
    algorithms: Algorithms | None = None

    def run(
        self,
        flow: str,
        steps: list[Any],
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}

        for idx, step in enumerate(steps, start=1):
            if on_step is not None:
                on_step({"event": "start", "index": idx, "id": step.id, "uses": step.uses})
            if step.uses.startswith("load/"):
                path = self.assets.resolve(step.uses, options=step.with_params)
                for out in step.outputs:
                    artifacts[out] = str(path)
            elif step.uses.startswith("models/"):
                fn = self.models.get(step.uses)
                inputs = [artifacts[x] for x in step.reads] if step.reads else []
                result = fn(*inputs) if inputs else fn()
                for out in step.outputs:
                    artifacts[out] = result
            elif step.uses.startswith("algorithms/"):
                if self.algorithms is None:
                    self.algorithms = Algorithms()
                alg = step.uses.split("/", 1)[1]
                if alg == "ndvi":
                    result = self.algorithms.ndvi(artifacts[step.reads[0]])
                elif alg == "ndvi_change":
                    result = self.algorithms.ndvi_change(artifacts[step.reads[0]], artifacts[step.reads[1]])
                elif alg == "stats_basic":
                    result = self.algorithms.stats_basic(artifacts[step.reads[0]])
                else:
                    raise ValueError(f"Unknown algorithm: {alg}")
                for out in step.outputs:
                    artifacts[out] = result
            elif step.uses.startswith("agent/"):
                # Optional LLM step
                agent_kind = step.uses.split("/", 1)[1]
                if agent_kind == "llm_response":
                    prompt_param = (step.with_params or {}).get("prompt") if hasattr(step, "with_params") else None
                    context_param = (step.with_params or {}).get("context") if hasattr(step, "with_params") else None
                    llm_text: str | None = None
                    try:
                        from cognitive_ui.core.llm import query_gemini

                        base_prompt = (
                            "You are an environmental science assistant. "
                            "Write a concise (<= 220 words), scientifically grounded answer. "
                            "Use numbers from provided context where available. "
                            "Be precise, avoid filler, and state assumptions if uncertain.\n\n"
                        )
                        user_prompt = (
                            "Respond to the USER REQUEST using the latest results first, "
                            "then briefly reference prior context if relevant.\n\n"
                            f"USER REQUEST: {str(prompt_param or '')}\n"
                        )
                        if context_param:
                            try:
                                import json as _json
                                user_prompt += "RECENT CONTEXT: " + _json.dumps(context_param)[-4000:]
                            except Exception:
                                user_prompt += f"RECENT CONTEXT: {context_param}"
                        llm_text = query_gemini(base_prompt + user_prompt)
                    except Exception as e:
                        # Log locally and degrade gracefully
                        print("LLM error:", e)
                        llm_text = None

                    result = {"text": llm_text or "LLM response unavailable. Please configure GEMINI_API_KEY."}
                else:
                    raise ValueError(f"Unknown agent: {agent_kind}")
                for out in step.outputs:
                    artifacts[out] = result
            elif step.uses.startswith("post/"):
                if step.uses == "post/summarize":
                    result = self.post.summarize(data=artifacts.get(step.reads[0]) if step.reads else None)
                    for out in step.outputs:
                        artifacts[out] = result
                elif step.uses == "post/visualize_ndvi":
                    result = self.post.visualize_ndvi(artifacts[step.reads[0]])
                    for out in step.outputs:
                        artifacts[out] = result
                elif step.uses == "post/visualize_raster":
                    result = self.post.visualize_raster(artifacts[step.reads[0]])
                    for out in step.outputs:
                        artifacts[out] = result
                else:
                    raise ValueError(f"Unknown post step: {step.uses}")
            else:
                raise ValueError(f"Unknown step type: {step.uses}")

            if on_step is not None:
                on_step({"event": "end", "index": idx, "id": step.id, "uses": step.uses})

        return {"flow": flow, "artifacts": artifacts}
