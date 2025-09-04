from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import CapabilitiesRegistry
from .types import OrchestrationContext, PipelinePlan, PipelineStep


@dataclass
class ContextUnderstandingAgent:
    """Extracts structure from a free-form prompt.

    In production, this can be backed by an LLM. For now, we implement a
    deterministic baseline to keep the stack runnable.
    """

    def analyze(self, prompt: str, extras: dict[str, Any] | None = None) -> OrchestrationContext:
        selected_area = (extras or {}).get("selected_area")
        attached_files = list((extras or {}).get("attached_files", []))
        options = dict((extras or {}).get("options", {}))
        return OrchestrationContext(
            prompt=prompt,
            selected_area=selected_area,
            attached_files=attached_files,
            options=options,
        )


@dataclass
class DecisionMakingAgent:
    """Decides which capability family to route to (e.g., precipitation forecast)."""

    def decide_flow(self, ctx: OrchestrationContext) -> str:
        text = ctx.prompt.lower()
        explain = any(k in text for k in ["explain", "explanation", "interpret", "scientific"])
        if "ndvi" in text and "change" in text:
            return "ndvi_change"
        if "ndvi" in text and explain:
            return "ndvi_explain"
        if "ndvi" in text:
            return "ndvi_single"
        if any(k in text for k in ["distribution", "histogram", "stats", "land cover"]) and explain:
            return "stats_explain"
        if any(k in text for k in ["distribution", "histogram", "stats", "land cover"]):
            return "stats_single"
        return "generic_environmental_query_v1"


@dataclass
class PlannerAgent:
    """Produces a pipeline plan from capabilities registry."""

    registry_path: Path

    def plan(self, flow: str, ctx: OrchestrationContext) -> PipelinePlan:
        # Registry is available for future use/validation
        _ = CapabilitiesRegistry(self.registry_path)

        steps: list[PipelineStep] = []

        # Data source selection heuristics: allow specifying via options or default to Kahovka
        loader = "load/kahovka_raster"
        ds_option = (ctx.options or {}).get("dataset") if hasattr(ctx, "options") else None
        if isinstance(ds_option, str) and ds_option.lower().startswith("prithvi"):
            loader = "load/prithvi_example_raster"
        steps.append(PipelineStep(id="data_1", uses=loader, outputs=["R1"]))

        # If user asked for ndvi change and provided two rasters, construct change pipeline
        if flow == "ndvi_change":
            # Inputs: either two uploaded files or default kahovka twice
            if ctx.attached_files and len(ctx.attached_files) >= 2:
                steps = [
                    PipelineStep(
                        id="in_1",
                        uses="input/file",
                        outputs=["R1"],
                        with_params={"path": ctx.attached_files[0]},
                    ),
                    PipelineStep(
                        id="in_2",
                        uses="input/file",
                        outputs=["R2"],
                        with_params={"path": ctx.attached_files[1]},
                    ),
                    PipelineStep(id="alg_1", uses="algorithms/ndvi", reads=["R1"], outputs=["N1"]),
                    PipelineStep(id="alg_2", uses="algorithms/ndvi", reads=["R2"], outputs=["N2"]),
                    PipelineStep(id="alg_3", uses="algorithms/ndvi_change", reads=["N1", "N2"], outputs=["C1"]),
                    PipelineStep(id="post_1", uses="post/summarize", reads=["C1"], outputs=["S1"]),
                ]
                notes = "NDVI change between two uploads; summarizing results."
            else:
                steps = [
                    PipelineStep(id="data_1", uses="load/kahovka_raster", outputs=["R1"]),
                    PipelineStep(id="alg_1", uses="algorithms/ndvi", reads=["R1"], outputs=["N1"]),
                    PipelineStep(id="post_1", uses="post/summarize", reads=["N1"], outputs=["S1"]),
                ]
                notes = "NDVI map from Kahovka raster; summarizing results."
        elif flow == "ndvi_single":
            steps = [
                PipelineStep(id="data_1", uses=loader, outputs=["R1"]),
                PipelineStep(id="alg_1", uses="algorithms/ndvi", reads=["R1"], outputs=["N1"]),
                PipelineStep(id="post_1", uses="post/visualize_ndvi", reads=["N1"], outputs=["IMG1"]),
                PipelineStep(id="post_2", uses="post/summarize", reads=["N1"], outputs=["S1"]),
            ]
            notes = "Single-scene NDVI analysis."
        elif flow == "ndvi_explain":
            steps = [
                PipelineStep(id="data_1", uses=loader, outputs=["R1"]),
                PipelineStep(id="alg_1", uses="algorithms/ndvi", reads=["R1"], outputs=["N1"]),
                PipelineStep(id="post_1", uses="post/visualize_ndvi", reads=["N1"], outputs=["IMG1"]),
                PipelineStep(
                    id="agent_1",
                    uses="agent/llm_response",
                    reads=["N1"],
                    outputs=["T1"],
                    with_params={"prompt": ctx.prompt},
                ),
                PipelineStep(id="post_2", uses="post/summarize", reads=["T1"], outputs=["S1"]),
            ]
            notes = "NDVI with explanatory agent response."
        elif flow == "stats_single":
            steps = [
                PipelineStep(id="data_1", uses=loader, outputs=["R1"]),
                PipelineStep(id="viz_1", uses="post/visualize_raster", reads=["R1"], outputs=["IMG0"]),
                PipelineStep(id="alg_1", uses="algorithms/stats_basic", reads=["R1"], outputs=["T1"]),
                PipelineStep(id="post_1", uses="post/summarize", reads=["T1"], outputs=["S1"]),
            ]
            notes = "Basic distribution statistics + raster preview."
        elif flow == "stats_explain":
            steps = [
                PipelineStep(id="data_1", uses=loader, outputs=["R1"]),
                PipelineStep(id="viz_1", uses="post/visualize_raster", reads=["R1"], outputs=["IMG0"]),
                PipelineStep(id="alg_1", uses="algorithms/stats_basic", reads=["R1"], outputs=["T1"]),
                PipelineStep(
                    id="agent_1",
                    uses="agent/llm_response",
                    reads=["T1"],
                    outputs=["T2"],
                    with_params={"prompt": ctx.prompt},
                ),
                PipelineStep(id="post_1", uses="post/summarize", reads=["T2"], outputs=["S1"]),
            ]
            notes = "Distribution analysis with explanatory agent response."
        else:
            # If no specific tool flow, answer via agent directly using the prompt
            steps = [
                PipelineStep(
                    id="agent_1",
                    uses="agent/llm_response",
                    outputs=["T1"],
                    with_params={
                        "prompt": ctx.prompt,
                        "context": (ctx.options or {}).get("chat"),
                    },
                ),
                PipelineStep(id="post_1", uses="post/summarize", reads=["T1"], outputs=["S1"]),
            ]
            notes = "No matching toolchain; responding via LLM agent."

        return PipelinePlan(flow=flow, steps=steps, notes=notes)


def orchestrate_request(prompt: str, *, extras: dict[str, Any] | None = None) -> PipelinePlan:
    cu = ContextUnderstandingAgent()
    dm = DecisionMakingAgent()
    registry_path = Path(__file__).resolve().parents[1] / "capabilities" / "capabilities.yaml"
    planner = PlannerAgent(registry_path=registry_path)

    ctx = cu.analyze(prompt, extras)
    flow = dm.decide_flow(ctx)
    return planner.plan(flow, ctx)
