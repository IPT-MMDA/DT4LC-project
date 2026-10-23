"""Plan proposal and confirmation (v0.6.0 story).

A plan that is not a single registered template (band math, or a new combination
of steps) is offered as a proposal before it runs. Named indices no built-in
analysis computes (NDMI, NBR, ...) are composed from a table without an LLM, or
refused when the files lack a band: the committed sample has B2–B6, so NBR (B7)
is refused and NDMI (B5, B6) is proposed.

LLM paths use a scripted router; every other test points the real router at a
closed Ollama port, so no test reaches a live provider.
"""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dta.config import ROOT_DIR
from dta.dti.coe import planner as planner_mod
from dta.dti.coe.llm import LLMResponse, get_llm_router, reset_llm_router
from dta.dti.coe.llm.ollama import OllamaProvider
from dta.dti.coe.named_indices import NAMED_INDICES, match_named_index, proposal_text
from dta.dti.coe.orchestrator import NO_ANALYSIS_STEP, orchestrate
from dta.dti.registry import load_registry
from dta.dti.schemas import Attachment, ChatRequest, ExecutionPlan

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")

NDMI_PROMPT = "Map moisture change with NDMI between the two dates"
NDMI_TEXT = (
    "I don't have a moisture-change algorithm. I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) "
    "from B5 and B6 for both dates and difference them — run that?"
)
NDMI_FORMULA = "(B5 - B6) / (B5 + B6)"
BURN_MESSAGE = (
    "Burn severity uses NBR = (NIR − SWIR2)/(NIR + SWIR2), which needs band B7 (SWIR2). "
    "The sample has B2–B6, so I can't compute it here."
)
POST = "post-processing/agent-analysis"


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Real router with Ollama on a closed port as its only provider: every generate() raises."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    reset_llm_router()
    assert all(isinstance(p, OllamaProvider) for p in get_llm_router().providers)
    yield
    reset_llm_router()


def _attachments(*paths: str) -> list[Attachment]:
    return [Attachment(id=f"a{i}", filename=Path(p).name, mime_type="image/tiff", path=p) for i, p in enumerate(paths)]


class ScriptedLLM:
    """Router stub answering each COE stage from its system prompt: intent, context, plan."""

    providers: list[Any] = []

    def __init__(self, plan_steps: list[dict[str, Any]], context: dict[str, Any] | None = None) -> None:
        self.plan_steps = plan_steps
        self.context = context or {"goal": "x", "desired_outputs": [], "required_inputs": ["RasterPath"]}
        self.stages: list[str] = []

    def generate(self, messages: Any, *args: Any, **kwargs: Any) -> LLMResponse:
        system = messages[0].content
        if system.startswith("You are an Intent Classifier"):
            stage, text = "intent", json.dumps({"intent": "pipeline", "reason": "analysis request"})
        elif system.startswith("You are a Context Understanding Agent"):
            stage, text = "context", json.dumps({**self.context, "hints": {"keywords": []}})
        elif system.startswith("You are a pipeline planner"):
            stage, text = "plan", json.dumps({"steps": self.plan_steps, "reasoning": "scripted"})
        else:
            stage, text = "other", "Scripted interpretation."
        self.stages.append(stage)
        return LLMResponse(text=text, model="scripted", provider="scripted")


def _with_llm(llm: ScriptedLLM) -> Any:
    """Patch every COE stage's router with ``llm``."""
    from contextlib import ExitStack

    stack = ExitStack()
    for module in ("intent_classifier", "context_agent", "llm_planner"):
        stack.enter_context(patch(f"dta.dti.coe.{module}.get_llm_router", return_value=llm))
    return stack


NBR_PLAN = [
    {"uses": "input/file-before"},
    {"uses": "input/file-after"},
    {"uses": "algorithms/band-math-change", "config": {"formula": "(B5 - B7) / (B5 + B7)", "name": "NBR"}},
    {"uses": POST},
]


# --- The named-index table -------------------------------------------------------------


def test_named_index_table() -> None:
    table = {i.name: i.formula.format(red="B4", nir="B5", swir1="B6", swir2="B7") for i in NAMED_INDICES}
    assert table == {
        "NDMI": "(B5 - B6) / (B5 + B6)",
        "NBR": "(B5 - B7) / (B5 + B7)",
        "SAVI": "1.5 * (B5 - B4) / (B5 + B4 + 0.5)",
        "NDBI": "(B6 - B5) / (B6 + B5)",
    }


@pytest.mark.parametrize(
    ("prompt", "name"),
    [
        (NDMI_PROMPT, "NDMI"),
        ("Map soil moisture", "NDMI"),
        ("Map burn severity", "NBR"),
        ("Where did it burn?", "NBR"),
        ("compute NBR", "NBR"),
        ("Map moisture with NDWI", None),  # a built-in index is named: not a composed request
        ("Detect water changes between these images", None),
        ("Map the burnished roofs", None),  # whole words only
    ],
)
def test_prompt_names_an_index(prompt: str, name: str | None) -> None:
    index = match_named_index(prompt)
    assert (index.name if index else None) == name


# --- Proposals from the table (no LLM) ---------------------------------------------------


def test_ndmi_on_the_two_sample_files_is_a_band_math_change_proposal() -> None:
    out = orchestrate(ChatRequest(prompt=NDMI_PROMPT, attachments=_attachments(SAMPLE_0601, SAMPLE_0609)))

    assert out["ok"], out.get("error")
    assert out["proposal"] == NDMI_TEXT
    plan = out["plan"]
    assert plan["composed"] is True
    assert [s["uses"] for s in plan["steps"]] == [
        "input/file-before",
        "input/file-after",
        "algorithms/band-math-change",
        POST,
    ]
    assert plan["steps"][0]["binds"] == {"RasterPathBefore": SAMPLE_0601}
    assert plan["steps"][1]["binds"] == {"RasterPathAfter": SAMPLE_0609}
    assert plan["steps"][2]["config"] == {"formula": NDMI_FORMULA, "name": "NDMI", "threshold": 0.2}
    assert proposal_text(ExecutionPlan(**plan), load_registry()) == NDMI_TEXT


def test_ndmi_on_one_file_is_a_single_date_proposal() -> None:
    out = orchestrate(ChatRequest(prompt="Map moisture with NDMI", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["proposal"] == (
        "I don't have a moisture algorithm. "
        "I can compute NDMI = (NIR − SWIR1)/(NIR + SWIR1) from B5 and B6 — run that?"
    )
    assert [s["uses"] for s in out["plan"]["steps"]] == ["input/file", "algorithms/band-math", POST]
    assert out["plan"]["steps"][1]["config"] == {"formula": NDMI_FORMULA, "name": "NDMI"}


# --- Burn severity is refused (B7) ------------------------------------------------------


@pytest.mark.parametrize("files", [[SAMPLE_0601], [SAMPLE_0601, SAMPLE_0609]], ids=["one-file", "two-files"])
def test_burn_severity_on_the_sample_is_refused_without_llm(files: list[str]) -> None:
    with patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(ChatRequest(prompt="Map burn severity", attachments=_attachments(*files)))

    assert out["ok"] is False
    assert out["error"] == BURN_MESSAGE
    assert "NBR" in json.dumps(out["candidate"])
    template.assert_not_called()  # refused before planning: never an unrelated template


def test_burn_severity_is_refused_with_an_llm_that_proposes_nbr() -> None:
    llm = ScriptedLLM(NBR_PLAN)
    with _with_llm(llm), patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(ChatRequest(prompt="Map burn severity", attachments=_attachments(SAMPLE_0601, SAMPLE_0609)))

    assert out["ok"] is False
    assert out["error"] == BURN_MESSAGE
    assert "plan" not in llm.stages  # the table refuses before the LLM planner
    template.assert_not_called()


def test_llm_plan_naming_a_missing_band_stops_without_template_retry() -> None:
    """An LLM PlanError for a band the file lacks is final: no template plan runs instead."""
    llm = ScriptedLLM(NBR_PLAN)
    with _with_llm(llm), patch.object(planner_mod, "plan_template", wraps=planner_mod.plan_template) as template:
        out = orchestrate(
            ChatRequest(prompt="Compute (B5 - B7) / (B5 + B7)", attachments=_attachments(SAMPLE_0601, SAMPLE_0609))
        )

    assert "plan" in llm.stages
    assert out["ok"] is False
    assert out["error"].startswith("Step algorithms/band-math-change: Band B7 is not in this file (B2–B6).")
    template.assert_not_called()


# --- Composed LLM plans ----------------------------------------------------------------------


def test_llm_band_math_plan_is_a_proposal_with_a_generic_text() -> None:
    steps = [
        {"uses": "input/file"},
        {"uses": "algorithms/band-math", "config": {"formula": "(B4 - B3) / (B4 + B3)"}},
        {"uses": POST},
    ]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="Compute (B4 - B3) / (B4 + B3)", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is True
    assert out["proposal"] == (
        "No built-in analysis matches this. I can compute (B4 - B3) / (B4 + B3) from B4 and B3 — run that?"
    )


def test_llm_plan_combining_two_analyses_is_a_proposal() -> None:
    steps = [{"uses": "input/file"}, {"uses": "algorithms/ndvi"}, {"uses": "algorithms/statistics"}, {"uses": POST}]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="NDVI then the statistics", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is True
    assert out["proposal"].startswith("No single built-in analysis matches this. I can run ")


def test_llm_plan_matching_a_template_runs_without_proposal() -> None:
    steps = [{"uses": "input/file"}, {"uses": "algorithms/ndvi"}, {"uses": POST}]
    with _with_llm(ScriptedLLM(steps)):
        out = orchestrate(ChatRequest(prompt="Show plant vigour", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert out["plan"]["composed"] is False
    assert "proposal" not in out


# --- A plan with no analysis step is not accepted -------------------------------------------


def test_plan_without_analysis_step_is_not_accepted() -> None:
    out = orchestrate(ChatRequest(prompt="Show me the landscape", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"] is False
    assert out["error"] == NO_ANALYSIS_STEP
    assert [s["uses"] for s in out["candidate"]["steps"]] == ["input/file", POST]


def test_skeletal_llm_plan_falls_back_to_the_template() -> None:
    """HD fallback: an LLM plan that only reads and summarises is replaced by the template plan."""
    ctx = {"goal": "calculate NDVI", "desired_outputs": ["NDVIMap"], "required_inputs": ["RasterPath"]}
    with _with_llm(ScriptedLLM([{"uses": "input/file"}, {"uses": POST}], context=ctx)):
        out = orchestrate(ChatRequest(prompt="calculate NDVI please", attachments=_attachments(SAMPLE_0601)))

    assert out["ok"], out.get("error")
    assert [s["uses"] for s in out["plan"]["steps"]] == ["input/file", "algorithms/ndvi", POST]
    assert out["plan"]["composed"] is False
