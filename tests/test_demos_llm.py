"""Demo-flow regression tests (Layer B) — full intent → plan path.

These tests call ``orchestrate(...)`` with the exact prompts used in the
rehearsed demo flow and assert that the resulting plan reaches the right
algorithm. They protect the natural-language intent recognition that
the demos rely on across future refactors.

Marked ``@pytest.mark.llm`` because parts of the orchestration path
(``context_agent``, sometimes the LLM-based intent classifier) require a
working LLM provider. CI deselects them via ``-m "not slow and not llm"``.
Engineers run them locally with::

    pytest -m llm tests/test_demos_llm.py

The **phrasing equivalences** the tests pin:

* "analyze vegetation changes"  ≡  "Analyze NDVI changes for these images"
  → both must reach the NDVI ChangeMap pipeline.
* "Analyze ice cover changes"
  → must reach the NDSI / Snow Classifier ChangeMap pipeline (the user
    never types "NDSI" — the planner has to make that mapping).
* "Analyze land-cover classes for this image"
  → must reach the LULC pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.schemas import Attachment, ChatRequest


def _attachment(path: str, idx: int) -> Attachment:
    return Attachment(
        id=f"att-{idx}",
        filename=Path(path).name,
        mime_type="image/tiff",
        path=path,
    )


def _plan_step_ids(result: dict) -> list[str]:
    """Extract the list of registry step IDs the plan dispatched to."""
    plan = result.get("plan") or {}
    return [step.get("uses", "") for step in plan.get("steps", [])]


def _plan_change_index_type(result: dict) -> str | None:
    """Return the IndexType bound on the change-detection step, if any."""
    plan = result.get("plan") or {}
    for step in plan.get("steps", []):
        if step.get("uses") == "algorithms/change-detection":
            return step.get("binds", {}).get("IndexType")
    return None


@pytest.mark.llm
class TestVegetationChangePhrasingEquivalence:
    """Pins the equivalence: implicit 'vegetation changes' ≡ explicit 'NDVI changes'.

    Both phrasings appear in the demo (Demo 1 step 3 and Demo 2.1). They must
    produce the same change-detection plan with IndexType=ndvi.
    """

    def test_implicit_vegetation_changes(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        req = ChatRequest(
            prompt="analyze vegetation changes",
            attachments=[_attachment(before, 1), _attachment(after, 2)],
        )
        result = orchestrate(req)
        assert result.get("ok"), result
        assert result.get("intent") == "pipeline"
        steps = _plan_step_ids(result)
        assert "algorithms/change-detection" in steps, steps
        assert _plan_change_index_type(result) == "ndvi"

    def test_explicit_ndvi_changes(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        req = ChatRequest(
            prompt="Analyze NDVI changes for these images",
            attachments=[_attachment(before, 1), _attachment(after, 2)],
        )
        result = orchestrate(req)
        assert result.get("ok"), result
        assert result.get("intent") == "pipeline"
        steps = _plan_step_ids(result)
        assert "algorithms/change-detection" in steps, steps
        assert _plan_change_index_type(result) == "ndvi"


@pytest.mark.llm
class TestIceCoverChangeMapping:
    """Demo 2.2: 'Analyze ice cover changes' must reach the NDSI pipeline.

    The user never types "NDSI" — the planner has to map "ice cover" →
    NDSI / snow-classifier. This pins the demo's natural-language routing
    across future planner-internals refactors.
    """

    def test_ice_cover_picks_ndsi(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        req = ChatRequest(
            prompt="Analyze ice cover changes",
            attachments=[_attachment(before, 1), _attachment(after, 2)],
        )
        result = orchestrate(req)
        assert result.get("ok"), result
        assert result.get("intent") == "pipeline"
        steps = _plan_step_ids(result)
        # Must reach the change-detection algorithm, NOT default-NDVI.
        assert "algorithms/change-detection" in steps, steps
        index_type = _plan_change_index_type(result)
        assert index_type == "ndsi", (
            f"'ice cover changes' must map to NDSI, not {index_type!r}. "
            "If this test fails after a refactor, the planner's index-type "
            "mapping for snow/ice keywords broke."
        )


@pytest.mark.llm
class TestLulcPhrasing:
    """Demo 1 steps 4-5: 'Analyze land-cover classes for this image'."""

    def test_lulc_phrasing_picks_lulc(self, synthetic_raster_path: str) -> None:
        req = ChatRequest(
            prompt="Analyze land-cover classes for this image",
            attachments=[_attachment(synthetic_raster_path, 1)],
        )
        result = orchestrate(req)
        assert result.get("ok"), result
        assert result.get("intent") == "pipeline"
        steps = _plan_step_ids(result)
        assert "algorithms/lulc-classifier" in steps, steps
