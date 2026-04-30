"""Tests for the registry-driven TriggerIndex.

The load-bearing capability: adding a new item to ``registry.yaml`` with
populated ``triggers`` and ``user_guide`` fields must extend the classifier's
recognised vocabulary **without any code edit**. These tests synthesise an
in-memory ``Registry`` containing a fictional item and assert the classifier
picks it up — both via the ``TriggerIndex`` directly and through the public
``classify_intent`` entrypoint. If these tests pass, the registry is the
single source of truth for which prompts route where.
"""

from __future__ import annotations

import pytest

from dta.dti.coe.intent_classifier import IntentType, classify_intent
from dta.dti.coe.triggers import (
    GENERIC_ACTION_VERBS,
    TriggerIndex,
    is_capability_question,
    reset_trigger_index,
)
from dta.dti.schemas import (
    Attachment,
    ChatRequest,
    Registry,
    RegistryItem,
    Runner,
    Triggers,
    UserGuide,
)


def _fake_registry_with_extra_item() -> Registry:
    """Build a Registry that adds a fictional algorithm to the real catalog.

    Uses real registry types/items so the input/file passthrough still works,
    plus an in-memory fictional ``algorithms/fictional-index`` entry whose
    triggers and responses don't appear anywhere in the real registry.yaml.
    """
    from dta.dti.registry import load_registry

    real = load_registry()
    fictional = RegistryItem(
        id="algorithms/fictional-index",
        kind="algorithm",
        display_name="Fictional Vegetation Index",
        keywords=["fviX"],
        inputs=["RasterPath"],
        outputs=["NDVIMap"],
        runner=Runner(type="python", entrypoint="never/loaded.py"),
        triggers=Triggers(
            keywords=["fviX", "fictional-index", "fictional vegetation"],
            action_phrases=["calculate fviX", "compute fictional-index"],
        ),
        user_guide=UserGuide(
            capability_response="Yes, I can compute the fictional FVIX index for testing.",
            missing_file_response="Upload a fake GeoTIFF for the fictional FVIX index.",
        ),
    )
    return Registry(
        version=real.version,
        types=real.types,
        instances=[*real.instances, fictional],
    )


@pytest.fixture
def fake_idx() -> TriggerIndex:
    """TriggerIndex over a registry that includes a fictional new item.

    Doesn't touch the cached ``get_trigger_index()`` singleton — those tests
    use the real registry.
    """
    return TriggerIndex.from_registry(_fake_registry_with_extra_item())


class TestExtensibilityViaTriggerIndex:
    """No-code-change extension: new YAML triggers feed straight into TriggerIndex."""

    def test_new_keyword_recognized_by_has_keyword(self, fake_idx: TriggerIndex) -> None:
        # The fictional keyword does not appear anywhere in the real registry.yaml.
        assert fake_idx.has_keyword("calculate fviX")
        assert fake_idx.has_keyword("show me the fictional vegetation map")

    def test_new_action_phrase_recognized_as_clear_action(self, fake_idx: TriggerIndex) -> None:
        assert fake_idx.is_clear_action("calculate fviX")
        assert fake_idx.is_clear_action("compute fictional-index")
        # The bare keyword is also a clear action ("ndvi" works the same way).
        assert fake_idx.is_clear_action("fictional-index")

    def test_new_item_returned_by_find_matching_item(self, fake_idx: TriggerIndex) -> None:
        item = fake_idx.find_matching_item("calculate fviX please")
        assert item is not None
        assert item.id == "algorithms/fictional-index"

    def test_new_item_capability_response_used(self, fake_idx: TriggerIndex) -> None:
        # Render a capability response for a prompt that maps to the fictional item.
        response = fake_idx.render_capability_response("can you calculate fviX?")
        assert "fictional FVIX" in response, (
            f"capability_response should come from the new item's user_guide; got: {response!r}"
        )

    def test_new_item_missing_file_response_used(self, fake_idx: TriggerIndex) -> None:
        response = fake_idx.render_missing_file_response("calculate fviX")
        assert "fake GeoTIFF" in response, f"missing_file_response should come from the new item; got: {response!r}"

    def test_render_system_prompt_lists_new_item(self, fake_idx: TriggerIndex) -> None:
        prompt = fake_idx.render_system_prompt()
        assert "Fictional Vegetation Index" in prompt, (
            "render_system_prompt should enumerate every user-runnable item from the registry"
        )

    def test_generic_verbs_are_constant(self) -> None:
        # GENERIC_ACTION_VERBS belongs to the English layer, not the registry layer:
        # the same set must apply regardless of which items are registered.
        idx_real = TriggerIndex.from_registry(_fake_registry_with_extra_item())
        for verb in GENERIC_ACTION_VERBS:
            assert verb in idx_real.all_keywords


class TestExtensibilityViaPublicClassifier:
    """End-to-end: classify_intent picks up new items via the cached singleton."""

    def test_classifier_recognizes_new_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A new YAML item must be recognized by classify_intent without any code change."""
        # Swap the registry-loader the cached TriggerIndex builder uses.
        monkeypatch.setattr("dta.dti.coe.triggers.load_registry", _fake_registry_with_extra_item)
        reset_trigger_index()
        try:
            att = Attachment(id="t", filename="a.tif", mime_type="image/tiff", path="/tmp/a.tif")
            req = ChatRequest(prompt="calculate fviX", attachments=[att])
            result = classify_intent(req)
            assert result["intent"] == IntentType.PIPELINE, result
        finally:
            reset_trigger_index()

    def test_classifier_uses_new_capability_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A capability question matching a new item returns that item's user_guide response."""
        monkeypatch.setattr("dta.dti.coe.triggers.load_registry", _fake_registry_with_extra_item)
        reset_trigger_index()
        try:
            req = ChatRequest(prompt="can you calculate fviX?", attachments=[])
            result = classify_intent(req)
            assert result["intent"] == IntentType.CONVERSATION, result
            assert "fictional FVIX" in result["response"], (
                "classify_intent should surface the new item's capability_response unchanged"
            )
        finally:
            reset_trigger_index()


class TestQuestionFraming:
    """is_capability_question is generic English mechanics, not algorithm-specific."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "can you calculate ndvi?",
            "could you detect boundaries?",
            "are you able to run change detection?",
            "is it possible to compute statistics?",
            "what can you do?",
            "do you support land cover classification?",
            "so, can you analyze this image?",
        ],
    )
    def test_capability_questions(self, prompt: str) -> None:
        assert is_capability_question(prompt)

    @pytest.mark.parametrize(
        "prompt",
        [
            "calculate ndvi",
            "detect field boundaries",
            "ndvi calculation",
            "what is ndvi",  # explanation, not capability
            "how does ndvi work",
        ],
    )
    def test_non_capability_questions(self, prompt: str) -> None:
        assert not is_capability_question(prompt)
