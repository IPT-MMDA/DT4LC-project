"""Tests for intent classifier.

Tests the classification of user requests into PIPELINE vs CONVERSATION intents.
"""

from dta.dti.coe.intent_classifier import (
    IntentType,
    _is_clear_action_request,
    _looks_like_action,
    classify_intent,
)
from dta.dti.schemas import Attachment, ChatRequest


class TestLooksLikeAction:
    """Tests for _looks_like_action helper function."""

    def test_action_keywords_detected(self) -> None:
        """Test that action keywords are detected."""
        action_prompts = [
            "calculate ndvi",
            "compute statistics",
            "detect boundaries",
            "extract statistics",
            "analyze vegetation",
            "run change detection",
            "process the snow data",
            "generate statistics",
        ]

        for prompt in action_prompts:
            assert _looks_like_action(prompt), f"Should detect action in: '{prompt}'"

    def test_non_action_prompts(self) -> None:
        """Test that non-action prompts return False."""
        non_action_prompts = [
            "hello",
            "thank you",
            "goodbye",
        ]

        for prompt in non_action_prompts:
            assert not _looks_like_action(prompt), f"Should not detect action in: '{prompt}'"


class TestIsClearActionRequest:
    """Tests for _is_clear_action_request helper function."""

    def test_clear_action_patterns(self) -> None:
        """Test that clear action patterns are recognized."""
        clear_actions = [
            "ndvi calculation",
            "calculate ndvi",
            "ndvi",
            "ndvi analysis",
            "detect boundaries",
            "change detection",
            "run change detection",
            "field boundaries",
            "extract statistics",
            "prithvi reconstruction",
        ]

        for prompt in clear_actions:
            assert _is_clear_action_request(prompt), f"Should be clear action: '{prompt}'"

    def test_question_prompts_not_actions(self) -> None:
        """Test that question prompts are not classified as actions."""
        questions = [
            "what is ndvi?",
            "how does change detection work?",
            "explain ndvi calculation",
            "what can we do next?",
            "where are the results?",
            "why did it fail?",
        ]

        for prompt in questions:
            assert not _is_clear_action_request(prompt), f"Should not be action: '{prompt}'"

    def test_ambiguous_prompts_not_clear_actions(self) -> None:
        """Test that ambiguous prompts are not classified as clear actions."""
        ambiguous = [
            "ndvi for this area would be nice",
            "I want to analyze something",
            "show me the vegetation data",
        ]

        for prompt in ambiguous:
            # These should NOT be clear actions (they go to LLM)
            assert not _is_clear_action_request(prompt), f"Should not be clear action: '{prompt}'"


class TestClassifyIntent:
    """Tests for classify_intent function."""

    def test_pipeline_with_attachments_and_action(self) -> None:
        """Test that requests with attachments and action keywords are PIPELINE."""
        att = Attachment(id="test", filename="test.tif", mime_type="image/tiff", path="/tmp/test.tif")
        req = ChatRequest(prompt="calculate ndvi", attachments=[att])

        result = classify_intent(req)

        assert result["intent"] == IntentType.PIPELINE
        assert "attachments" in result.get("reason", "").lower() or "action" in result.get("reason", "").lower()

    def test_clear_action_without_attachments_asks_for_file(self) -> None:
        """Test that clear action requests without attachments return CONVERSATION with helpful response."""
        clear_actions = [
            "calculate ndvi",
            "detect boundaries",
        ]

        for prompt in clear_actions:
            req = ChatRequest(prompt=prompt, attachments=[])
            result = classify_intent(req)

            # When an action is requested without a file, we return CONVERSATION
            # with a helpful response asking for the file
            assert result["intent"] == IntentType.CONVERSATION, f"'{prompt}' should be CONVERSATION (no file)"
            assert "response" in result, f"'{prompt}' should include helpful response"

    def test_conversation_questions(self) -> None:
        """Test that capability questions are classified as CONVERSATION.

        These prompts use quick classification patterns (capability questions)
        so they work without LLM calls.
        """
        # Use capability question patterns that are handled by quick classification
        capability_questions = [
            "can you calculate ndvi?",
            "could you detect field boundaries?",
            "are you able to run change detection?",
            "what can you do?",
        ]

        for prompt in capability_questions:
            req = ChatRequest(prompt=prompt, attachments=[])
            result = classify_intent(req)

            assert result["intent"] == IntentType.CONVERSATION, f"'{prompt}' should be CONVERSATION"
            assert result.get("response"), f"'{prompt}' should include helpful response"

    def test_conversation_has_response(self) -> None:
        """Test that CONVERSATION intent includes a response."""
        req = ChatRequest(prompt="what can we do with this data?", attachments=[])
        result = classify_intent(req)

        if result["intent"] == IntentType.CONVERSATION:
            assert result.get("response"), "CONVERSATION should include a response"


class TestClassifyIntentEdgeCases:
    """Edge case tests for intent classification."""

    def test_empty_prompt(self) -> None:
        """Test handling of empty prompt."""
        req = ChatRequest(prompt="", attachments=[])
        result = classify_intent(req)

        # Should not crash and should return a valid intent
        assert result["intent"] in [IntentType.PIPELINE, IntentType.CONVERSATION]

    def test_mixed_case_prompts_with_attachments(self) -> None:
        """Test that classification is case-insensitive when attachments are present."""
        att = Attachment(id="test", filename="test.tif", mime_type="image/tiff", path="/tmp/test.tif")
        prompts = [
            "CALCULATE NDVI",
            "Calculate Ndvi",
            "calculate NDVI",
        ]

        for prompt in prompts:
            req = ChatRequest(prompt=prompt, attachments=[att])
            result = classify_intent(req)

            assert result["intent"] == IntentType.PIPELINE, f"'{prompt}' should be PIPELINE with attachment"

    def test_whitespace_handling_with_attachment(self) -> None:
        """Test that extra whitespace is handled with attachments."""
        att = Attachment(id="test", filename="test.tif", mime_type="image/tiff", path="/tmp/test.tif")
        req = ChatRequest(prompt="  ndvi calculation  ", attachments=[att])
        result = classify_intent(req)

        assert result["intent"] == IntentType.PIPELINE
