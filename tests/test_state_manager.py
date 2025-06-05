#!/usr/bin/env python3
"""
Unit tests for the SessionStateManager.

These tests ensure the state manager works correctly and maintains
backward compatibility during the migration period.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock streamlit before importing state manager
streamlit_mock = MagicMock()
streamlit_mock.session_state = {}


class TestSessionStateManager(unittest.TestCase):
    """Test cases for SessionStateManager."""

    def setUp(self):
        """Set up test environment before each test."""
        # Clear session state
        streamlit_mock.session_state.clear()

        # Patch streamlit module
        self.streamlit_patcher = patch("cognitive_ui.state_manager.st", streamlit_mock)
        self.streamlit_patcher.start()

        # Import after patching
        from cognitive_ui.state_manager import DataSource, SessionStateManager, VisualizationOption

        self.SessionStateManager = SessionStateManager
        self.DataSource = DataSource
        self.VisualizationOption = VisualizationOption

        # Create a fresh instance for each test
        self.state = SessionStateManager()

    def tearDown(self):
        """Clean up after each test."""
        self.streamlit_patcher.stop()

    def test_initialization(self):
        """Test that state manager initializes with correct defaults."""
        self.assertTrue(self.state.is_initialized)
        self.assertEqual(self.state.data_source, self.DataSource.EXAMPLE)
        self.assertIsNone(self.state.twin)
        self.assertEqual(self.state.current_query, "What are the main environmental challenges in this area?")
        self.assertFalse(self.state.has_historical_data)
        self.assertFalse(self.state.trigger_historical_generation)
        self.assertFalse(self.state.is_emergency_mode)
        self.assertEqual(self.state.visualization_option, self.VisualizationOption.RGB)

    def test_data_source_enum_handling(self):
        """Test data source property handles both enum and string values."""
        # Set using enum
        self.state.data_source = self.DataSource.KAHOVKA
        self.assertEqual(self.state.data_source, self.DataSource.KAHOVKA)

        # Set using string
        self.state.data_source = "upload_data"
        self.assertEqual(self.state.data_source, self.DataSource.UPLOAD)

    def test_visualization_option_handling(self):
        """Test visualization option property handles both enum and string values."""
        # Set using enum
        self.state.visualization_option = self.VisualizationOption.FALSE_COLOR
        self.assertEqual(self.state.visualization_option, self.VisualizationOption.FALSE_COLOR)

        # Set using string
        self.state.visualization_option = "SWIR Composite"
        self.assertEqual(self.state.visualization_option, self.VisualizationOption.SWIR)

    def test_complex_objects(self):
        """Test complex state objects (layers, content, visualizations, uploads)."""
        # Test layer visibility
        self.assertTrue(self.state.layers.physical)
        self.state.layers.physical = False
        self.assertFalse(self.state.layers.physical)

        # Test generated content
        self.assertIsNone(self.state.content.interpretation)
        self.state.update_content(interpretation="Test interpretation")
        self.assertEqual(self.state.content.interpretation, "Test interpretation")

        # Test upload state
        self.assertIsNone(self.state.uploads.current_data_path)
        self.state.uploads.current_data_path = "/tmp/test.tif"
        self.assertEqual(self.state.uploads.current_data_path, "/tmp/test.tif")

    def test_visualization_updates(self):
        """Test visualization update methods."""
        test_array = np.random.rand(224, 224, 3).astype(np.float32)

        # Update visualization
        self.state.update_visualization("current_rgb", test_array)
        np.testing.assert_array_equal(self.state.visualizations.current_rgb, test_array)

        # Test get_current_visualization
        current = self.state.get_current_visualization()
        np.testing.assert_array_equal(current, test_array)

    def test_get_current_visualization_logic(self):
        """Test the logic for getting current visualization based on options."""
        # Create test arrays
        rgb_array = np.ones((224, 224, 3), dtype=np.float32) * 1
        false_array = np.ones((224, 224, 3), dtype=np.float32) * 2
        swir_array = np.ones((224, 224, 3), dtype=np.float32) * 3

        # Set up visualizations
        self.state.update_visualization("current_rgb", rgb_array)
        self.state.update_visualization("current_false", false_array)
        self.state.update_visualization("current_swir", swir_array)

        # Test RGB selection
        self.state.visualization_option = self.VisualizationOption.RGB
        current = self.state.get_current_visualization()
        np.testing.assert_array_equal(current, rgb_array)

        # Test False Color selection
        self.state.visualization_option = self.VisualizationOption.FALSE_COLOR
        current = self.state.get_current_visualization()
        np.testing.assert_array_equal(current, false_array)

        # Test SWIR selection
        self.state.visualization_option = self.VisualizationOption.SWIR
        current = self.state.get_current_visualization()
        np.testing.assert_array_equal(current, swir_array)

    def test_kahovka_special_handling(self):
        """Test special handling for Kahovka data source."""
        # Set up Kahovka visualizations
        kahovka_rgb = np.ones((224, 224, 3), dtype=np.float32) * 10
        self.state.update_visualization("kahovka_rgb", kahovka_rgb)

        # Set data source to Kahovka
        self.state.data_source = self.DataSource.KAHOVKA

        # Should return Kahovka-specific visualization
        current = self.state.get_current_visualization()
        np.testing.assert_array_equal(current, kahovka_rgb)

    def test_reset_twin(self):
        """Test reset_twin clears appropriate state."""
        # Set up some state
        self.state.twin = MagicMock()
        self.state.has_historical_data = True
        self.state.update_content(interpretation="Some text")
        test_array = np.random.rand(224, 224, 3).astype(np.float32)
        self.state.update_visualization("current_rgb", test_array)

        # Reset
        self.state.reset_twin()

        # Check state was cleared
        self.assertIsNone(self.state.twin)
        self.assertFalse(self.state.has_historical_data)
        self.assertIsNone(self.state.content.interpretation)
        self.assertIsNone(self.state.visualizations.current_rgb)

    def test_update_content_multiple_fields(self):
        """Test updating multiple content fields at once."""
        self.state.update_content(
            interpretation="Test interp", causal_hypotheses="Test causal", interventions="Test interventions"
        )

        self.assertEqual(self.state.content.interpretation, "Test interp")
        self.assertEqual(self.state.content.causal_hypotheses, "Test causal")
        self.assertEqual(self.state.content.interventions, "Test interventions")

    def test_legacy_compatibility(self):
        """Test legacy compatibility methods."""
        # Test legacy get
        test_array = np.random.rand(224, 224, 3).astype(np.float32)
        self.state.update_visualization("current_rgb", test_array)

        # Should retrieve using legacy key
        retrieved = self.state.get_legacy("visualization_rgb")
        np.testing.assert_array_equal(retrieved, test_array)

        # Test legacy set for content
        self.state.set_legacy("interpretation", "Legacy text")
        self.assertEqual(self.state.content.interpretation, "Legacy text")

        # Test legacy set for direct session state
        self.state.set_legacy("some_custom_key", "custom_value")
        self.assertEqual(streamlit_mock.session_state["some_custom_key"], "custom_value")

    def test_clear_generated_content(self):
        """Test clearing all generated content."""
        # Set some content
        self.state.update_content(interpretation="Text 1", causal_hypotheses="Text 2", interventions="Text 3")

        # Clear
        self.state.clear_generated_content()

        # Check all cleared
        self.assertIsNone(self.state.content.interpretation)
        self.assertIsNone(self.state.content.causal_hypotheses)
        self.assertIsNone(self.state.content.interventions)
        self.assertIsNone(self.state.content.query_response)
        self.assertIsNone(self.state.content.synthesis_response)
        self.assertIsNone(self.state.content.uncertainty_response)


class TestStateManagerIntegration(unittest.TestCase):
    """Integration tests for state manager with mock Streamlit environment."""

    def setUp(self):
        """Set up integration test environment."""
        streamlit_mock.session_state.clear()
        self.streamlit_patcher = patch("cognitive_ui.state_manager.st", streamlit_mock)
        self.streamlit_patcher.start()

        from cognitive_ui.state_manager import state

        self.state = state

    def tearDown(self):
        """Clean up after integration tests."""
        self.streamlit_patcher.stop()

    def test_singleton_behavior(self):
        """Test that the global state instance behaves correctly."""
        # Import again to ensure we get the same instance
        from cognitive_ui.state_manager import state as state2

        # Set a value through one reference
        self.state.current_query = "Test query"

        # Should be visible through the other reference
        self.assertEqual(state2.current_query, "Test query")

    def test_persistence_across_reruns(self):
        """Test that state persists across simulated reruns."""
        # Set some state
        self.state.data_source = self.state.DataSource.KAHOVKA
        self.state.has_historical_data = True

        # Simulate a rerun by creating a new instance
        # (in real Streamlit, session_state persists)
        from cognitive_ui.state_manager import SessionStateManager

        new_state = SessionStateManager()

        # State should persist
        self.assertEqual(new_state.data_source.value, "kahovka_data")
        self.assertTrue(new_state.has_historical_data)


if __name__ == "__main__":
    unittest.main()
