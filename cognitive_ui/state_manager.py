#!/usr/bin/env python3
"""
Centralized Session State Management for Cognitive Digital Twin UI

This module provides a type-safe, centralized approach to managing Streamlit session state,
ensuring consistency, preventing errors, and making the codebase more maintainable.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
import streamlit as st

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin


class DataSource(Enum):
    """Enumeration of available data sources."""

    EXAMPLE = "example_dataset"
    KAHOVKA = "kahovka_data"
    UPLOAD = "upload_data"


class VisualizationOption(Enum):
    """Enumeration of visualization band combinations."""

    RGB = "Natural Color (RGB)"
    FALSE_COLOR = "False Color (NIR-R-G)"
    SWIR = "SWIR Composite"


@dataclass
class LayerVisibility:
    """Container for layer visibility states."""

    # Main layers
    physical: bool = True
    cognitive: bool = True
    interface: bool = True

    # Cognitive sub-layers
    scientific_interpretation: bool = True
    causal_reasoning: bool = True
    intervention_suggestions: bool = True

    # Interface sub-layers
    query_processing: bool = True
    knowledge_synthesis: bool = True
    uncertainty_communication: bool = True


@dataclass
class GeneratedContent:
    """Container for LLM-generated content."""

    interpretation: str | None = None
    causal_hypotheses: str | None = None
    interventions: str | None = None
    query_response: str | None = None
    synthesis_response: str | None = None
    uncertainty_response: str | None = None


@dataclass
class VisualizationData:
    """Container for visualization arrays."""

    current: NDArray[np.float32] | None = None
    current_rgb: NDArray[np.float32] | None = None
    current_false: NDArray[np.float32] | None = None
    current_swir: NDArray[np.float32] | None = None

    historical: NDArray[np.float32] | None = None
    historical_rgb: NDArray[np.float32] | None = None
    historical_false: NDArray[np.float32] | None = None
    historical_swir: NDArray[np.float32] | None = None

    # Special handling for Kahovka data
    kahovka: NDArray[np.float32] | None = None
    kahovka_rgb: NDArray[np.float32] | None = None
    kahovka_false: NDArray[np.float32] | None = None
    kahovka_swir: NDArray[np.float32] | None = None


@dataclass
class UploadState:
    """Container for file upload state."""

    current_data_path: str | None = None
    current_data_date: str = "2023-01-01"
    previous_upload_path: str | None = None
    user_crs: str | None = None

    historical_data_path: str | None = None
    historical_date: str = "2017-01-01"


T = TypeVar("T")


class SessionStateManager:
    """
    Centralized manager for Streamlit session state.

    This class provides type-safe access to session state with validation,
    default values, and clear documentation of all state variables.
    """

    # Define all state keys as class constants for easy reference
    INITIALIZED = "initialized"
    DATA_SOURCE = "data_source"
    TWIN = "twin"
    CURRENT_QUERY = "current_query"
    HISTORICAL_DATA_ADDED = "historical_data_added"
    TRIGGER_HIST_GEN = "trigger_hist_gen"
    IN_EMERGENCY_MODE = "in_emergency_mode"
    VIZ_OPTION = "viz_option"

    # Complex state objects
    LAYER_VISIBILITY = "layer_visibility"
    GENERATED_CONTENT = "generated_content"
    VISUALIZATION_DATA = "visualization_data"
    UPLOAD_STATE = "upload_state"

    def __init__(self):
        """Initialize the state manager."""
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure all required state variables are initialized with defaults."""
        if self.INITIALIZED not in st.session_state:
            # Core state
            st.session_state[self.INITIALIZED] = True
            st.session_state[self.DATA_SOURCE] = DataSource.EXAMPLE.value
            st.session_state[self.TWIN] = None
            st.session_state[self.CURRENT_QUERY] = "What are the main environmental challenges in this area?"
            st.session_state[self.HISTORICAL_DATA_ADDED] = False
            st.session_state[self.TRIGGER_HIST_GEN] = False
            st.session_state[self.IN_EMERGENCY_MODE] = False
            st.session_state[self.VIZ_OPTION] = VisualizationOption.RGB.value

            # Complex objects
            st.session_state[self.LAYER_VISIBILITY] = LayerVisibility()
            st.session_state[self.GENERATED_CONTENT] = GeneratedContent()
            st.session_state[self.VISUALIZATION_DATA] = VisualizationData()
            st.session_state[self.UPLOAD_STATE] = UploadState()

    # --- Core State Properties ---

    @property
    def is_initialized(self) -> bool:
        """Check if the session has been initialized."""
        return cast(bool, st.session_state.get(self.INITIALIZED, False))

    @property
    def data_source(self) -> DataSource:
        """Get the current data source."""
        value = st.session_state.get(self.DATA_SOURCE, DataSource.EXAMPLE.value)
        return DataSource(value)

    @data_source.setter
    def data_source(self, value: DataSource | str) -> None:
        """Set the current data source."""
        if isinstance(value, str):
            value = DataSource(value)
        st.session_state[self.DATA_SOURCE] = value.value

    @property
    def twin(self) -> CognitiveDigitalTwin | None:
        """Get the digital twin instance."""
        return cast(CognitiveDigitalTwin | None, st.session_state.get(self.TWIN))

    @twin.setter
    def twin(self, value: CognitiveDigitalTwin | None) -> None:
        """Set the digital twin instance."""
        st.session_state[self.TWIN] = value

    @property
    def current_query(self) -> str:
        """Get the current query text."""
        return cast(str, st.session_state.get(self.CURRENT_QUERY, ""))

    @current_query.setter
    def current_query(self, value: str) -> None:
        """Set the current query text."""
        st.session_state[self.CURRENT_QUERY] = value

    @property
    def has_historical_data(self) -> bool:
        """Check if historical data has been added."""
        return cast(bool, st.session_state.get(self.HISTORICAL_DATA_ADDED, False))

    @has_historical_data.setter
    def has_historical_data(self, value: bool) -> None:
        """Set the historical data flag."""
        st.session_state[self.HISTORICAL_DATA_ADDED] = value

    @property
    def trigger_historical_generation(self) -> bool:
        """Check if historical data generation should be triggered."""
        return cast(bool, st.session_state.get(self.TRIGGER_HIST_GEN, False))

    @trigger_historical_generation.setter
    def trigger_historical_generation(self, value: bool) -> None:
        """Set the historical generation trigger."""
        st.session_state[self.TRIGGER_HIST_GEN] = value

    @property
    def is_emergency_mode(self) -> bool:
        """Check if the app is in emergency mode."""
        return cast(bool, st.session_state.get(self.IN_EMERGENCY_MODE, False))

    @is_emergency_mode.setter
    def is_emergency_mode(self, value: bool) -> None:
        """Set emergency mode state."""
        st.session_state[self.IN_EMERGENCY_MODE] = value

    @property
    def visualization_option(self) -> VisualizationOption:
        """Get the current visualization option."""
        value = st.session_state.get(self.VIZ_OPTION, VisualizationOption.RGB.value)
        return VisualizationOption(value)

    @visualization_option.setter
    def visualization_option(self, value: VisualizationOption | str) -> None:
        """Set the visualization option."""
        if isinstance(value, str):
            value = VisualizationOption(value)
        st.session_state[self.VIZ_OPTION] = value.value

    # --- Complex State Objects ---

    @property
    def layers(self) -> LayerVisibility:
        """Get layer visibility settings."""
        return cast(LayerVisibility, st.session_state.get(self.LAYER_VISIBILITY, LayerVisibility()))

    @property
    def content(self) -> GeneratedContent:
        """Get generated content container."""
        return cast(GeneratedContent, st.session_state.get(self.GENERATED_CONTENT, GeneratedContent()))

    @property
    def visualizations(self) -> VisualizationData:
        """Get visualization data container."""
        return cast(VisualizationData, st.session_state.get(self.VISUALIZATION_DATA, VisualizationData()))

    @property
    def uploads(self) -> UploadState:
        """Get upload state container."""
        return cast(UploadState, st.session_state.get(self.UPLOAD_STATE, UploadState()))

    # --- Convenience Methods ---

    def get_current_visualization(self) -> NDArray[np.float32] | None:
        """Get the current visualization based on selected option and data source."""
        viz_data = self.visualizations
        viz_option = self.visualization_option

        # Special handling for Kahovka data
        if self.data_source == DataSource.KAHOVKA:
            if viz_option == VisualizationOption.RGB and viz_data.kahovka_rgb is not None:
                return viz_data.kahovka_rgb
            elif viz_option == VisualizationOption.FALSE_COLOR and viz_data.kahovka_false is not None:
                return viz_data.kahovka_false
            elif viz_option == VisualizationOption.SWIR and viz_data.kahovka_swir is not None:
                return viz_data.kahovka_swir
            return viz_data.kahovka

        # Regular data handling
        if viz_option == VisualizationOption.RGB and viz_data.current_rgb is not None:
            return viz_data.current_rgb
        elif viz_option == VisualizationOption.FALSE_COLOR and viz_data.current_false is not None:
            return viz_data.current_false
        elif viz_option == VisualizationOption.SWIR and viz_data.current_swir is not None:
            return viz_data.current_swir

        return viz_data.current

    def get_historical_visualization(self) -> NDArray[np.float32] | None:
        """Get the historical visualization based on selected option."""
        viz_data = self.visualizations
        viz_option = self.visualization_option

        if viz_option == VisualizationOption.RGB and viz_data.historical_rgb is not None:
            return viz_data.historical_rgb
        elif viz_option == VisualizationOption.FALSE_COLOR and viz_data.historical_false is not None:
            return viz_data.historical_false
        elif viz_option == VisualizationOption.SWIR and viz_data.historical_swir is not None:
            return viz_data.historical_swir

        return viz_data.historical

    def reset_twin(self) -> None:
        """Reset the digital twin and related state."""
        self.twin = None
        self.has_historical_data = False
        # Clear generated content
        st.session_state[self.GENERATED_CONTENT] = GeneratedContent()
        # Reset visualizations
        st.session_state[self.VISUALIZATION_DATA] = VisualizationData()

    def update_content(self, **kwargs: str | None) -> None:
        """Update generated content fields."""
        content = self.content
        for key, value in kwargs.items():
            if hasattr(content, key):
                setattr(content, key, value)
        st.session_state[self.GENERATED_CONTENT] = content

    def update_visualization(self, viz_type: str, data: NDArray[np.float32] | None) -> None:
        """
        Update a specific visualization.

        Args:
            viz_type: Type of visualization (e.g., 'current_rgb', 'historical_false')
            data: The visualization array
        """
        viz_data = self.visualizations
        if hasattr(viz_data, viz_type):
            setattr(viz_data, viz_type, data)
            st.session_state[self.VISUALIZATION_DATA] = viz_data

    def clear_generated_content(self) -> None:
        """Clear all generated content."""
        st.session_state[self.GENERATED_CONTENT] = GeneratedContent()

    # --- Legacy Compatibility Methods ---
    # These methods provide backward compatibility with existing code

    def get_legacy(self, key: str, default: Any = None) -> Any:
        """Get a value using legacy key format (for migration period)."""
        # Map legacy keys to new structure
        legacy_map = {
            "visualization": lambda: self.visualizations.current,
            "visualization_rgb": lambda: self.visualizations.current_rgb,
            "visualization_false": lambda: self.visualizations.current_false,
            "visualization_swir": lambda: self.visualizations.current_swir,
            "historical_visualization": lambda: self.visualizations.historical,
            "historical_visualization_rgb": lambda: self.visualizations.historical_rgb,
            "historical_visualization_false": lambda: self.visualizations.historical_false,
            "historical_visualization_swir": lambda: self.visualizations.historical_swir,
            "interpretation": lambda: self.content.interpretation,
            "causal_hypotheses": lambda: self.content.causal_hypotheses,
            "interventions": lambda: self.content.interventions,
            "query_response": lambda: self.content.query_response,
            "synthesis_response": lambda: self.content.synthesis_response,
            "uncertainty_response": lambda: self.content.uncertainty_response,
        }

        if key in legacy_map:
            return legacy_map[key]()
        return st.session_state.get(key, default)

    def set_legacy(self, key: str, value: Any) -> None:
        """Set a value using legacy key format (for migration period)."""
        # Map legacy keys to new setters
        if key.startswith("visualization") or key.startswith("historical_visualization"):
            self.update_visualization(key.replace("visualization", "current"), value)
        elif key in [
            "interpretation",
            "causal_hypotheses",
            "interventions",
            "query_response",
            "synthesis_response",
            "uncertainty_response",
        ]:
            self.update_content(**{key: value})
        else:
            st.session_state[key] = value


# Create a global instance for easy access
state = SessionStateManager()
