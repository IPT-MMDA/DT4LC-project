#!/usr/bin/env python3
"""
Cognitive Digital Twin Interactive UI - Refactored with State Manager

This is an example of how app.py would look after refactoring to use
the centralized SessionStateManager.
"""

import streamlit as st

from cognitive_ui.config import UI_TABS
from cognitive_ui.manager import (
    fix_kahovka_visualization,
    generate_synthetic_historical_data,
    initialize_twin,
)
from cognitive_ui.state_manager import DataSource, state
from cognitive_ui.ui.components import display_sidebar
from cognitive_ui.ui.tabs.change_analysis import display_change_analysis_tab
from cognitive_ui.ui.tabs.dataset_analysis import display_dataset_analysis_tab
from cognitive_ui.ui.tabs.problem_solving import display_problem_solving_tab
from cognitive_ui.utils import debug_info


def main() -> None:
    """Main application entry point with refactored state management."""
    st.set_page_config(
        page_title="Cognitive Digital Twin",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Display global title and logos
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
            <h1 style='text-align: left; margin-bottom: 2rem;'>
                DT4LC: Developing Scalable Digital Twin Models for Land Cover Change Detection Using Machine Learning
            </h1>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.image("assets/7cd1dc0d1.png", width=100)
        st.image("assets/logo_q5_fin5.png", width=100)

    # State is automatically initialized by the state manager
    # No need for manual initialization checks
    debug_info("Session initialized", f"Data source: {state.data_source.value}")

    # Custom styling
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize the digital twin
    twin = initialize_twin()

    # Fix Kahovka visualization if needed
    fix_kahovka_visualization()

    # Check for synthetic historical data generation trigger
    if state.trigger_historical_generation and twin is not None:
        success = generate_synthetic_historical_data(twin)
        if success:
            debug_info("Generated synthetic historical data", "Success")
        # Clear trigger flag
        state.trigger_historical_generation = False

    # Auto-generate historical data if needed
    elif (
        twin is not None
        and state.visualizations.historical is None
        and state.visualizations.historical_rgb is None
        and not state.has_historical_data
    ):
        debug_info("Auto-generating historical data", "Missing visualizations")
        success = generate_synthetic_historical_data(twin)
        if success:
            debug_info("Auto-generated historical data", "Success")
            state.has_historical_data = True

    # Display the sidebar first (this will set up the environment and data)
    display_sidebar(twin)

    # Get historical data state for use in tabs
    has_historical = state.has_historical_data

    # Define tabs using the centralized configuration
    tab1, tab2, tab3 = st.tabs([UI_TABS["DATASET_ANALYSIS"], UI_TABS["CHANGE_ANALYSIS"], UI_TABS["PROBLEM_SOLVING"]])

    # Display content in each tab
    with tab1:
        display_dataset_analysis_tab(twin)

    with tab2:
        display_change_analysis_tab(twin, has_historical)

    with tab3:
        display_problem_solving_tab(twin)

    if twin is None:
        st.error("Unable to initialize the Cognitive Digital Twin. Please try refreshing the page.")


def display_sidebar_refactored(twin) -> None:
    """Example of refactored sidebar using state manager."""
    with st.sidebar:
        # Dataset selection
        with st.expander("🗃️ Dataset", expanded=True):
            # Store the previous data source to detect changes
            previous_data_source = state.data_source

            # Let user select data source using enum values
            current_data_source = st.radio(
                "Select data source",
                options=[ds.value for ds in DataSource],
                format_func=lambda x: {
                    DataSource.KAHOVKA.value: "Kahovka Dam (2023)",
                    DataSource.EXAMPLE.value: "Sample Prithvi Imagery (2017)",
                    DataSource.UPLOAD.value: "Upload Your Own Data",
                }[x],
                index=[ds.value for ds in DataSource].index(state.data_source.value),
            )

            # Convert string back to enum
            selected_source = DataSource(current_data_source)

            # If data source has changed, reset twin and update session state
            if previous_data_source != selected_source:
                debug_info("Data source changed", f"From {previous_data_source.value} to {selected_source.value}")
                state.data_source = selected_source

                # Handle Kahovka data source change immediately for better UX
                if selected_source == DataSource.KAHOVKA and state.visualizations.kahovka_rgb is not None:
                    # Update visualization without full twin reset for faster UI update
                    state.update_visualization("current", state.visualizations.kahovka_rgb)
                    debug_info("Updated visualization immediately for Kahovka", "Success")

                # Only reset if we're not in emergency mode
                if not state.is_emergency_mode:
                    state.reset_twin()
                    st.rerun()
                else:
                    st.warning("Currently in emergency mode. Please click Reset Twin to change data source.")


def display_dataset_analysis_tab_refactored(twin) -> None:
    """Example of refactored dataset analysis tab."""
    st.header("Dataset Analysis")
    st.caption("Analysis of the most recent satellite imagery")
    st.divider()

    # Current satellite imagery and land cover
    with st.container(border=True):
        st.subheader("Satellite Imagery")

        # Use state manager to get current visualization
        current_viz = state.get_current_visualization()
        if current_viz is not None:
            import matplotlib.pyplot as plt

            from cognitive_ui.core.visualization import plot_to_image
            from cognitive_ui.ui.widgets_visualization import display_centered_image

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(current_viz)

            # Dynamic title based on data source and viz option
            title_parts = []
            if state.data_source == DataSource.KAHOVKA:
                title_parts.append("Kahovka Dam")
            elif state.data_source == DataSource.UPLOAD:
                title_parts.append("Uploaded Data")
            else:
                title_parts.append("Satellite Imagery")

            title_parts.append(f"({state.visualization_option.value})")
            ax.set_title(" ".join(title_parts))
            ax.axis("off")

            display_centered_image(plot_to_image(fig), width=None)

    # Scientific Interpretation section
    st.divider()
    with st.container(border=True):
        st.subheader("Scientific Interpretation")
        st.markdown(
            "*Analyzes current satellite data to provide scientific explanations of observed patterns and phenomena.*"
        )

        if st.button("Generate Scientific Interpretation", key="interpret_btn"):
            with st.spinner("Generating scientific interpretation..."):
                if twin is not None:
                    from cognitive_ui.config import UI_MAX_TEXT_LENGTH
                    from cognitive_ui.utils import truncate_text

                    interpretation = twin.generate_scientific_interpretation()
                    max_length = UI_MAX_TEXT_LENGTH.get("interpretation", 800)
                    interpretation = truncate_text(interpretation, max_length)

                    # Use state manager to update content
                    state.update_content(interpretation=interpretation)
                else:
                    st.error("Twin not initialized. Cannot generate interpretation.")

    # Display interpretation if available
    if state.content.interpretation:
        with st.expander("View Scientific Interpretation", expanded=True):
            with st.container(border=True):
                st.markdown("##### 🔍 Scientific Analysis")
                st.markdown(state.content.interpretation)


if __name__ == "__main__":
    main()
