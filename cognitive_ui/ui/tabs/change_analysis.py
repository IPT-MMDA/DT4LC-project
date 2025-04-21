#!/usr/bin/env python3
"""
Change Analysis Tab for Cognitive Digital Twin Interactive UI

This module contains the UI components for the Change Analysis tab.
"""

import matplotlib.pyplot as plt
import streamlit as st
from typing import Literal

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin
from cognitive_ui.core.visualization import plot_to_image
from cognitive_ui.utils import truncate_text

from ..widgets_visualization import display_centered_image, display_centered_chart


def display_change_analysis_tab(twin: CognitiveDigitalTwin | None = None, has_historical: bool = False) -> None:
    """Display the Change Analysis tab content.

    Args:
        twin: The CognitiveDigitalTwin instance
        has_historical: Flag indicating if historical data is available
    """
    # CHANGE ANALYSIS TAB - components that need historical data
    st.header("Change Analysis")

    if has_historical:
        st.caption("Comparison and analysis of changes over time")
        st.divider()

        # Add visualization option selection
        viz_options = ["Natural Color (RGB)", "False Color (NIR-R-G)", "SWIR Composite"]
        current_viz = st.session_state.get("viz_option", "Natural Color (RGB)")

        # Create the selection UI
        selected_viz = st.radio(
            "Visualization Option",
            options=viz_options,
            index=viz_options.index(current_viz) if current_viz in viz_options else 0,
            horizontal=True,
            help="Select visualization bands to emphasize different features",
        )

        # Update session state if option changed
        if selected_viz != st.session_state.get("viz_option"):
            st.session_state.viz_option = selected_viz
            print(f"Visualization option changed to: {selected_viz}")
            # Update current visualization based on the selected option
            if selected_viz == "False Color (NIR-R-G)" and hasattr(st.session_state, "visualization_false"):
                st.session_state.visualization = st.session_state.visualization_false
            elif selected_viz == "SWIR Composite" and hasattr(st.session_state, "visualization_swir"):
                st.session_state.visualization = st.session_state.visualization_swir
            else:
                st.session_state.visualization = st.session_state.visualization_rgb

        # Side-by-side current and historical visualizations
        current_col, gap, historical_col = st.columns([10, 1, 10])

        # --- CURRENT STATE COLUMN ---
        with current_col:
            with st.container(border=True):
                st.subheader("Current State")
                if hasattr(st.session_state, "visualization"):
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(st.session_state.visualization)
                    # Dynamic title based on data source
                    title_parts = []
                    if st.session_state.data_source == "kahovka_data":
                        title_parts.append("Kahovka Dam")
                    elif st.session_state.data_source == "upload_data":
                        title_parts.append("Uploaded Data")
                    else:
                        title_parts.append("Current Imagery")
                    if hasattr(st.session_state, "viz_option"):
                        title_parts.append(f"({st.session_state.viz_option})")
                    ax.set_title(" ".join(title_parts))
                    ax.axis("off")
                    display_centered_image(plot_to_image(fig), width=None)

        # --- HISTORICAL STATE COLUMN ---
        with historical_col:
            with st.container(border=True):
                st.subheader("Historical State")

                # Get the current visualization option from session state
                viz_option = st.session_state.get("viz_option", "standard")

                # Debug print to see what's in session state
                vis_keys = [k for k in st.session_state.keys() if isinstance(k, str) and "visualization" in k]
                print(f"Visualization keys in session state: {vis_keys}")
                print(f"Current visualization option: {viz_option}")

                # Check for all historical visualization keys
                has_hist_rgb = (
                    "historical_visualization_rgb" in st.session_state
                    and st.session_state.historical_visualization_rgb is not None
                )
                has_hist_false = (
                    "historical_visualization_false" in st.session_state
                    and st.session_state.historical_visualization_false is not None
                )
                has_hist_swir = (
                    "historical_visualization_swir" in st.session_state
                    and st.session_state.historical_visualization_swir is not None
                )
                has_hist_base = (
                    "historical_visualization" in st.session_state
                    and st.session_state.historical_visualization is not None
                )

                print(
                    f"Has historical visualizations: RGB={has_hist_rgb}, False={has_hist_false}, SWIR={has_hist_swir}, Base={has_hist_base}"
                )

                # First check if any historical visualization exists
                if has_hist_rgb or has_hist_false or has_hist_swir or has_hist_base:
                    # Display the historical visualization based on the current visualization option
                    historical_vis_to_display = None

                    # Select appropriate visualization based on option
                    if viz_option == "False Color (NIR-R-G)" and has_hist_false:
                        historical_vis_to_display = st.session_state.historical_visualization_false
                        print("Using False Color historical visualization")
                    elif viz_option == "SWIR Composite" and has_hist_swir:
                        historical_vis_to_display = st.session_state.historical_visualization_swir
                        print("Using SWIR historical visualization")
                    elif has_hist_rgb:
                        # Default to the RGB historical visualization
                        historical_vis_to_display = st.session_state.historical_visualization_rgb
                        print("Using RGB historical visualization")
                    elif has_hist_base:
                        # Fall back to base historical visualization
                        historical_vis_to_display = st.session_state.historical_visualization
                        print("Using base historical visualization")

                    # If we found a valid visualization, display it
                    if historical_vis_to_display is not None:
                        print(
                            f"Displaying historical visualization with shape: {historical_vis_to_display.shape if hasattr(historical_vis_to_display, 'shape') else 'unknown'}"
                        )

                        # Create and display the visualization
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(historical_vis_to_display)

                        # Dynamic title based on data source and historical date
                        title_parts = []
                        if st.session_state.data_source == "kahovka_data":
                            title_parts.append("Kahovka Dam")
                        elif st.session_state.data_source == "upload_data":
                            title_parts.append("Uploaded Data")
                        else:
                            title_parts.append("Historical Imagery")

                        # Add visualization type
                        if viz_option != "standard":
                            title_parts.append(f"({viz_option})")

                        # Add date if available
                        if hasattr(st.session_state, "historical_date"):
                            title_parts.append(f"- {st.session_state.historical_date}")

                        ax.set_title(" ".join(title_parts))
                        ax.axis("off")
                        display_centered_image(plot_to_image(fig), width=None)
                    else:
                        # This should rarely happen, but just in case
                        print("WARNING: All historical visualization variables are None!")
                        st.warning("Historical data exists but visualizations couldn't be created properly.")
                        if st.button("Regenerate Historical Visualizations", key="regen_hist_vis"):
                            st.session_state.trigger_hist_gen = True
                            st.rerun()
                else:
                    print("WARNING: No historical visualization keys found in session state!")
                    st.warning(
                        "No historical visualization available. Please generate or upload historical data using the sidebar options."
                    )
                    # Add a direct button to generate historical data for better user experience
                    if st.button("Generate Sample Historical Data", key="gen_hist_tab"):
                        st.session_state.trigger_hist_gen = True
                        st.rerun()

        # --- DETECTED CHANGES SECTION ---
        st.divider()
        with st.container(border=True):
            st.subheader("Detected Changes")
            if (
                twin is not None
                and "physical_state" in twin.__dict__
                and "detected_changes" in twin.physical_state
                and twin.physical_state["detected_changes"]
            ):
                changes = twin.physical_state["detected_changes"]

                # Display changes info
                synthetic = any(change.get("synthetic", False) for change in changes)
                if synthetic:
                    st.info("Note: The changes below include synthetically generated examples for demonstration.")
                else:
                    st.caption("Based on analysis of current and historical imagery")

                # Create metric cards for each change
                num_changes = len(changes)
                cols_changes = st.columns(num_changes)

                for i, change in enumerate(changes):
                    with cols_changes[i]:
                        with st.container(border=True):
                            param_name = change.get("parameter", "N/A").replace("_", " ").title()
                            delta_val = change.get("change", 0)
                            change_delta_color: Literal["normal", "inverse", "off"] = "off"
                            if delta_val > 0:
                                change_delta_color = "inverse"  # Green for positive change
                            elif delta_val < 0:
                                change_delta_color = "normal"  # Red for negative change

                            st.metric(
                                label=f"{param_name}",
                                value=f"{change.get('current_value', 0):.2f}",
                                delta=f"{delta_val:.2f}",
                                delta_color=change_delta_color,
                                help=f"Change in {param_name.lower()} between historical and current states",
                            )
                            st.caption(f"Historical: {change.get('previous_value', 0):.2f}")
            else:
                st.info("No specific changes detected between the historical and current states.")

        # --- LAND COVER COMPARISON SECTION ---
        st.divider()
        with st.container(border=True):
            st.subheader("Land Cover Comparison")
            st.caption("Comparison of land cover distribution between current and historical data")

            # Check if we have land cover data in both current and historical states
            has_current_land_cover = (
                twin is not None
                and "physical_state" in twin.__dict__
                and "parameters" in twin.physical_state
                and "land_cover" in twin.physical_state["parameters"]
                and twin.physical_state["parameters"]["land_cover"]
            )

            has_historical_land_cover = False
            historical_land_cover = {}

            # Check for historical land cover and get it properly
            if (
                twin is not None
                and "physical_state" in twin.__dict__
                and "historical_states" in twin.physical_state
                and twin.physical_state["historical_states"]
            ):
                historical_state = twin.physical_state["historical_states"][-1]
                if "parameters" in historical_state and "land_cover" in historical_state["parameters"]:
                    historical_land_cover = historical_state["parameters"]["land_cover"]
                    has_historical_land_cover = bool(historical_land_cover)  # Check if not empty dict

            if has_current_land_cover and has_historical_land_cover:
                current_col, historical_col = st.columns(2)

                # Current land cover
                with current_col:
                    st.markdown("##### Current Land Cover")
                    if (
                        twin is not None
                        and "physical_state" in twin.__dict__
                        and "parameters" in twin.physical_state
                        and "land_cover" in twin.physical_state["parameters"]
                    ):
                        current_land_cover = twin.physical_state["parameters"]["land_cover"]
                    else:
                        current_land_cover = {}

                    # Use same land types for both charts to ensure consistency
                    land_types = list(set(list(current_land_cover.keys()) + list(historical_land_cover.keys())))

                    # Get values or 0 if the type doesn't exist in current data
                    cover_values = [current_land_cover.get(lt, 0) for lt in land_types]

                    fig, ax = plt.subplots(figsize=(5, 3))
                    colors = ["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"]
                    # Ensure we have enough colors by cycling
                    if len(land_types) > len(colors):
                        colors = colors * (len(land_types) // len(colors) + 1)

                    bars = ax.bar(land_types, cover_values, color=colors[: len(land_types)])

                    # Add data labels on top of each bar
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:  # Only add labels for non-zero values
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.5,
                                f"{height:.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                    ax.set_ylabel("Percentage (%)")
                    ax.set_ylim(0, max(cover_values) * 1.15 if cover_values else 10)  # Give space for labels
                    plt.xticks(rotation=45, ha="right")

                    # Use responsive chart display
                    display_centered_chart(fig, width=None, in_sidebar=False)

                # Historical land cover
                with historical_col:
                    st.markdown("##### Historical Land Cover")

                    # Get values or 0 if the type doesn't exist in historical data
                    hist_cover_values = [historical_land_cover.get(lt, 0) for lt in land_types]

                    fig, ax = plt.subplots(figsize=(5, 3))
                    # Use a different color scheme for historical data
                    hist_colors = ["#2C699A", "#48A9A6", "#54C6EB", "#69C1C7", "#83DCB6", "#A6EBC9"]
                    # Ensure we have enough colors by cycling
                    if len(land_types) > len(hist_colors):
                        hist_colors = hist_colors * (len(land_types) // len(hist_colors) + 1)

                    bars = ax.bar(land_types, hist_cover_values, color=hist_colors[: len(land_types)])

                    # Add data labels on top of each bar
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:  # Only add labels for non-zero values
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.5,
                                f"{height:.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                    ax.set_ylabel("Percentage (%)")
                    ax.set_ylim(0, max(hist_cover_values) * 1.15 if hist_cover_values else 10)  # Give space for labels
                    plt.xticks(rotation=45, ha="right")

                    # Use responsive chart display
                    display_centered_chart(fig, width=None, in_sidebar=False)

                    # Add date if available
                    if "timestamp" in historical_state:
                        st.caption(f"Date: {historical_state['timestamp']}")
                    elif hasattr(st.session_state, "historical_date"):
                        st.caption(f"Date: {st.session_state.historical_date}")
            else:
                st.info(
                    "Land cover comparison not available. Either current or historical land cover data is missing."
                )

        # --- CAUSAL REASONING ---
        st.divider()
        with st.container(border=True):
            st.subheader("Causal Reasoning")
            st.markdown(
                "*Identifies potential cause-effect relationships for observed environmental changes over time.*"
            )

            causal_button = st.button("Generate Causal Hypotheses", key="causal_btn")
            if causal_button:
                with st.spinner("Generating causal hypotheses..."):
                    if twin is not None:
                        causal_hypotheses = twin.generate_causal_hypothesis()
                        max_length = 800  # Default max length
                        causal_hypotheses = truncate_text(causal_hypotheses, max_length)
                        st.session_state.causal_hypotheses = causal_hypotheses
                    else:
                        st.error("Twin not initialized. Cannot generate causal hypotheses.")

            # Display causal hypotheses if available
            if (
                st.session_state.get("causal_hypotheses")
                and "Please add historical data" not in st.session_state.causal_hypotheses
            ):
                with st.expander("View Causal Reasoning", expanded=True):
                    with st.container(border=True):
                        st.markdown("##### 🔄 Causal Analysis")
                        st.markdown(st.session_state.causal_hypotheses)
    else:
        # If no historical data, show information on how to add it
        st.caption("Historical data required for temporal analysis")

        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("https://img.icons8.com/fluency/96/time-machine.png", width=80)
            with col2:
                st.subheader("Historical Data Required")
                st.markdown("""
                Temporal analysis requires historical data to be uploaded or generated.
                
                You can:
                * Upload historical satellite imagery using the sidebar
                * Generate synthetic historical data for demonstration
                """)

                # Add direct button to generate synthetic data in the temporal tab
                if st.button("Generate Sample Historical Data", key="gen_hist_temporal"):
                    st.session_state.trigger_hist_gen = True
                    st.rerun()

        # Set a warning message when no historical data
        if not has_historical:
            st.warning("No land cover data for historical state. Please generate or upload historical data.")
            st.caption("Historical data required for change analysis")
            # Add a direct button to generate historical data for better user experience
            if st.button("Generate Sample Historical Data", key="gen_hist_lcv"):
                st.session_state.trigger_hist_gen = True
                st.rerun()
            return

    # If there's no twin or not enough historical data, show a fallback message
    if twin is None or not has_historical:
        st.info(
            """Change analysis requires historical data to be uploaded or generated. 
               Please use the sidebar options to add historical data."""
        )
        # Add a direct button to generate historical data for better user experience
        if st.button("Generate Sample Historical Data", key="gen_hist_root"):
            st.session_state.trigger_hist_gen = True
            st.rerun()
