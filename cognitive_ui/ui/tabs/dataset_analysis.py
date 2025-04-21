#!/usr/bin/env python3
"""
Dataset Analysis Tab for Cognitive Digital Twin Interactive UI

This module contains the UI components for the Dataset Analysis tab.
"""

from typing import Literal

import matplotlib.pyplot as plt
import streamlit as st

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin
from cognitive_ui.config import UI_MAX_TEXT_LENGTH
from cognitive_ui.core.visualization import plot_to_image
from cognitive_ui.utils import truncate_text

from ..widgets_visualization import display_centered_chart, display_centered_image, display_centered_metric


def display_dataset_analysis_tab(twin: CognitiveDigitalTwin | None = None) -> None:
    """Display the Dataset Analysis tab content.

    Args:
        twin: The CognitiveDigitalTwin instance
    """
    # DATASET ANALYSIS TAB - components that work with current satellite data only
    st.header("Dataset Analysis")
    st.caption("Analysis of the most recent satellite imagery")
    st.divider()

    # Current satellite imagery and land cover
    with st.container(border=True):
        st.subheader("Satellite Imagery")
        # Display only the current visualization part from physical layer
        if hasattr(st.session_state, "visualization"):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(st.session_state.visualization)
            # Dynamic title based on data source and viz option
            title_parts = []
            if st.session_state.data_source == "kahovka_data":
                title_parts.append("Kahovka Dam")
            elif st.session_state.data_source == "upload_data":
                title_parts.append("Uploaded Data")
            else:
                title_parts.append("Satellite Imagery")
            if hasattr(st.session_state, "viz_option"):
                title_parts.append(f"({st.session_state.viz_option})")
            ax.set_title(" ".join(title_parts))
            ax.axis("off")

            # Use centered image display with percentage width instead of fixed pixels
            display_centered_image(plot_to_image(fig), width=None)

    # Land cover and vegetation indices - two-column layout
    if twin is not None and "physical_state" in twin.__dict__ and "parameters" in twin.physical_state:
        params = twin.physical_state["parameters"]

        # Create two columns for the metrics
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("Land Cover Distribution")
                if "land_cover" in params and params["land_cover"]:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    land_types = list(params["land_cover"].keys())
                    cover_values = list(params["land_cover"].values())

                    # Use a visually appealing color scheme
                    bars = ax.bar(
                        land_types,
                        cover_values,
                        color=["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"],
                    )

                    # Add data labels on top of each bar
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.5,
                            f"{height:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

                    ax.set_ylabel("Percentage (%)")
                    ax.set_ylim(0, max(cover_values) * 1.15)  # Give space for labels
                    plt.xticks(rotation=45, ha="right")

                    # Use responsive chart display
                    display_centered_chart(fig, width=None, in_sidebar=False)
                else:
                    st.info("No land cover data available.")

        with col2:
            with st.container(border=True):
                st.subheader("Vegetation Indices")
                if "vegetation_indices" in params and params["vegetation_indices"]:
                    # Display vegetation indices with interpretation where applicable
                    for index, value in params["vegetation_indices"].items():
                        if index.lower() == "ndvi":
                            # Add interpretation hint for NDVI
                            if value > 0.5:
                                interpretation = "Healthy vegetation"
                                ndvi_delta_color: Literal["normal", "inverse", "off"] = "normal"  # Green
                            elif value > 0.2:
                                interpretation = "Moderate vegetation"
                                ndvi_delta_color = "off"  # Gray
                            else:
                                interpretation = "Sparse vegetation"
                                ndvi_delta_color = "inverse"  # Red

                            display_centered_metric(
                                label=f"{index.upper()}",
                                value=f"{value:.4f}",
                                delta=interpretation,
                                delta_color=ndvi_delta_color,
                                help="Normalized Difference Vegetation Index measures plant health and density",
                                in_sidebar=False,
                            )
                        else:
                            # Regular display for other indices
                            display_centered_metric(
                                label=index.upper(),
                                value=f"{value:.4f}",
                                help=f"Vegetation index value for {index.upper()}",
                                in_sidebar=False,
                            )
                else:
                    st.info("No vegetation indices available.")
    else:
        st.info("No physical parameters (land cover, indices) extracted for this dataset.")

    # Scientific Interpretation - always available for current data
    st.divider()
    with st.container(border=True):
        st.subheader("Scientific Interpretation")
        st.markdown(
            "*Analyzes current satellite data to provide scientific explanations of observed patterns and phenomena.*"
        )
        interpret_button = st.button("Generate Scientific Interpretation", key="interpret_btn")
        if interpret_button:
            with st.spinner("Generating scientific interpretation..."):
                if twin is not None:
                    interpretation = twin.generate_scientific_interpretation()
                    if isinstance(UI_MAX_TEXT_LENGTH, dict) and "interpretation" in UI_MAX_TEXT_LENGTH:
                        max_length = UI_MAX_TEXT_LENGTH["interpretation"]
                    else:
                        max_length = 800  # Default if not in config
                    interpretation = truncate_text(interpretation, max_length)
                    st.session_state.interpretation = interpretation
                else:
                    st.error("Twin not initialized. Cannot generate interpretation.")

    # Display interpretation if available
    if st.session_state.get("interpretation"):
        with st.expander("View Scientific Interpretation", expanded=True):
            with st.container(border=True):
                st.markdown("##### 🔍 Scientific Analysis")
                st.markdown(st.session_state.interpretation)
