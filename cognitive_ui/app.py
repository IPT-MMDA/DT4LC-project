#!/usr/bin/env python3
"""
Cognitive Digital Twin Interactive UI

A Streamlit-based interactive UI for the Cognitive Digital Twin Framework.
This app provides a user-friendly interface to interact with the framework's
various components and visualize results.

Configuration:
- Streamlit-specific settings: .streamlit/config.toml
- Application settings: cognitive_ui/config.py
"""

import streamlit as st

from cognitive_ui.config import UI_TABS
from cognitive_ui.manager import (
    fix_kahovka_visualization,
    generate_synthetic_historical_data,
    initialize_twin,
)
from cognitive_ui.ui.components import display_sidebar
from cognitive_ui.ui.chat import display_chat_interface
from cognitive_ui.utils import debug_info


def main() -> None:
    """Main application entry point.

    Sets up the Streamlit UI, initializes the digital twin,
    and displays the interactive interface with visualization tabs.
    """
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

    # Initialize session state variables if they don't exist
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.data_source = "example_dataset"
        st.session_state.historical_data_added = False
        st.session_state.trigger_hist_gen = False
        st.session_state.current_query = "What are the main environmental challenges in this area?"
        debug_info("Session state initialized", "First run")

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

    # We no longer preload heavy twin computations. Sidebar still manages dataset selection
    # and optional uploads/adjustments.
    twin = None
    display_sidebar(twin)

    # Minimalist layout: chat only
    display_chat_interface()

    # No error if twin is None; orchestration runs in-process without preloading.


if __name__ == "__main__":
    main()
