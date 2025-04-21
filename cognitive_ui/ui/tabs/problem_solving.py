#!/usr/bin/env python3
"""
Problem Solving & Queries Tab for Cognitive Digital Twin Interactive UI

This module contains the UI components for the Problem Solving & Queries tab.
"""

import streamlit as st
from typing import Optional

from cognitive_ui.cognitive_functions import CognitiveDigitalTwin
from cognitive_ui.config import UI_MAX_TEXT_LENGTH, SYNTHESIS_QUERY, UNCERTAINTY_QUERY
from cognitive_ui.utils import truncate_text


def display_problem_solving_tab(twin: Optional[CognitiveDigitalTwin] = None) -> None:
    """Display the Problem Solving & Queries tab content.

    Args:
        twin: The CognitiveDigitalTwin instance
    """
    # PROBLEM SOLVING & QUERIES TAB - interactive problem solving
    st.header("Problem Solving & Queries")
    st.caption("Analyze problems, generate solutions, and query the digital twin")
    st.divider()

    # Natural language query section with larger input field
    with st.container(border=True):
        st.subheader("Natural Language Query")
        st.markdown(
            "*Ask questions about the environmental data to receive detailed, scientifically-grounded answers.*"
        )

        # Provide query input field
        if "current_query" not in st.session_state:
            st.session_state.current_query = "What are the main environmental challenges in this area?"

        # Use a text area with adjustable height
        query = st.text_area(
            "Enter your question:", value=st.session_state.current_query, height=100, key="query_input"
        )

        # Update the session state with the new query
        st.session_state.current_query = query

        query_button = st.button("Process Query", key="query_btn")
        if query_button:
            with st.spinner("Processing query..."):
                if twin is not None:
                    response = twin.process_query(st.session_state.current_query)
                    # Truncate based on window size
                    if isinstance(UI_MAX_TEXT_LENGTH, dict) and "query" in UI_MAX_TEXT_LENGTH:
                        max_length = UI_MAX_TEXT_LENGTH["query"]
                    else:
                        max_length = 1200  # Default if not in config
                    response = truncate_text(response, max_length)
                    st.session_state.query_response = response
                else:
                    st.error("Twin not initialized. Cannot process query.")

    # Display response if available
    if st.session_state.get("query_response"):
        with st.expander("View Query Response", expanded=True):
            with st.container(border=True):
                st.markdown("##### 🔍 Query Analysis")
                st.markdown(st.session_state.query_response)

    # Intervention suggestions - always available
    st.divider()
    with st.container(border=True):
        st.subheader("Intervention Suggestions")
        st.markdown(
            "*Proposes evidence-based interventions to address environmental challenges based on scientific analysis.*"
        )
        interventions_button = st.button("Generate Intervention Suggestions", key="intervention_btn")
        if interventions_button:
            with st.spinner("Generating intervention suggestions..."):
                if twin is not None:
                    interventions = twin.suggest_interventions()
                    # Truncate based on window size
                    if isinstance(UI_MAX_TEXT_LENGTH, dict) and "interventions" in UI_MAX_TEXT_LENGTH:
                        max_length = UI_MAX_TEXT_LENGTH["interventions"]
                    else:
                        max_length = 1000  # Default if not in config
                    interventions = truncate_text(interventions, max_length)
                    st.session_state.interventions = interventions
                else:
                    st.error("Twin not initialized. Cannot suggest interventions.")

    # Display interventions if available
    if st.session_state.get("interventions"):
        with st.expander("View Intervention Suggestions", expanded=True):
            with st.container(border=True):
                st.markdown("##### 💡 Suggested Actions")
                st.markdown(st.session_state.interventions)

    # Advanced insights - two column layout
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Knowledge Synthesis")
            st.markdown("*Integrates information across multiple scientific domains for holistic understanding.*")

            synthesis_button = st.button("Synthesize Knowledge", key="synthesis_btn")
            if synthesis_button:
                with st.spinner("Synthesizing knowledge..."):
                    if twin is not None:
                        synthesis_query = SYNTHESIS_QUERY  # HACK
                        synthesis_response = twin.process_query(synthesis_query)
                        # Truncate based on window size
                        if isinstance(UI_MAX_TEXT_LENGTH, dict) and "synthesis" in UI_MAX_TEXT_LENGTH:
                            max_length = UI_MAX_TEXT_LENGTH["synthesis"]
                        else:
                            max_length = 800  # Default if not in config
                        synthesis_response = truncate_text(synthesis_response, max_length)
                        st.session_state.synthesis_response = synthesis_response
                    else:
                        st.error("Twin not initialized. Cannot generate synthesis.")

            # Display synthesis response if available
            if st.session_state.get("synthesis_response"):
                with st.expander("View Knowledge Synthesis", expanded=True):
                    with st.container(border=True):
                        st.markdown("##### 🧠 Synthesized Knowledge")
                        st.markdown(st.session_state.synthesis_response)

    with col2:
        with st.container(border=True):
            st.subheader("Uncertainty Communication")
            st.markdown("*Articulates confidence levels in findings and presents alternative explanations.*")

            uncertainty_button = st.button("Communicate Uncertainty", key="uncertainty_btn")
            if uncertainty_button:
                with st.spinner("Analyzing uncertainty..."):
                    if twin is not None:
                        uncertainty_query = UNCERTAINTY_QUERY  # HACK

                        uncertainty_response = twin.process_query(uncertainty_query)
                        # Truncate based on window size
                        if isinstance(UI_MAX_TEXT_LENGTH, dict) and "uncertainty" in UI_MAX_TEXT_LENGTH:
                            max_length = UI_MAX_TEXT_LENGTH["uncertainty"]
                        else:
                            max_length = 800  # Default if not in config
                        uncertainty_response = truncate_text(uncertainty_response, max_length)
                        st.session_state.uncertainty_response = uncertainty_response
                    else:
                        st.error("Twin not initialized. Cannot generate uncertainty analysis.")

    # Display uncertainty response if available
    if st.session_state.get("uncertainty_response"):
        with st.expander("View Uncertainty Analysis", expanded=True):
            with st.container(border=True):
                st.markdown("##### ⚖️ Confidence Assessment")
                st.markdown(st.session_state.uncertainty_response)
