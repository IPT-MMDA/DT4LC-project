import io
from typing import Literal

import streamlit as st
from matplotlib.figure import Figure


def display_centered_image(image_data: io.BytesIO, width: int | None = None) -> None:
    """Display an image centered in the container.

    Args:
        image_data: BytesIO object containing the image data
        width: Optional width to resize the image to
    """
    cols = st.columns([1, 3, 1])
    with cols[1]:
        if width:
            st.image(image_data, width=width)
        else:
            st.image(image_data)


def display_centered_chart(fig: Figure, width: int | None = None, in_sidebar: bool = False) -> None:
    """Centers and displays a matplotlib figure.

    Args:
        fig: The matplotlib figure to display
        width: Optional width to resize the image to
        in_sidebar: Whether the chart is being displayed in the sidebar
    """
    if in_sidebar:
        st.sidebar.pyplot(fig, use_container_width=True)
    else:
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.pyplot(fig, use_container_width=True)


def display_centered_metric(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    help: str | None = None,
    in_sidebar: bool = False,
) -> None:
    """Display a metric centered in the container.

    Args:
        label: The metric label
        value: The metric value
        delta: Optional delta text
        delta_color: Color scheme for delta ("normal", "inverse", or "off")
        help: Optional help text
        in_sidebar: Whether the metric is being displayed in the sidebar
    """
    if in_sidebar:
        if delta:
            st.sidebar.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help)
        else:
            st.sidebar.metric(label=label, value=value, help=help)
    else:
        cols = st.columns([1, 2, 1])
        with cols[1]:
            if delta:
                st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help)
            else:
                st.metric(label=label, value=value, help=help)
