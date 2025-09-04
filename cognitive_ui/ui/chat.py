from __future__ import annotations

from typing import Any

import streamlit as st

from cognitive_ui.config import TMP_DIR
from cognitive_ui.interface import run_flow


def _save_attachments(uploaded_files: list[Any]) -> list[str]:
    saved_paths: list[str] = []
    for f in uploaded_files:
        target = TMP_DIR / f.name
        with open(target, "wb") as out:
            out.write(f.getvalue())
        saved_paths.append(str(target))
    return saved_paths


def display_chat_interface() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Describe your analysis goal. Example: 'ndvi change on kahovka data'. "
                    "You can attach up to two rasters for before/after analysis."
                ),
            }
        ]

    st.markdown("#### Problem Solving")
    st.caption("Minimal, chat‑centric interaction with the Digital Twin.")

    # Controls
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Reset chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.context_signals = {}
            st.rerun()

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])  # content is short text; larger payloads shown separately

    with st.expander("Attachments (optional)", expanded=False):
        files = st.file_uploader(
            "Upload up to two GeoTIFFs (before/after)",
            type=["tif", "tiff", "geotiff"],
            accept_multiple_files=True,
            key="chat_file_uploader",
        )

    # Quick dataset switcher to steer planner (keeps sidebar intact but allows in-chat override)
    ds_choice = st.selectbox(
        "Dataset context",
        ["Kahovka (2023)", "Simple Prithvi Imagery (2017)"],
        index=0,
        help="Used to hint the planner which loader to use",
    )

    prompt = st.chat_input("Ask DT to analyze ...")
    if prompt:
        # Render user message immediately and persist in history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            # Progressive updates
            status = st.empty()
            status.markdown("Planning…")
            attach_paths = _save_attachments(files) if files else []
            if "context_signals" not in st.session_state:
                st.session_state.context_signals = {}
            options = {
                "dataset": "prithvi_example" if "Prithvi" in ds_choice else "kahovka",
                "chat": {"messages": st.session_state.chat_history[-10:], "signals": st.session_state.context_signals},
            }
            payload = run_flow(prompt, attachments=attach_paths, options=options)

            status.markdown("Executing…")
            plan = payload.get("plan", {})
            result = payload.get("result", {})
            artifacts = result.get("artifacts", {})

            # Live progress feed (per step)
            progress = result.get("progress", [])
            if progress:
                with st.expander("Execution progress", expanded=False):
                    for evt in progress:
                        if evt.get("event") == "start":
                            st.write(f"➡️ Start: {evt.get('id')} ({evt.get('uses')})")
                        elif evt.get("event") == "end":
                            st.write(f"✅ End: {evt.get('id')} ({evt.get('uses')})")

            # Display NDVI visualization if present
            img = artifacts.get("IMG1", {}).get("image") if isinstance(artifacts.get("IMG1"), dict) else None
            if img is not None:
                st.image(img, caption="NDVI visualization", use_container_width=True)

            # Show raster preview if present for stats flow
            base_img = artifacts.get("IMG0", {}).get("image") if isinstance(artifacts.get("IMG0"), dict) else None
            if base_img is not None:
                st.image(base_img, caption="Dataset preview", use_container_width=True)

            # Summarize
            summary = None
            if isinstance(artifacts.get("S1"), dict):
                summary = artifacts["S1"].get("summary")
            short = summary or "Pipeline executed. See plan below."
            st.write(short)

            # If stats histogram present, render compact chart
            if isinstance(artifacts.get("T1"), dict) and "histogram" in artifacts["T1"]:
                hist = artifacts["T1"]["histogram"]
                try:
                    import altair as alt  # noqa: F401
                    import pandas as pd  # noqa: F401

                    df = pd.DataFrame({
                        "bin": hist.get("bins", [])[1:],
                        "count": hist.get("counts", []),
                    })
                    chart = alt.Chart(df).mark_bar().encode(x="bin:Q", y="count:Q").properties(height=120)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass
            status.empty()

            # Update context signals for LLM context in subsequent turns
            st.session_state.context_signals = st.session_state.get("context_signals", {})
            st.session_state.context_signals.update({
                "last_summary": short,
                "has_ndvi_image": bool(img is not None),
                "has_raster_preview": bool(base_img is not None),
                "has_histogram": bool(isinstance(artifacts.get("T1"), dict) and "histogram" in artifacts["T1"]),
            })

            with st.expander("View pipeline plan", expanded=False):
                st.json(plan)

            st.session_state.chat_history.append({"role": "assistant", "content": short})
