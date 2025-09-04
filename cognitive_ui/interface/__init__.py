"""Interface layer between UI and orchestration runtime.

This package exposes small helper functions that UIs can use to invoke the
pipeline either in-process (direct call to orchestrator + DTA) or via the
external API server. The goal is to decouple UI concerns from execution
concerns so that we can swap UIs later (Streamlit now, JS later).
"""

from .controller import run_flow

__all__ = ["run_flow"]
