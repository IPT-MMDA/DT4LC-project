"""Minimal API server for orchestration flows.

Exposes a single POST /flow endpoint that accepts a JSON payload with fields:
  - prompt: str
  - selected_area: dict (optional)
  - attached_files: list[str] (optional)
  - options: dict (optional)

Returns a JSON structure containing the pipeline plan and execution summary.
"""
