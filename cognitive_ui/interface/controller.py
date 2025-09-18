from __future__ import annotations

from typing import Any

from cognitive_ui.coe_adapter import run_orchestrated_flow


def run_flow(
    prompt: str,
    *,
    selected_area: dict[str, Any] | None = None,
    attachments: list[str] | None = None,
    options: dict[str, Any] | None = None,
    via_server: bool = False,
) -> dict[str, Any]:
    """Run a pipeline flow.

    If `via_server` is False, runs in-process using the orchestrator + DTA.
    If True, this would call the external API (hook left for future JS UI).
    """

    if not via_server:
        extras: dict[str, Any] = {
            "selected_area": selected_area,
            "attached_files": attachments or [],
            "options": options or {},
        }
        # Unpack to satisfy the adapter signature without type ignores
        return run_orchestrated_flow(prompt, **extras)

    # Placeholder for future API call
    raise NotImplementedError("Server-based execution will be wired when API is enabled.")
