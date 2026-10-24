"""Operator settings: ``GET/PUT /v1/settings`` and ``POST /v1/settings/assistant/test``.

Only operators (signed in with an ``OPERATOR_EMAILS`` address, or any caller under
``DEV_ROLE=operator``) reach these routes. For everyone else they do not match at all, so
the answer is the same 404 as for an unknown path, whatever the method or body.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.routing import Match

from ..schemas import SettingsResponse, SettingsTestResponse, SettingsUpdate
from ..sessions import CurrentSession, Session
from ..settings import apply_update, probe_assistant, settings_document


class OperatorRoute(APIRoute):  # type: ignore[misc]
    """A route that only an operator's request matches (the session middleware runs before routing)."""

    def matches(self, scope: dict[str, Any]) -> tuple[Match, dict[str, Any]]:
        session: Session | None = scope.get("state", {}).get("session")
        if session is None or not session.operator:
            return Match.NONE, {}
        result: tuple[Match, dict[str, Any]] = super().matches(scope)
        return result


router = APIRouter(prefix="/v1/settings", tags=["settings"], route_class=OperatorRoute)


@router.get("", response_model=SettingsResponse, summary="Server settings (operators)")  # type: ignore[misc]
async def get_settings() -> SettingsResponse:
    """Assistant, models, access, data sources and server; ``editable`` marks what ``PUT`` accepts.

    API keys appear only as ``key_present``. 404 for guests and members.
    """
    doc = await asyncio.to_thread(settings_document)
    return SettingsResponse.model_validate(doc)  # type: ignore[no-any-return]


@router.put("", response_model=SettingsResponse, summary="Change server settings (operators)")  # type: ignore[misc]
async def put_settings(update: SettingsUpdate, session: CurrentSession) -> SettingsResponse:
    """Apply a partial document and return the new ``GET``; ``null`` drops a field's override.

    Values are written to the overrides file (``DT4LC_SETTINGS_PATH``), never ``.env``, and
    take effect at once. 422 for an unknown or read-only key, a wrong type, a value out of
    range, or an ``operator_emails`` that drops the signed-in caller's own address (``null``
    included, when the env ``OPERATOR_EMAILS`` it falls back to lacks it).
    """
    access = update.access
    kept: list[str] | None = access.operator_emails if access is not None else None
    if access is not None and kept is None and "operator_emails" in access.model_fields_set:
        # null drops the override ("Reset to .env"), so the env list is the one that applies
        kept = [e.strip().lower() for e in os.environ.get("OPERATOR_EMAILS", "").split(",") if e.strip()]
    if session.user is not None and access is not None and kept is not None and session.user.email.lower() not in kept:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", "access", "operator_emails"),
                    "msg": f"Keep your own address ({session.user.email}): without it you lose operator access",
                    "input": access.operator_emails,
                }
            ]
        )
    await asyncio.to_thread(apply_update, update)
    return await get_settings()  # type: ignore[no-any-return]


@router.post(  # type: ignore[misc]
    "/assistant/test",
    response_model=SettingsTestResponse,
    summary="Probe the assistant's providers (operators)",
)
async def test_assistant() -> SettingsTestResponse:
    """Send a one-word prompt to every provider of the router, in order and concurrently.

    Each probe reports ``ok``, ``latency_ms`` and, on failure, ``error`` with keys redacted;
    a provider that does not answer within 20 s fails.
    """
    return SettingsTestResponse.model_validate(await probe_assistant())  # type: ignore[no-any-return]
