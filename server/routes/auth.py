"""GitHub sign-in and sign-out (``/v1/auth``).

``/github/login`` redirects to GitHub with a ``state`` that a short-lived signed cookie also
holds; ``/github/callback`` checks it, exchanges the code, reads the primary verified email
and decides the role from the allowlists. An allowed user gets a new signed-in session (the
guest session's jobs and uploads move to the user id) and lands on ``/app``; anyone else
lands on ``/app?signin=denied&reason=...`` or ``/app?signin=error`` as a guest.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hmac
import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .. import sessions
from ..github_oauth import GitHubConfig, OAuthError, authorize_url, fetch_identity, github_config
from ..jobs import get_job_queue
from ..schemas import AuthStatusResponse, ErrorDetail, ErrorResponse, SessionResponse
from ..sessions import (
    CurrentSession,
    User,
    cookie_secure,
    new_session,
    role_for_email,
    session_info,
    set_session,
    sign,
)
from ..uploads import adopt_uploads
from ..utils import UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])

STATE_COOKIE = "dt4lc_oauth_state"
STATE_PATH = "/v1/auth/github"  # sent to login and callback only
STATE_TTL_S = 600

_NOT_CONFIGURED = {404: {"model": ErrorResponse, "description": "SIGN_IN_NOT_CONFIGURED"}}


def _not_configured() -> JSONResponse:
    error = ErrorDetail(code="SIGN_IN_NOT_CONFIGURED", message="Sign-in is not configured on this server")
    return JSONResponse(ErrorResponse(error=error).model_dump(mode="json"), status_code=404)


def _encode_state(state: str) -> str:
    payload = f"{state}.{int(sessions.clock()) + STATE_TTL_S}"
    return f"{payload}.{sign(payload)}"


def _state_matches(cookie: str | None, state: str) -> bool:
    """Whether the state cookie is ours, unexpired, and holds ``state``."""
    payload, _, signature = (cookie or "").rpartition(".")
    expected, _, expires = payload.rpartition(".")
    if not payload or not hmac.compare_digest(signature, sign(payload)):
        return False
    return expires.isdigit() and int(expires) > sessions.clock() and hmac.compare_digest(expected, state)


def _back_to_app(config: GitHubConfig, query: dict[str, str] | None = None) -> RedirectResponse:
    url = f"{config.redirect_base}/app" + (f"?{urlencode(query)}" if query else "")
    response = RedirectResponse(url, status_code=302)
    response.delete_cookie(STATE_COOKIE, path=STATE_PATH)
    return response


@router.get("/status", response_model=AuthStatusResponse, summary="Sign-in status")  # type: ignore[misc]
async def auth_status(session: CurrentSession) -> AuthStatusResponse:
    """Whether GitHub sign-in is configured, the signed-in user (or ``null``) and the role."""
    return AuthStatusResponse(
        configured=github_config() is not None,
        user=asdict(session.user) if session.user else None,
        role=session.role,
    )


@router.get(  # type: ignore[misc]
    "/github/login",
    status_code=302,
    response_model=None,
    summary="Start GitHub sign-in",
    responses={302: {"description": "To GitHub's authorize page"}, **_NOT_CONFIGURED},
)
async def github_login() -> RedirectResponse | JSONResponse:
    """302 to GitHub's authorize URL (``scope=read:user user:email``, a fresh ``state``), with the
    ``state`` also in a signed ``dt4lc_oauth_state`` cookie for 10 minutes. 404
    ``SIGN_IN_NOT_CONFIGURED`` without ``GITHUB_CLIENT_ID``, ``GITHUB_CLIENT_SECRET`` and ``OAUTH_REDIRECT_BASE``.
    """
    config = github_config()
    if config is None:
        return _not_configured()
    state = secrets.token_urlsafe(24)
    response = RedirectResponse(authorize_url(config, state), status_code=302)
    response.set_cookie(
        STATE_COOKIE,
        _encode_state(state),
        max_age=STATE_TTL_S,
        path=STATE_PATH,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
    )
    return response


@router.get(  # type: ignore[misc]
    "/github/callback",
    status_code=302,
    response_model=None,
    summary="GitHub sign-in callback",
    responses={302: {"description": "To /app, with ?signin=denied|error when not signed in"}, **_NOT_CONFIGURED},
)
async def github_callback(
    request: Request,
    session: CurrentSession,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse | JSONResponse:
    """Finish sign-in and redirect to the app.

    - Allowed (``OPERATOR_EMAILS`` → operator; ``ALLOWED_EMAILS`` or ``ALLOWED_DOMAINS`` → member):
      a new session cookie with the user; a guest session's jobs and uploads move to the user id; ``/app``.
    - Primary email not verified: ``/app?signin=denied&reason=email_unverified``.
    - Not on an allowlist: ``/app?signin=denied&reason=not_allowlisted&email=<primary email>``.
    - A missing or wrong ``state``, a GitHub error or a failed exchange: ``/app?signin=error``.
    """
    config = github_config()
    if config is None:
        return _not_configured()
    if error or not code or not state or not _state_matches(request.cookies.get(STATE_COOKIE), state):
        logger.warning(f"GitHub sign-in refused: {error or 'missing code or state, or state mismatch'}")
        return _back_to_app(config, {"signin": "error"})
    try:
        identity = await asyncio.to_thread(fetch_identity, config, code)
    except OAuthError as exc:
        logger.warning(f"GitHub sign-in failed: {exc}")
        return _back_to_app(config, {"signin": "error"})
    email = identity.verified_email
    if email is None:
        logger.info(f"GitHub sign-in denied for {identity.login}: primary email not verified")
        return _back_to_app(config, {"signin": "denied", "reason": "email_unverified"})
    role = role_for_email(email)
    if role is None:
        logger.info(f"GitHub sign-in denied for {identity.login}: not on the allowlist")
        return _back_to_app(config, {"signin": "denied", "reason": "not_allowlisted", "email": email})

    user = User(
        id=f"github:{identity.id}",
        login=identity.login,
        name=identity.name,
        email=email,
        avatar_url=identity.avatar_url,
    )
    if session.user is None:
        jobs = await get_job_queue().adopt(session.id, user.id, role)
        uploads = await asyncio.to_thread(adopt_uploads, UPLOAD_DIR, session.id, user.id, role)
        logger.info(f"{identity.login} signed in as {role}; adopted {jobs} jobs and {uploads} uploads")
    set_session(request, replace(new_session(), role=role, user=user))
    return _back_to_app(config)


@router.post("/logout", response_model=SessionResponse, summary="Sign out")  # type: ignore[misc]
async def logout(request: Request) -> SessionResponse:
    """Replace the caller's session with a fresh guest session; returns it as ``GET /v1/session`` does."""
    fresh = set_session(request, new_session())
    return SessionResponse.model_validate(session_info(fresh))  # type: ignore[no-any-return]
