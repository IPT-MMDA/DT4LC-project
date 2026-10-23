"""Sessions, roles and the guest analysis quota.

Every request carries a signed, ``httpOnly`` session cookie (:data:`COOKIE_NAME`); the
middleware issues one when the request has none or a cookie that fails the check, and
puts the :class:`Session` on ``request.state.session``. Until sign-in exists every
session is a guest; ``DEV_ROLE`` overrides the role of every request.

Ownership: jobs and uploads record the session id as ``owner``. A caller sees only its
own; operators see everything. Rows without an owner (created before v0.6.0) are
visible to operators only.

Quota: guests get ``GUEST_ANALYSES_PER_HOUR`` analyses per rolling hour, counted per
session and per client IP; :func:`enforce_analysis_quota` refuses when either is used up.
The counts live in memory, so a restart resets them.
"""

from __future__ import annotations

import base64
from collections import deque
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Annotated, Any, Literal

from fastapi import Depends, Request
from starlette.datastructures import MutableHeaders

logger = logging.getLogger(__name__)

Role = Literal["guest", "member", "operator"]
ROLES: tuple[Role, ...] = ("guest", "member", "operator")
COOKIE_NAME = "dt4lc_session"
WINDOW_S = 3600.0  # the guest quota counts analyses in the last hour

# Wall clock for session expiry and the quota window; tests replace it.
clock: Callable[[], float] = time.time

# Used when SESSION_SECRET is unset: sessions then reset when the server restarts.
_FALLBACK_SECRET = secrets.token_bytes(32)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        logger.warning(f"{name} is not an integer; using {default}")
        return default


def _env_list(name: str) -> frozenset[str]:
    return frozenset(item.strip().lower() for item in os.environ.get(name, "").split(",") if item.strip())


def guest_analyses_per_hour() -> int:
    return _env_int("GUEST_ANALYSES_PER_HOUR", 5)


def guest_session_hours() -> int:
    return _env_int("GUEST_SESSION_HOURS", 24)


def dev_role() -> Role | None:
    """``DEV_ROLE`` if it names a role, else ``None`` (an invalid value is ignored)."""
    value = os.environ.get("DEV_ROLE", "").strip().lower()
    return value if value in ROLES else None  # type: ignore[return-value]


def cookie_secure() -> bool:
    return os.environ.get("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")


def _secret() -> bytes:
    value = os.environ.get("SESSION_SECRET", "")
    return value.encode() if value else _FALLBACK_SECRET


@dataclass(frozen=True)
class SignInConfig:
    """Allowlists for the sign-in story (Phase 5); nothing reads them yet. Lower-cased."""

    allowed_emails: frozenset[str]
    allowed_domains: frozenset[str]
    operator_emails: frozenset[str]


def sign_in_config() -> SignInConfig:
    """``ALLOWED_EMAILS``, ``ALLOWED_DOMAINS`` and ``OPERATOR_EMAILS``, comma-separated."""
    return SignInConfig(
        allowed_emails=_env_list("ALLOWED_EMAILS"),
        allowed_domains=frozenset(d.lstrip("@") for d in _env_list("ALLOWED_DOMAINS")),
        operator_emails=_env_list("OPERATOR_EMAILS"),
    )


def can_download_models(role: Role) -> bool:
    """Whether ``role`` may trigger an on-demand model download (``MODEL_DOWNLOAD_ON_DEMAND``).

    ``members`` (default): members and operators; ``operators``: operators only;
    ``all``: everyone; ``none``: nobody.
    """
    policy = os.environ.get("MODEL_DOWNLOAD_ON_DEMAND", "").strip().lower() or "members"
    allowed = {
        "all": ROLES,
        "members": ("member", "operator"),
        "operators": ("operator",),
        "none": (),
    }.get(policy, ("member", "operator"))
    return role in allowed


def log_startup_warnings() -> None:
    if not os.environ.get("SESSION_SECRET"):
        logger.warning("SESSION_SECRET is not set: using a random secret, so sessions reset when the server restarts")
    raw_role = os.environ.get("DEV_ROLE", "").strip()
    if raw_role and dev_role() is None:
        logger.warning(f"DEV_ROLE={raw_role!r} is not one of {', '.join(ROLES)}; ignored")
    elif raw_role:
        logger.warning(f"DEV_ROLE={dev_role()}: every request is treated as {dev_role()}")


# --- Sessions ----------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """The caller: ``id`` is the owner of its jobs and uploads; ``role`` already reflects ``DEV_ROLE``."""

    id: str
    role: Role
    created_at: float
    expires_at: float
    ip: str = ""

    @property
    def guest(self) -> bool:
        return self.role == "guest"

    @property
    def operator(self) -> bool:
        return self.role == "operator"

    def can_see(self, owner: str | None) -> bool:
        """Operators see everything; everyone else only what it owns (never an ownerless legacy row)."""
        return self.operator or (owner is not None and owner == self.id)


def new_session() -> Session:
    now = clock()
    return Session(
        id=secrets.token_hex(16),
        role="guest",
        created_at=now,
        expires_at=now + guest_session_hours() * 3600,
    )


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def encode_cookie(session: Session) -> str:
    """``<base64url JSON>.<HMAC-SHA256 hex>`` of id, role, created_at and expires_at."""
    data = {"sid": session.id, "role": session.role, "iat": int(session.created_at), "exp": int(session.expires_at)}
    payload = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode().rstrip("=")
    return f"{payload}.{_sign(payload)}"


def decode_cookie(value: str) -> Session | None:
    """The session in ``value``, or ``None`` if it is malformed, tampered with or expired."""
    payload, _, signature = value.partition(".")
    if not payload or not hmac.compare_digest(signature, _sign(payload)):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        session = Session(id=str(data["sid"]), role=data["role"], created_at=data["iat"], expires_at=data["exp"])
    except (ValueError, KeyError, TypeError):
        return None
    if session.role not in ROLES or session.expires_at <= clock():
        return None
    return session


def _set_cookie_header(session: Session) -> str:
    max_age = max(round(session.expires_at - clock()), 0)
    header = f"{COOKIE_NAME}={encode_cookie(session)}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax"
    return header + "; Secure" if cookie_secure() else header


Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
ASGIApp = Callable[[Scope, Callable[[], Awaitable[Message]], Callable[[Message], Awaitable[None]]], Awaitable[None]]


class SessionMiddleware:
    """Puts the caller's :class:`Session` on ``request.state.session``; sets the cookie when it issues one."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Callable[[], Awaitable[Message]], send: Callable[[Message], Awaitable[None]]
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cookie = Request(scope).cookies.get(COOKIE_NAME)
        session = decode_cookie(cookie) if cookie else None
        issued = session is None
        if session is None:
            session = new_session()
        client = scope.get("client")
        session = replace(session, role=dev_role() or session.role, ip=client[0] if client else "")
        scope.setdefault("state", {})["session"] = session
        if not issued:
            await self.app(scope, receive, send)
            return
        set_cookie = _set_cookie_header(session)

        async def send_with_cookie(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("set-cookie", set_cookie)
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def current_session(request: Request) -> Session:
    """FastAPI dependency: the caller's session (set by :class:`SessionMiddleware`)."""
    session: Session = request.state.session
    return session


# Route parameter type: ``session: CurrentSession``.
CurrentSession = Annotated[Session, Depends(current_session)]


# --- Guest quota -------------------------------------------------------------


class GuestQuotaError(Exception):
    """A guest asked for an analysis beyond ``GUEST_ANALYSES_PER_HOUR``; the app answers 429 ``GUEST_QUOTA``."""

    def __init__(self, limit: int, used: int, resets_at: float) -> None:
        self.limit = limit
        self.used = used
        self.resets_at = resets_at
        self.retry_after_s = max(int(resets_at - clock()) + 1, 1)
        at = datetime.fromtimestamp(resets_at, timezone.utc).strftime("%H:%M")
        super().__init__(
            f"Guests can run {limit} analyses per hour on this server. The next one is available at {at} UTC."
        )

    @property
    def details(self) -> dict[str, Any]:
        return {"limit": self.limit, "used": self.used, "resets_at": iso(self.resets_at)}


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class AnalysisCounter:
    """Analysis start times in the last :data:`WINDOW_S` seconds, per ``session:<id>`` and ``ip:<addr>``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starts: dict[str, deque[float]] = {}

    def _recent(self, key: str, now: float) -> deque[float]:
        starts = self._starts.get(key, deque())
        while starts and starts[0] <= now - WINDOW_S:
            starts.popleft()
        return starts

    def _keys(self, session: Session) -> list[str]:
        # Members and operators count only per session (shown, never limited); only guests share the IP bucket.
        return [f"session:{session.id}", f"ip:{session.ip}"] if session.guest else [f"session:{session.id}"]

    def usage(self, session: Session) -> tuple[int, float | None]:
        """``(used, resets_at)``: the larger of the session and IP counts, and when it next drops by one."""
        now = clock()
        with self._lock:
            windows = [self._recent(key, now) for key in self._keys(session)]
        used = max(len(w) for w in windows)
        if used == 0:
            return 0, None
        return used, max(w[0] for w in windows if len(w) == used) + WINDOW_S

    def enforce(self, session: Session) -> float:
        """Record one analysis for ``session`` and return its time, or raise :class:`GuestQuotaError`
        if a guest has none left."""
        now = clock()
        limit = guest_analyses_per_hour()
        with self._lock:
            windows = {key: self._recent(key, now) for key in self._keys(session)}
            if session.guest:
                full = [w for w in windows.values() if len(w) >= limit]
                if full:
                    used = max(len(w) for w in full)
                    resets_at = max(w[0] for w in full) + WINDOW_S if limit > 0 else now + WINDOW_S
                    raise GuestQuotaError(limit, used, resets_at)
            for key, starts in windows.items():
                starts.append(now)
                self._starts[key] = starts
        return now

    def release(self, session: Session, at: float) -> None:
        """Forget the analysis :meth:`enforce` recorded for ``session`` at ``at``."""
        with self._lock:
            for key in self._keys(session):
                starts = self._starts.get(key)
                if starts is not None and at in starts:
                    starts.remove(at)

    def prune(self) -> None:
        """Drop keys with no analysis in the window (run by the periodic cleanup)."""
        now = clock()
        with self._lock:
            for key in list(self._starts):
                if not self._recent(key, now):
                    del self._starts[key]

    def clear(self) -> None:
        with self._lock:
            self._starts.clear()


analysis_counter = AnalysisCounter()


def enforce_analysis_quota(session: Session) -> float:
    """Count one analysis for ``session`` and return when; raises :class:`GuestQuotaError` for a guest over the limit.

    Call it once per analysis that will run: ``POST /v1/jobs`` does; a proposal counts only
    when it is confirmed, so a submission whose planning ends in a proposal is released
    (:func:`release_analysis`) and the confirm endpoint calls this again.
    """
    return analysis_counter.enforce(session)


def release_analysis(session: Session, at: float) -> None:
    """Uncount the analysis :func:`enforce_analysis_quota` counted at ``at`` (it became a proposal)."""
    analysis_counter.release(session, at)


def session_info(session: Session) -> dict[str, Any]:
    """``GET /v1/session``: what the UI badge needs."""
    used, resets_at = analysis_counter.usage(session)
    limit = guest_analyses_per_hour() if session.guest else None
    return {
        "role": session.role,
        "guest": session.guest,
        "analyses": {
            "limit": limit,
            "used": used,
            "remaining": max(limit - used, 0) if limit is not None else None,
            "resets_at": iso(resets_at) if limit is not None and resets_at is not None else None,
        },
        "kept_hours": guest_session_hours() if session.guest else None,
        "expires_at": iso(session.expires_at),
        "can_download_models": can_download_models(session.role),
    }
