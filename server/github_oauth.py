"""GitHub OAuth: configuration, the authorize URL, and the code-for-identity exchange.

Sign-in is configured when ``GITHUB_CLIENT_ID``, ``GITHUB_CLIENT_SECRET`` and
``OAUTH_REDIRECT_BASE`` are all set. ``GITHUB_OAUTH_BASE`` (default ``https://github.com``)
and ``GITHUB_API_BASE`` (default ``https://api.github.com``) point tests and E2E at a fake
provider. The calls are blocking (``requests``); routes run them in a thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlencode

import requests

SCOPE = "read:user user:email"
CALLBACK_PATH = "/v1/auth/github/callback"
TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GitHubConfig:
    client_id: str
    client_secret: str
    redirect_base: str  # public origin of the app, e.g. https://demo.example.org (no trailing slash)
    oauth_base: str
    api_base: str

    @property
    def callback_url(self) -> str:
        return self.redirect_base + CALLBACK_PATH


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def github_config() -> GitHubConfig | None:
    """The GitHub app from the environment, or ``None`` when sign-in is not configured."""
    client_id, secret, base = _env("GITHUB_CLIENT_ID"), _env("GITHUB_CLIENT_SECRET"), _env("OAUTH_REDIRECT_BASE")
    if not (client_id and secret and base):
        return None
    return GitHubConfig(
        client_id=client_id,
        client_secret=secret,
        redirect_base=base.rstrip("/"),
        oauth_base=_env("GITHUB_OAUTH_BASE", "https://github.com").rstrip("/"),
        api_base=_env("GITHUB_API_BASE", "https://api.github.com").rstrip("/"),
    )


def sign_in_configured() -> bool:
    return github_config() is not None


def authorize_url(config: GitHubConfig, state: str) -> str:
    query = {"client_id": config.client_id, "redirect_uri": config.callback_url, "scope": SCOPE, "state": state}
    return f"{config.oauth_base}/login/oauth/authorize?{urlencode(query)}"


class OAuthError(Exception):
    """The provider refused the code, answered with an error, or could not be reached."""


@dataclass(frozen=True)
class GitHubIdentity:
    id: int
    login: str
    name: str | None
    avatar_url: str | None
    # The primary email if GitHub has verified it, else ``None``; ``primary_email`` is the primary either way.
    verified_email: str | None
    primary_email: str | None


def _json(response: requests.Response) -> Any:
    if response.status_code >= 400:
        raise OAuthError(f"{response.request.method} {response.url}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise OAuthError(f"{response.url}: not JSON") from exc


def fetch_identity(config: GitHubConfig, code: str) -> GitHubIdentity:
    """Exchange ``code`` for a token, then read ``/user`` and ``/user/emails``. Blocking.

    Raises:
        OAuthError: any failure of the exchange or the two reads.
    """
    try:
        token = _json(
            requests.post(
                f"{config.oauth_base}/login/oauth/access_token",
                json={
                    "client_id": config.client_id,
                    "client_secret": config.client_secret,
                    "code": code,
                    "redirect_uri": config.callback_url,
                },
                headers={"Accept": "application/json"},
                timeout=TIMEOUT_S,
            )
        )
        if not isinstance(token, dict) or not isinstance(token.get("access_token"), str):
            error = token.get("error") if isinstance(token, dict) else None
            raise OAuthError(f"token exchange failed: {error or 'no access_token'}")
        headers = {"Authorization": f"Bearer {token['access_token']}", "Accept": "application/vnd.github+json"}
        user = _json(requests.get(f"{config.api_base}/user", headers=headers, timeout=TIMEOUT_S))
        emails = _json(requests.get(f"{config.api_base}/user/emails", headers=headers, timeout=TIMEOUT_S))
    except requests.RequestException as exc:
        raise OAuthError(str(exc)) from exc
    if not isinstance(user, dict) or not isinstance(user.get("id"), int) or not isinstance(user.get("login"), str):
        raise OAuthError("/user has no id or login")
    if not isinstance(emails, list):
        raise OAuthError("/user/emails is not a list")
    primary = next((e for e in emails if isinstance(e, dict) and e.get("primary") is True), None)
    primary_email = primary.get("email") if primary and isinstance(primary.get("email"), str) else None
    verified = primary_email if primary_email and primary and primary.get("verified") is True else None
    return GitHubIdentity(
        id=user["id"],
        login=user["login"],
        name=user.get("name") if isinstance(user.get("name"), str) else None,
        avatar_url=user.get("avatar_url") if isinstance(user.get("avatar_url"), str) else None,
        verified_email=verified,
        primary_email=primary_email,
    )
