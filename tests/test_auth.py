"""GitHub sign-in, roles from the allowlists, stable owners and adoption (v0.6.0 run 3, Phase 5).

GitHub is a tiny in-process HTTP server on 127.0.0.1 that ``GITHUB_OAUTH_BASE`` and
``GITHUB_API_BASE`` point at; nothing leaves the machine. It implements what a fake must:

- ``POST /login/oauth/access_token`` (JSON ``{client_id, client_secret, code, redirect_uri}``,
  ``Accept: application/json``) → ``{"access_token", "token_type", "scope"}``, or
  ``{"error": "bad_verification_code"}`` with HTTP 200 for an unknown code, as GitHub does;
- ``GET /user`` (``Authorization: Bearer <token>``) → ``{"id", "login", "name", "avatar_url"}``;
- ``GET /user/emails`` → ``[{"email", "primary", "verified", "visibility"}]``.

The browser-facing ``GET /login/oauth/authorize`` (redirect to ``redirect_uri?code&state``) is
not needed here: the tests read ``state`` from the login redirect and call the callback.
"""

from __future__ import annotations

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from server import sessions
from server.app import app
import server.jobs as jobs_mod
from server.jobs import Job, JobQueue, JobStatus, MemoryJobStore, SQLiteJobStore
import server.paths as paths_mod
import server.routes.auth as auth_mod
import server.routes.files as files_mod
from server.uploads import cleanup_guest_uploads, owner_file

BASE = "http://testserver"  # OAUTH_REDIRECT_BASE: the TestClient's origin
T0 = 1_800_000_000.0


class FakeGitHub:
    """Accounts by OAuth code; records the token exchanges it answered."""

    def __init__(self) -> None:
        self.accounts: dict[str, dict[str, Any]] = {}
        self.exchanges: list[dict[str, Any]] = []
        self.url = ""

    def account(
        self,
        code: str,
        *,
        id: int,
        email: str | None,
        verified: bool = True,
        extra: list[dict[str, Any]] | None = None,
    ) -> None:
        login = f"user{id}"
        emails = [{"email": email, "primary": True, "verified": verified, "visibility": "private"}] if email else []
        self.accounts[code] = {
            "user": {"id": id, "login": login, "name": f"User {id}", "avatar_url": f"https://avatars.test/{id}"},
            "emails": emails + (extra or []),
        }


def _handler(fake: FakeGitHub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _account(self) -> dict[str, Any] | None:
            token = self.headers.get("Authorization", "").removeprefix("Bearer ")
            return fake.accounts.get(token.removeprefix("tok-")) if token.startswith("tok-") else None

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.exchanges.append(body)
            if self.path != "/login/oauth/access_token":
                return self._send(404, {"message": "Not Found"})
            if body.get("code") not in fake.accounts:
                return self._send(200, {"error": "bad_verification_code"})
            token = {"access_token": f"tok-{body['code']}", "token_type": "bearer", "scope": "read:user,user:email"}
            return self._send(200, token)

        def do_GET(self) -> None:  # noqa: N802
            account = self._account()
            if account is None:
                return self._send(401, {"message": "Bad credentials"})
            if self.path == "/user":
                return self._send(200, account["user"])
            if self.path == "/user/emails":
                return self._send(200, account["emails"])
            return self._send(404, {"message": "Not Found"})

    return Handler


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGitHub]:
    """Sign-in configured against the fake; requests run as their real role (no ``DEV_ROLE``)."""
    fake = FakeGitHub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setenv("GITHUB_CLIENT_ID", "cid")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("OAUTH_REDIRECT_BASE", BASE + "/")
    monkeypatch.setenv("GITHUB_OAUTH_BASE", fake.url)
    monkeypatch.setenv("GITHUB_API_BASE", fake.url)
    monkeypatch.delenv("DEV_ROLE", raising=False)
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[JobQueue]:
    """A queue without workers: submitted jobs stay pending."""
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    yield q
    q._thread_pool.shutdown(wait=False)


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    for mod in (files_mod, paths_mod, auth_mod):
        monkeypatch.setattr(mod, "UPLOAD_DIR", upload_dir)
    return upload_dir


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False, follow_redirects=False)


def _sid(client: TestClient) -> str:
    return sessions.decode_cookie(client.cookies[sessions.COOKIE_NAME]).id  # type: ignore[union-attr]


def _sign_in(client: TestClient, code: str, *, state: str | None = None) -> httpx.Response:
    login = client.get("/v1/auth/github/login")
    assert login.status_code == 302, login.text
    real_state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    return client.get("/v1/auth/github/callback", params={"code": code, "state": state or real_state})


def _submit(client: TestClient) -> int:
    return client.post("/v1/jobs", json={"prompt": "Compute NDVI"}).status_code


def _geotiff() -> bytes:
    with (
        MemoryFile() as mem,
        mem.open(
            driver="GTiff",
            width=32,
            height=32,
            count=3,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(32.0, 47.0, 0.01, 0.01),
        ) as dst,
    ):
        dst.write(np.ones((3, 32, 32), dtype=np.uint16))
        dst.close()
        return bytes(mem.read())


# --- Not configured ------------------------------------------------------------


def test_without_env_sign_in_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEV_ROLE", raising=False)
    client = _client()

    assert client.get("/v1/auth/status").json() == {"configured": False, "user": None, "role": "guest"}
    assert client.get("/v1/session").json()["sign_in"] == {"configured": False}
    for url in ("/v1/auth/github/login", "/v1/auth/github/callback?code=c&state=s"):
        r = client.get(url)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "SIGN_IN_NOT_CONFIGURED"


def test_partial_env_is_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_CLIENT_ID", "cid")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "csecret")
    assert _client().get("/v1/auth/status").json()["configured"] is False


# --- Login redirect ------------------------------------------------------------


def test_login_redirects_to_github_with_state_cookie(github: FakeGitHub) -> None:
    client = _client()

    r = client.get("/v1/auth/github/login")

    assert r.status_code == 302
    url = urlparse(r.headers["location"])
    assert f"{url.scheme}://{url.netloc}{url.path}" == f"{github.url}/login/oauth/authorize"
    query = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert query["client_id"] == "cid"
    assert query["redirect_uri"] == f"{BASE}/v1/auth/github/callback"
    assert query["scope"] == "read:user user:email"
    assert len(query["state"]) >= 32
    state_cookie = next(c for c in r.headers.get_list("set-cookie") if c.startswith("dt4lc_oauth_state="))
    assert "HttpOnly" in state_cookie and "Path=/v1/auth/github" in state_cookie and "Max-Age=600" in state_cookie
    assert "samesite=lax" in state_cookie.lower()
    assert client.get("/v1/auth/status").json() == {"configured": True, "user": None, "role": "guest"}


# --- Roles from the allowlists -------------------------------------------------


def test_allowlisted_email_becomes_a_member_without_quota(
    github: FakeGitHub, queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="Ana@Example.org")
    client = _client()
    guest_sid = _sid_after_first_request(client)

    r = _sign_in(client, "c1")

    assert r.status_code == 302
    assert r.headers["location"] == f"{BASE}/app"
    assert github.exchanges[-1] == {
        "client_id": "cid",
        "client_secret": "csecret",
        "code": "c1",
        "redirect_uri": f"{BASE}/v1/auth/github/callback",
    }
    user = {
        "id": "github:42",
        "login": "user42",
        "name": "User 42",
        "email": "Ana@Example.org",
        "avatar_url": "https://avatars.test/42",
    }
    session = client.get("/v1/session").json()
    assert (session["role"], session["guest"], session["user"]) == ("member", False, user)
    assert session["sign_in"] == {"configured": True}
    assert session["analyses"]["limit"] is None and session["kept_hours"] is None
    assert client.get("/v1/auth/status").json() == {"configured": True, "user": user, "role": "member"}
    assert _sid(client) != guest_sid  # a new session id on sign-in
    assert [_submit(client) for _ in range(7)] == [202] * 7
    assert {j["owner"] for j in client.get("/v1/jobs").json()["jobs"]} == {"github:42"}


def _sid_after_first_request(client: TestClient) -> str:
    client.get("/v1/session")
    return _sid(client)


def test_operator_email_becomes_an_operator(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERATOR_EMAILS", "ops@example.org")
    github.account("c1", id=7, email="ops@example.org")
    client = _client()

    assert _sign_in(client, "c1").headers["location"] == f"{BASE}/app"

    assert client.get("/v1/auth/status").json()["role"] == "operator"
    assert client.get("/v1/session").json()["can_download_models"] is True


def test_domain_allowlist(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_DOMAINS", "@unige.ch")
    github.account("c1", id=1, email="Bo@UNIGE.ch")
    github.account("c2", id=2, email="eve@notunige.ch")

    member = _client()
    _sign_in(member, "c1")
    assert member.get("/v1/auth/status").json()["role"] == "member"

    other = _client()
    assert "reason=not_allowlisted" in _sign_in(other, "c2").headers["location"]


def test_not_allowlisted_email_is_denied_and_stays_a_guest(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=9, email="eve@example.com")
    client = _client()
    guest_sid = _sid_after_first_request(client)

    r = _sign_in(client, "c1")

    assert r.status_code == 302
    assert r.headers["location"] == f"{BASE}/app?signin=denied&reason=not_allowlisted&email=eve%40example.com"
    assert not any(c.startswith(f"{sessions.COOKIE_NAME}=") for c in r.headers.get_list("set-cookie"))
    assert _sid(client) == guest_sid
    assert client.get("/v1/auth/status").json() == {"configured": True, "user": None, "role": "guest"}


# --- Refusals ------------------------------------------------------------------


def test_state_mismatch_is_refused(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="ana@example.org")
    client = _client()

    r = _sign_in(client, "c1", state="forged")

    assert r.headers["location"] == f"{BASE}/app?signin=error"
    assert github.exchanges == []  # refused before the code is exchanged
    assert client.get("/v1/auth/status").json()["user"] is None

    # No state cookie at all (the callback opened in another browser)
    fresh = _client()
    r = fresh.get("/v1/auth/github/callback", params={"code": "c1", "state": "anything"})
    assert r.headers["location"] == f"{BASE}/app?signin=error"


def test_expired_state_is_refused(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="ana@example.org")
    now = [T0]
    monkeypatch.setattr(sessions, "clock", lambda: now[0])
    client = _client()
    login = client.get("/v1/auth/github/login")
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    now[0] += 601

    r = client.get("/v1/auth/github/callback", params={"code": "c1", "state": state})

    assert r.headers["location"] == f"{BASE}/app?signin=error"


def test_unverified_primary_email_is_refused(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org,ana@other.org")
    github.account(
        "c1",
        id=42,
        email="ana@example.org",
        verified=False,
        extra=[{"email": "ana@other.org", "primary": False, "verified": True, "visibility": None}],
    )
    client = _client()

    r = _sign_in(client, "c1")

    assert r.headers["location"] == f"{BASE}/app?signin=denied&reason=email_unverified"
    assert client.get("/v1/auth/status").json()["user"] is None


@pytest.mark.parametrize("case", ["bad_code", "github_error", "provider_down"])
def test_oauth_errors_redirect_with_signin_error(
    github: FakeGitHub, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    client = _client()
    login = client.get("/v1/auth/github/login")
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    params = {"code": "unknown", "state": state}
    if case == "github_error":
        params = {"error": "access_denied", "state": state}
    elif case == "provider_down":
        monkeypatch.setenv("GITHUB_OAUTH_BASE", "http://127.0.0.1:9")  # nothing listens on the discard port

    r = client.get("/v1/auth/github/callback", params=params)

    assert r.status_code == 302
    assert r.headers["location"] == f"{BASE}/app?signin=error"
    assert client.get("/v1/auth/status").json()["user"] is None


# --- Logout, DEV_ROLE, allowlist changes ---------------------------------------


def test_logout_gives_a_fresh_guest(github: FakeGitHub, queue: JobQueue, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="ana@example.org")
    client = _client()
    _sign_in(client, "c1")
    member_sid = _sid(client)
    assert _submit(client) == 202

    r = client.post("/v1/auth/logout")

    assert r.status_code == 200
    body = r.json()
    assert (body["role"], body["guest"], body["user"]) == ("guest", True, None)
    assert body["analyses"]["limit"] == 5
    assert _sid(client) != member_sid
    assert client.get("/v1/auth/status").json() == {"configured": True, "user": None, "role": "guest"}
    assert client.get("/v1/jobs").json()["jobs"] == []  # the member's jobs stay with the member

    _sign_in(client, "c1")  # and come back on the next sign-in
    assert len(client.get("/v1/jobs").json()["jobs"]) == 1


def test_dev_role_still_overrides(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERATOR_EMAILS", "ops@example.org")
    github.account("c1", id=7, email="ops@example.org")
    client = _client()
    _sign_in(client, "c1")
    assert client.get("/v1/session").json()["role"] == "operator"

    monkeypatch.setenv("DEV_ROLE", "guest")
    session = client.get("/v1/session").json()
    assert (session["role"], session["user"]["id"]) == ("guest", "github:7")
    assert client.get("/v1/auth/status").json()["role"] == "guest"

    monkeypatch.setenv("DEV_ROLE", "member")
    assert _client().get("/v1/auth/status").json() == {"configured": True, "user": None, "role": "member"}


def test_allowlist_changes_apply_to_signed_in_users(github: FakeGitHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="ana@example.org")
    client = _client()
    _sign_in(client, "c1")

    monkeypatch.setenv("OPERATOR_EMAILS", "ana@example.org")
    assert client.get("/v1/auth/status").json()["role"] == "operator"
    assert sessions.decode_cookie(client.cookies[sessions.COOKIE_NAME]).role == "operator"  # type: ignore[union-attr]

    monkeypatch.delenv("OPERATOR_EMAILS")
    monkeypatch.delenv("ALLOWED_EMAILS")
    assert client.get("/v1/auth/status").json() == {"configured": True, "user": None, "role": "guest"}


# --- Stable owner and adoption -------------------------------------------------


def test_sign_in_adopts_the_guest_sessions_jobs_and_uploads(
    github: FakeGitHub, queue: JobQueue, uploads: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")
    github.account("c1", id=42, email="ana@example.org")
    github.account("c2", id=42, email="ana@example.org")  # the same account, another device
    laptop, stranger = _client(), _client()
    assert _submit(laptop) == 202
    up = laptop.post("/v1/upload", files={"file": ("scene.tif", _geotiff(), "image/tiff")}).json()
    sample = laptop.post("/v1/sample").json()["attachments"]
    assert _submit(stranger) == 202
    guest_sid = _sid(laptop)

    _sign_in(laptop, "c1")

    jobs = laptop.get("/v1/jobs").json()["jobs"]
    assert [j["owner"] for j in jobs] == ["github:42"]
    assert queue._store.get(jobs[0]["id"]).owner_role == "member"  # type: ignore[union-attr]
    files = {f["path"]: f for f in laptop.get("/v1/files").json()["files"] if f["source"] == "upload"}
    assert set(files) == {up["path"], *(a["path"] for a in sample)}
    assert {f["owner"] for f in files.values()} == {"github:42"}
    assert files[up["path"]]["kind"] == "upload"
    assert {files[a["path"]]["kind"] for a in sample} == {"sample"}  # still one sample row
    record = json.loads(owner_file(Path(up["path"])).read_text())
    assert (record["owner"], record["role"], record["adopted_from"]) == ("github:42", "member", guest_sid)
    assert laptop.get("/v1/download", params={"path": str(Path(up["path"]).with_suffix(".png"))}).status_code == 200

    # The stranger's guest job was not adopted
    assert [j["owner"] for j in stranger.get("/v1/jobs").json()["jobs"]] == [_sid(stranger)]

    # Another device signed in as the same user sees the same items
    desktop = _client()
    _sign_in(desktop, "c2")
    assert [j["id"] for j in desktop.get("/v1/jobs").json()["jobs"]] == [jobs[0]["id"]]
    assert {f["path"] for f in desktop.get("/v1/files").json()["files"] if f["source"] == "upload"} == set(files)

    # Adopted uploads are a member's now: the guest cleanup keeps them
    assert cleanup_guest_uploads(uploads, 0) == 0


@pytest.mark.asyncio
async def test_adopt_rewrites_owner_in_the_jobs_db(tmp_path: Path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save(Job(id="mine", status=JobStatus.COMPLETED, prompt="p", owner="sid-a", owner_role="guest"))
    store.save(Job(id="other", status=JobStatus.COMPLETED, prompt="p", owner="sid-b", owner_role="guest"))
    queue = JobQueue(max_workers=1, store=store)
    try:
        assert await queue.adopt("sid-a", "github:42", "member") == 1
    finally:
        queue._thread_pool.shutdown(wait=False)

    reopened = SQLiteJobStore(tmp_path / "jobs.db")
    mine, other = reopened.get("mine"), reopened.get("other")
    assert (mine.owner, mine.owner_role) == ("github:42", "member")  # type: ignore[union-attr]
    assert (other.owner, other.owner_role) == ("sid-b", "guest")  # type: ignore[union-attr]


def test_signing_in_again_adopts_nothing_from_another_user(
    github: FakeGitHub, queue: JobQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org,bo@example.org")
    github.account("c1", id=1, email="ana@example.org")
    github.account("c2", id=2, email="bo@example.org")
    client = _client()
    _sign_in(client, "c1")
    assert _submit(client) == 202

    _sign_in(client, "c2")  # switch account without signing out

    assert client.get("/v1/jobs").json()["jobs"] == []
    assert {j.owner for j in queue._jobs_cache.values()} == {"github:1"}


def test_a_cookie_without_a_user_is_always_a_guest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cookies written under DEV_ROLE before sign-in existed carry ``role: operator``; now they are guests."""
    monkeypatch.delenv("DEV_ROLE", raising=False)
    client = _client()
    stale = sessions.replace(sessions.new_session(), role="operator")
    client.cookies.set(sessions.COOKIE_NAME, sessions.encode_cookie(stale))

    r = client.get("/v1/session")

    assert r.json()["role"] == "guest"
    rewritten = sessions.decode_cookie(r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1])
    assert (rewritten.id, rewritten.role) == (stale.id, "guest")  # type: ignore[union-attr]
