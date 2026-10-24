"""Sessions, roles, owner scoping and the guest quota (v0.6.0 "Sessions, roles, quotas")."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import mercantile
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from server import sessions
import server.app as app_mod
from server.app import app
import server.jobs as jobs_mod
from server.jobs import Job, JobQueue, JobStatus, MemoryJobStore, SQLiteJobStore
import server.paths as paths_mod
import server.routes.files as files_mod
from server.uploads import cleanup_guest_uploads, owner_file

T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z
HOUR = 3600.0
TILE = mercantile.tile(32.5, 46.5, 10)
TILE_URL = f"/v1/tiles/{TILE.z}/{TILE.x}/{TILE.y}"


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(sessions, "clock", clock)
    return clock


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
    monkeypatch.setattr(files_mod, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(paths_mod, "UPLOAD_DIR", upload_dir)
    return upload_dir


def _client() -> TestClient:
    """A browser: its own cookie jar, same client IP ("testclient") as every other TestClient."""
    return TestClient(app, raise_server_exceptions=False)


def _role(monkeypatch: pytest.MonkeyPatch, role: str | None) -> None:
    if role is None:
        monkeypatch.delenv("DEV_ROLE", raising=False)
    else:
        monkeypatch.setenv("DEV_ROLE", role)


def _submit(client: TestClient, prompt: str = "Compute NDVI") -> tuple[int, dict]:
    r = client.post("/v1/jobs", json={"prompt": prompt})
    return r.status_code, r.json()


def _geotiff_bytes() -> bytes:
    with (
        MemoryFile() as mem,
        mem.open(
            driver="GTiff",
            width=100,
            height=100,
            count=3,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(32.0, 47.0, 0.01, 0.01),
        ) as dst,
    ):
        dst.write(np.arange(3 * 100 * 100, dtype=np.uint16).reshape(3, 100, 100))
        dst.close()
        return bytes(mem.read())


# --- Cookie --------------------------------------------------------------------


def test_cookie_is_issued_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock
) -> None:
    _role(monkeypatch, None)
    client = _client()

    first = client.get("/v1/session")
    set_cookie = first.headers["set-cookie"]
    assert set_cookie.startswith(f"{sessions.COOKIE_NAME}=")
    assert "HttpOnly" in set_cookie
    assert "SameSite=Lax" in set_cookie
    assert "Path=/" in set_cookie
    assert f"Max-Age={24 * 3600}" in set_cookie
    assert "Secure" not in set_cookie

    status, job = _submit(client)
    assert status == 202
    again = client.get(f"/v1/jobs/{job['id']}")
    assert "set-cookie" not in again.headers
    assert again.status_code == 200
    session_id = sessions.decode_cookie(client.cookies[sessions.COOKIE_NAME]).id  # type: ignore[union-attr]
    assert job["owner"] == session_id


def test_secure_flag_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
    assert "; Secure" in _client().get("/v1/session").headers["set-cookie"]


def test_session_docs_describe_sign_in() -> None:
    """BE59: /docs shows the route docstring, which predated sign-in."""
    description = _client().get("/openapi.json").json()["paths"]["/v1/session"]["get"]["description"]
    assert "until sign-in exists" not in description
    assert "signing in" in description.lower() and "DEV_ROLE" in description


def test_tampered_cookie_gets_a_new_session(monkeypatch: pytest.MonkeyPatch, queue: JobQueue) -> None:
    _role(monkeypatch, None)
    client = _client()
    _, job = _submit(client)
    payload, _, signature = client.cookies[sessions.COOKIE_NAME].partition(".")
    data = json.loads(sessions.base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    data["role"] = "operator"
    forged = sessions.base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    client.cookies.clear()
    client.cookies.set(sessions.COOKIE_NAME, f"{forged}.{signature}")

    r = client.get(f"/v1/jobs/{job['id']}")

    assert r.status_code == 404  # a fresh guest session, not an operator
    issued = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
    assert sessions.decode_cookie(issued).role == "guest"  # type: ignore[union-attr]


def test_cookie_signed_with_another_secret_or_expired_is_rejected(
    monkeypatch: pytest.MonkeyPatch, fake_clock: FakeClock
) -> None:
    monkeypatch.setenv("SESSION_SECRET", "one")
    value = sessions.encode_cookie(sessions.new_session())
    assert sessions.decode_cookie(value) is not None
    monkeypatch.setenv("SESSION_SECRET", "two")
    assert sessions.decode_cookie(value) is None
    monkeypatch.setenv("SESSION_SECRET", "one")
    fake_clock.now += 24 * HOUR
    assert sessions.decode_cookie(value) is None
    assert sessions.decode_cookie("garbage") is None


# --- Owner scoping -------------------------------------------------------------


def test_another_session_cannot_see_stream_or_cancel_a_job(monkeypatch: pytest.MonkeyPatch, queue: JobQueue) -> None:
    _role(monkeypatch, "member")
    alice, bob = _client(), _client()
    _, job = _submit(alice)
    job_id = job["id"]
    assert job["owner"]

    assert bob.get(f"/v1/jobs/{job_id}").status_code == 404
    assert bob.get(f"/v1/jobs/{job_id}/events").status_code == 404
    assert bob.post(f"/v1/jobs/{job_id}/cancel").status_code == 404
    assert bob.get("/v1/jobs").json()["jobs"] == []
    assert bob.get("/v1/jobs").json()["total"] == 0

    listed = alice.get("/v1/jobs").json()
    assert [j["id"] for j in listed["jobs"]] == [job_id]
    assert listed["total"] == 1
    assert listed["jobs"][0]["owner"] == job["owner"]
    assert alice.post(f"/v1/jobs/{job_id}/cancel").status_code == 200
    with alice.stream("GET", f"/v1/jobs/{job_id}/events") as r:
        assert r.status_code == 200
        assert "event: cancelled" in r.read().decode()

    _role(monkeypatch, "operator")
    operator = _client()
    assert operator.get(f"/v1/jobs/{job_id}").status_code == 200
    assert [j["id"] for j in operator.get("/v1/jobs").json()["jobs"]] == [job_id]


def test_ownerless_legacy_job_is_visible_to_operators_only(monkeypatch: pytest.MonkeyPatch, queue: JobQueue) -> None:
    queue._store.save(Job(id="legacy", status=JobStatus.COMPLETED, prompt="old", completed_at=datetime.now()))
    for role in ("guest", "member"):
        _role(monkeypatch, role)
        client = _client()
        assert client.get("/v1/jobs/legacy").status_code == 404
        assert client.get("/v1/jobs").json()["jobs"] == []
    _role(monkeypatch, "operator")
    assert _client().get("/v1/jobs/legacy").status_code == 200


def test_job_artifacts_are_served_to_their_owner_only(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, _outputs_in_tmp: Path
) -> None:
    _role(monkeypatch, "guest")
    alice, bob = _client(), _client()
    _, job = _submit(alice)
    job_dir = _outputs_in_tmp / job["id"]
    job_dir.mkdir()
    png = job_dir / "1_NDVIMap_ndvi_map.png"
    png.write_bytes(b"\x89PNG")
    tif = job_dir / "1_NDVIMap_ndvi.tif"
    tif.write_bytes(_geotiff_bytes())

    assert alice.get("/v1/download", params={"path": str(png)}).status_code == 200
    assert alice.get(TILE_URL, params={"path": str(tif)}).status_code == 200
    assert bob.get("/v1/download", params={"path": str(png)}).status_code == 404
    assert bob.get(TILE_URL, params={"path": str(tif)}).status_code == 404

    # Pipelines run outside the queue (/v1/execute, /v1/chat) write OUTPUTS_PATH/run-*: no owner, served as before
    run_png = _outputs_in_tmp / "run-abc" / "x.png"
    run_png.parent.mkdir()
    run_png.write_bytes(b"\x89PNG")
    assert bob.get("/v1/download", params={"path": str(run_png)}).status_code == 200

    _role(monkeypatch, "operator")
    assert _client().get("/v1/download", params={"path": str(png)}).status_code == 200


def test_uploads_carry_owner_and_are_scoped(monkeypatch: pytest.MonkeyPatch, uploads: Path) -> None:
    _role(monkeypatch, "guest")
    alice, bob = _client(), _client()

    r = alice.post("/v1/upload", files={"file": ("scene.tif", _geotiff_bytes(), "image/tiff")})
    assert r.status_code == 200, r.text
    up = r.json()
    alice_id = sessions.decode_cookie(alice.cookies[sessions.COOKIE_NAME]).id  # type: ignore[union-attr]
    assert up["owner"] == alice_id
    sidecar = json.loads(owner_file(Path(up["path"])).read_text())
    assert sidecar["owner"] == alice_id
    assert sidecar["role"] == "guest"

    assert _uploads_listed(alice) == [(up["path"], alice_id)]
    assert _uploads_listed(bob) == []

    preview = {"path": str(Path(up["path"]).with_suffix(".png"))}
    assert alice.get("/v1/download", params=preview).status_code == 200
    assert alice.get(TILE_URL, params={"path": up["path"]}).status_code == 200
    assert bob.get("/v1/download", params=preview).status_code == 404
    assert bob.get("/v1/download", params={"path": up["path"]}).status_code == 404
    assert bob.get(TILE_URL, params={"path": up["path"]}).status_code == 404
    # The sidecar itself is never served to a non-operator
    sidecar_path = {"path": str(owner_file(Path(up["path"])))}
    assert alice.get("/v1/download", params=sidecar_path).status_code == 404

    _role(monkeypatch, "operator")
    assert _uploads_listed(_client()) == [(up["path"], alice_id)]


def _uploads_listed(client: TestClient) -> list[tuple[str, str | None]]:
    files = client.get("/v1/files").json()["files"]
    return [(f["path"], f["owner"]) for f in files if f["source"] == "upload"]


# --- Guest quota ---------------------------------------------------------------


def test_sixth_guest_analysis_in_an_hour_is_refused(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock
) -> None:
    _role(monkeypatch, "guest")
    client = _client()
    for _ in range(5):
        assert _submit(client)[0] == 202
        fake_clock.now += 60  # analyses at T0, T0+1 min, ..., T0+4 min

    status, body = _submit(client)

    assert status == 429
    assert body == {
        "ok": False,
        "error": {
            "code": "GUEST_QUOTA",
            "message": "Guests can run 5 analyses per hour on this server. The next one is available at 09:00 UTC.",
            "details": {"limit": 5, "used": 5, "resets_at": "2027-01-15T09:00:00+00:00"},
        },
    }
    assert queue._store.count() == 5  # refused before a job exists
    r = client.post("/v1/jobs", json={"prompt": "Compute NDVI"})
    assert r.headers["retry-after"] == str(int(HOUR - 5 * 60) + 1)

    # The window rolls: one hour after the first analysis, one slot reopens
    fake_clock.now = T0 + HOUR
    assert _submit(client)[0] == 202
    assert _submit(client)[0] == 429
    fake_clock.now = T0 + HOUR + 60
    assert _submit(client)[0] == 202


def test_per_ip_quota_is_shared_across_sessions(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock
) -> None:
    _role(monkeypatch, "guest")
    first, second = _client(), _client()
    for _ in range(3):
        assert _submit(first)[0] == 202
    for _ in range(2):
        assert _submit(second)[0] == 202

    status, body = _submit(second)  # its own session has used 2, but the IP has used 5

    assert status == 429
    assert body["error"]["code"] == "GUEST_QUOTA"
    assert body["error"]["details"]["used"] == 5
    assert second.get("/v1/session").json()["analyses"]["remaining"] == 0


@pytest.mark.parametrize("role", ["member", "operator"])
def test_members_and_operators_are_not_limited(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock, role: str
) -> None:
    _role(monkeypatch, role)
    client = _client()
    for _ in range(8):
        assert _submit(client)[0] == 202


def test_guest_limit_from_env(monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock) -> None:
    _role(monkeypatch, "guest")
    monkeypatch.setenv("GUEST_ANALYSES_PER_HOUR", "2")
    client = _client()
    assert [_submit(client)[0] for _ in range(3)] == [202, 202, 429]


def test_enforce_helper_for_confirmed_proposals(fake_clock: FakeClock) -> None:
    """The proposal story calls ``enforce_analysis_quota`` when a proposal is confirmed."""
    guest = sessions.Session(id="s1", role="guest", created_at=T0, expires_at=T0 + 24 * HOUR, ip="10.0.0.1")
    for _ in range(5):
        sessions.enforce_analysis_quota(guest)
    with pytest.raises(sessions.GuestQuotaError) as exc:
        sessions.enforce_analysis_quota(guest)
    assert exc.value.details == {"limit": 5, "used": 5, "resets_at": "2027-01-15T09:00:00+00:00"}


# --- GET /v1/session -----------------------------------------------------------


def test_session_endpoint_for_guest(monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock) -> None:
    _role(monkeypatch, None)  # no DEV_ROLE: every request is a guest
    client = _client()
    assert client.get("/v1/session").json() == {
        "role": "guest",
        "guest": True,
        "user": None,
        "sign_in": {"configured": False},
        "analyses": {"limit": 5, "used": 0, "remaining": 5, "resets_at": None},
        "kept_hours": 24,
        "expires_at": "2027-01-16T08:00:00+00:00",
        "can_download_models": False,
    }
    _submit(client)
    fake_clock.now += 600
    assert client.get("/v1/session").json()["analyses"] == {
        "limit": 5,
        "used": 1,
        "remaining": 4,
        "resets_at": "2027-01-15T09:00:00+00:00",
    }


@pytest.mark.parametrize("role", ["member", "operator"])
def test_session_endpoint_for_member_and_operator(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, fake_clock: FakeClock, role: str
) -> None:
    _role(monkeypatch, role)
    client = _client()
    _submit(client)
    assert client.get("/v1/session").json() == {
        "role": role,
        "guest": False,
        "user": None,
        "sign_in": {"configured": False},
        "analyses": {"limit": None, "used": 1, "remaining": None, "resets_at": None},
        "kept_hours": None,
        "expires_at": "2027-01-16T08:00:00+00:00",
        "can_download_models": True,
    }


@pytest.mark.parametrize(
    ("policy", "allowed"),
    [
        (None, {"member", "operator"}),
        ("members", {"member", "operator"}),
        ("operators", {"operator"}),
        ("all", {"guest", "member", "operator"}),
        ("none", set()),
    ],
)
def test_can_download_models_policy(monkeypatch: pytest.MonkeyPatch, policy: str | None, allowed: set[str]) -> None:
    if policy is None:
        monkeypatch.delenv("MODEL_DOWNLOAD_ON_DEMAND", raising=False)
    else:
        monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", policy)
    assert {role for role in sessions.ROLES if sessions.can_download_models(role)} == allowed


def test_invalid_dev_role_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _role(monkeypatch, "admin")
    assert _client().get("/v1/session").json()["role"] == "guest"


def test_sign_in_config_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EMAILS", " Ana@Example.org, bo@x.io ,")
    monkeypatch.setenv("ALLOWED_DOMAINS", "@unige.ch,umd.edu")
    monkeypatch.setenv("OPERATOR_EMAILS", "ops@unige.ch")
    assert sessions.sign_in_config() == sessions.SignInConfig(
        allowed_emails=frozenset({"ana@example.org", "bo@x.io"}),
        allowed_domains=frozenset({"unige.ch", "umd.edu"}),
        operator_emails=frozenset({"ops@unige.ch"}),
    )


# --- Cleanup -------------------------------------------------------------------


def _finished(job_id: str, role: str | None, age_hours: float) -> Job:
    created = datetime.now() - timedelta(hours=age_hours)
    return Job(
        id=job_id,
        status=JobStatus.COMPLETED,
        prompt="p",
        created_at=created,
        completed_at=created + timedelta(seconds=5),
        owner=f"sid-{job_id}" if role else None,
        owner_role=role,
    )


@pytest.mark.asyncio
async def test_cleanup_removes_expired_guest_jobs_and_keeps_members(tmp_path: Path, _outputs_in_tmp: Path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    queue = JobQueue(max_workers=1, store=store, retention_hours=None)
    jobs = [
        _finished("guest-old", "guest", 25),
        _finished("guest-new", "guest", 2),
        _finished("member-old", "member", 25),
        _finished("operator-old", "operator", 25),
        _finished("legacy-old", None, 25),
    ]
    for job in jobs:
        store.save(job)
        (_outputs_in_tmp / job.id).mkdir()
        (_outputs_in_tmp / job.id / "a.tif").write_bytes(b"tif")
    try:
        assert await queue.cleanup_old_jobs(guest_hours=24) == 1
    finally:
        queue._thread_pool.shutdown(wait=False)

    assert sorted(store.keys()) == ["guest-new", "legacy-old", "member-old", "operator-old"]
    assert not (_outputs_in_tmp / "guest-old").exists()
    for kept in ("guest-new", "member-old", "operator-old", "legacy-old"):
        assert (_outputs_in_tmp / kept / "a.tif").is_file()


def _upload(upload_dir: Path, stem: str, role: str, age_hours: float) -> list[Path]:
    files = [upload_dir / f"{stem}.tif", upload_dir / f"{stem}.png"]
    for f in files:
        f.write_bytes(b"x")
    sidecar = owner_file(files[0])
    created = (datetime.now() - timedelta(hours=age_hours)).isoformat()
    sidecar.write_text(json.dumps({"owner": f"sid-{stem}", "role": role, "created_at": created}))
    return [*files, sidecar]


def test_cleanup_removes_expired_guest_uploads(tmp_path: Path) -> None:
    old_guest = _upload(tmp_path, "aa_old", "guest", 25)
    new_guest = _upload(tmp_path, "bb_new", "guest", 1)
    old_member = _upload(tmp_path, "cc_member", "member", 25)
    legacy = tmp_path / "dd_legacy.tif"
    legacy.write_bytes(b"x")

    assert cleanup_guest_uploads(tmp_path, 24) == 1

    assert not any(f.exists() for f in old_guest)
    assert all(f.exists() for f in [*new_guest, *old_member, legacy])


@pytest.mark.asyncio
async def test_periodic_cleanup_runs_in_the_lifespan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[JobQueue] = []

    async def fake_run(queue: JobQueue) -> None:
        calls.append(queue)

    monkeypatch.setattr(app_mod, "cleanup_interval_s", lambda: 0.01)
    monkeypatch.setattr(app_mod, "run_cleanup", fake_run)
    queue = JobQueue(max_workers=1, store=MemoryJobStore())
    task = asyncio.create_task(app_mod._cleanup_periodically(queue))
    await asyncio.sleep(0.1)
    task.cancel()
    queue._thread_pool.shutdown(wait=False)
    assert len(calls) >= 2


def test_cleanup_interval_is_hourly_unless_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GUEST_CLEANUP_MINUTES", raising=False)
    assert app_mod.cleanup_interval_s() == 3600.0
    monkeypatch.setenv("GUEST_CLEANUP_MINUTES", "15")
    assert app_mod.cleanup_interval_s() == 900.0
    monkeypatch.setenv("GUEST_CLEANUP_MINUTES", "0")  # never a busy loop
    assert app_mod.cleanup_interval_s() == 60.0
    monkeypatch.setenv("GUEST_CLEANUP_MINUTES", "soon")
    assert app_mod.cleanup_interval_s() == 3600.0


@pytest.mark.asyncio
async def test_run_cleanup_covers_jobs_and_uploads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.db")
    store.save(_finished("guest-old", "guest", 25))
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    old_upload = _upload(upload_dir, "aa_old", "guest", 25)
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", upload_dir)
    queue = JobQueue(max_workers=1, store=store, retention_hours=None)
    try:
        await app_mod.run_cleanup(queue)
    finally:
        queue._thread_pool.shutdown(wait=False)
    assert store.keys() == []
    assert not any(f.exists() for f in old_upload)


def test_server_queue_keeps_member_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The global queue has no blanket retention: the scheduled cleanup must not delete members' jobs."""
    monkeypatch.setenv("JOB_STORAGE", "memory")
    monkeypatch.setattr(jobs_mod, "_job_queue", None)
    queue = jobs_mod.get_job_queue()
    try:
        assert queue.retention_hours is None
    finally:
        queue._thread_pool.shutdown(wait=False)


# --- Migration -----------------------------------------------------------------


def test_migrates_a_jobs_db_without_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "jobs.db"
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, prompt TEXT NOT NULL,
            attachments TEXT NOT NULL DEFAULT '[]', context TEXT, plan TEXT, result TEXT,
            progress REAL NOT NULL DEFAULT 0.0, error TEXT, created_at TEXT NOT NULL,
            started_at TEXT, completed_at TEXT, events TEXT NOT NULL DEFAULT '[]'
        )
    """)
    con.execute(
        "INSERT INTO jobs (id, status, prompt, progress, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("pre060", "completed", "Compute NDVI", 1.0, "2026-10-20T10:00:00", "2026-10-20T10:00:05"),
    )
    con.commit()
    con.close()

    store = SQLiteJobStore(db)

    columns = {row["name"] for row in store._conn.execute("PRAGMA table_info(jobs)")}
    assert {"owner", "owner_role"} <= columns
    job = store.get("pre060")
    assert job is not None
    assert (job.owner, job.owner_role) == (None, None)
    assert store.get_owner("pre060") is None
    job.owner, job.owner_role = "sid", "guest"
    store.save(job)
    assert store.get_owner("pre060") == "sid"
    assert store.list_summaries()[0].owner_role == "guest"
