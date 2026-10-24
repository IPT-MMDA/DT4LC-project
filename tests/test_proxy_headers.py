"""Behind nginx the per-IP guest quota counts the forwarded client IP.

uvicorn runs with ``--proxy-headers --forwarded-allow-ips "$FORWARDED_ALLOW_IPS"``; its
``ProxyHeadersMiddleware`` rewrites the ASGI client from ``X-Forwarded-For`` when the socket
peer is trusted, and the session middleware takes ``Session.ip`` from that client.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
import yaml

from server.app import app
import server.jobs as jobs_mod
from server.jobs import JobQueue, MemoryJobStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guest_queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[JobQueue]:
    monkeypatch.setenv("DEV_ROLE", "guest")
    monkeypatch.setenv("GUEST_ANALYSES_PER_HOUR", "2")
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    yield q
    q._thread_pool.shutdown(wait=False)


def _browser(trusted_hosts: str, forwarded_for: str) -> TestClient:
    """A browser with its own cookie jar behind a proxy that sends ``X-Forwarded-For``; every
    TestClient connects from the same socket address, ``testclient``."""
    proxied = ProxyHeadersMiddleware(app, trusted_hosts=trusted_hosts)  # type: ignore[arg-type]
    return TestClient(proxied, headers={"X-Forwarded-For": forwarded_for}, raise_server_exceptions=False)


def _submit(client: TestClient) -> int:
    return client.post("/v1/jobs", json={"prompt": "Compute NDVI"}).status_code


def test_quota_counts_the_forwarded_ip(guest_queue: JobQueue) -> None:
    alice = _browser("testclient", "203.0.113.7")
    bob = _browser("testclient", "198.51.100.4")

    # Two forwarded IPs from one socket: separate quotas
    assert [_submit(alice) for _ in range(2)] == [202, 202]
    assert [_submit(bob) for _ in range(2)] == [202, 202]
    assert _submit(alice) == 429

    # The same forwarded IP in another session shares the first one's quota
    carol = _browser("testclient", "203.0.113.7")
    r = carol.post("/v1/jobs", json={"prompt": "Compute NDVI"})
    assert r.status_code == 429
    assert r.json()["error"]["details"]["used"] == 2
    assert carol.get("/v1/session").json()["analyses"]["remaining"] == 0


def test_forwarded_for_from_an_untrusted_peer_is_ignored(guest_queue: JobQueue) -> None:
    alice = _browser("127.0.0.1", "203.0.113.7")
    spoofer = _browser("127.0.0.1", "198.51.100.4")

    assert [_submit(alice) for _ in range(2)] == [202, 202]
    assert _submit(spoofer) == 429  # both count as "testclient"


def test_dockerfile_and_compose_enable_proxy_headers() -> None:
    cmd = next(line for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("CMD "))
    script = json.loads(cmd.removeprefix("CMD "))[-1]
    assert "--proxy-headers" in script
    assert '--forwarded-allow-ips "$FORWARDED_ALLOW_IPS"' in script

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    env = compose["services"]["backend"]["environment"]
    assert any(e.startswith("FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-") for e in env)
