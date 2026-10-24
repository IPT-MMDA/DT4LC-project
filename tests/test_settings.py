"""Operator settings: ``GET/PUT /v1/settings`` and ``POST /v1/settings/assistant/test`` (v0.6.0 run 3).

Every test writes its overrides to its own ``DT4LC_SETTINGS_PATH`` and sets fake API keys,
so the real ``.env`` keys never reach a response and no provider is called over the network.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from dta.config import CACHE_PATH, ROOT_DIR
import dta.config.overrides as overrides
from dta.dti.coe.llm import LLMResponse, get_llm_router, reset_llm_router
from server import sessions
from server.app import app
import server.jobs as jobs_mod
from server.jobs import JobQueue, MemoryJobStore
import server.routes.health as health
import server.settings as settings_mod

FAKE_KEYS = {
    "GEMINI_API_KEY": "AIzaFAKE-gemini-value-0001",
    "GROQ_API_KEY": "gsk_FAKE_groq_value_0002",
    "ANTHROPIC_API_KEY": "sk-ant-FAKE-anthropic-0003",
    "MISTRAL_API_KEY": "FAKE-mistral-value-0004",
    "OPENAI_API_KEY": "sk-FAKE-openai-value-0005",
    "GPT_SECRET": "FAKE-gpt-secret-0006",
    "GITHUB_CLIENT_SECRET": "FAKE-github-secret-0007",
    "SESSION_SECRET": "FAKE-session-secret-0008",
}
SECTIONS = {"assistant", "models", "access", "data_sources", "server"}


@pytest.fixture
def settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Own overrides file, fake keys, gemini then groq (both key-only, so no network), fresh caches."""
    path = tmp_path / "state" / "settings.json"
    monkeypatch.setenv("DT4LC_SETTINGS_PATH", str(path))
    for name, value in FAKE_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")
    for name in ("GEMINI", "GROQ"):
        monkeypatch.setenv(f"LLM_ENABLE_{name}", "true")
    for name in (
        "GEMINI_MODELS",
        "GROQ_MODEL",
        "MODEL_DOWNLOAD_ON_DEMAND",
        "GUEST_ANALYSES_PER_HOUR",
        "GUEST_SESSION_HOURS",
        "GEE_PROJECT_ID",
        "GEE_SERVICE_ACCOUNT_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(overrides, "_cache", None)
    monkeypatch.setattr(health, "_assistant_cache", None)
    monkeypatch.setattr(settings_mod, "_storage_cache", {})
    monkeypatch.setattr(jobs_mod, "_job_queue", JobQueue(max_workers=1, store=MemoryJobStore()))
    reset_llm_router()
    yield path
    reset_llm_router()


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _user_cookie(email: str) -> str:
    now = time.time()
    user = sessions.User(id="github:7", login="u7", name="U", email=email, avatar_url=None)
    session = sessions.Session(id="s" * 32, role="member", created_at=now, expires_at=now + 3600, user=user)
    return sessions.encode_cookie(session)


def _as_user(email: str) -> TestClient:
    client = _client()
    client.cookies.set(sessions.COOKIE_NAME, _user_cookie(email))
    return client


def _put(body: dict[str, Any]) -> Any:
    r = _client().put("/v1/settings", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --- Access ---------------------------------------------------------------------


@pytest.mark.parametrize("role", ["guest", "member"])
def test_guests_and_members_get_the_unknown_route_404(
    role: str, settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEV_ROLE", role)
    client = _client()
    unknown = client.get("/v1/no-such-route")
    assert unknown.status_code == 404

    calls = [
        client.get("/v1/settings"),
        client.put("/v1/settings", json={"access": {"guest_analyses_per_hour": 99}}),
        client.put("/v1/settings", content=b"{not json", headers={"content-type": "application/json"}),
        client.delete("/v1/settings"),
        client.post("/v1/settings/assistant/test"),
    ]

    for r in calls:
        assert (r.status_code, r.json()) == (404, unknown.json())
    assert not settings_file.exists()


def test_signed_in_operator_email_gets_200_and_member_email_404(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEV_ROLE")
    monkeypatch.setenv("OPERATOR_EMAILS", "ops@example.org")
    monkeypatch.setenv("ALLOWED_EMAILS", "ana@example.org")

    assert _as_user("ops@example.org").get("/v1/settings").status_code == 200
    assert _as_user("ana@example.org").get("/v1/settings").status_code == 404


def test_dev_role_operator_gets_five_sections_and_no_key_value(settings_file: Path) -> None:
    r = _client().get("/v1/settings")

    assert r.status_code == 200
    body = r.json()
    assert set(body) == SECTIONS
    for value in FAKE_KEYS.values():
        assert value not in r.text
    assistant = body["assistant"]
    assert assistant["provider_order"] == ["gemini", "groq"]
    assert list(assistant["providers"])[:2] == ["gemini", "groq"]
    assert set(assistant["providers"]) == {"gemini", "groq", "ollama", "apertus", "anthropic", "mistral", "openai"}
    gemini, ollama = assistant["providers"]["gemini"], assistant["providers"]["ollama"]
    assert gemini["key_present"] is True and gemini["available"] is True and gemini["multiple_models"] is True
    assert ollama["key_required"] is False and ollama["available"] is False
    assert gemini["editable"] == {
        "enabled": True,
        "models": True,
        "multiple_models": False,
        "key_required": False,
        "key_present": False,
        "available": False,
    }
    assert gemini["source"] == {"enabled": "env", "models": "default"}
    assert (assistant["connected"], assistant["provider"]) == (True, "gemini")
    assert assistant["test_path"] == "/v1/settings/assistant/test"
    models = body["models"]
    assert models["download_policy"] == "members" and models["source"] == {"download_policy": "default"}
    assert models["pre_download"] == {"method": "POST", "path": "/v1/ml-models/{id}/download"}
    assert {"id", "installed", "size_bytes", "licence"} <= set(models["items"][0])
    access = body["access"]
    assert (access["guest_analyses_per_hour"], access["guest_session_hours"]) == (5, 24)
    assert access["sign_in"] == {"configured": False}
    assert access["editable"]["sign_in"] is False and access["editable"]["operator_emails"] is True
    data = body["data_sources"]
    assert data["earth_engine"] == {"configured": False, "project_id_present": False, "key_file_present": False}
    assert data["sample"]["name"] == "kakhovka_hls_l30"
    assert not any(data["editable"].values())
    server = body["server"]
    assert set(server["paths"]) == {"cache", "outputs", "temp", "uploads", "models"} == set(server["storage"])
    assert server["deployment"] == "local" and not any(server["editable"].values())
    assert server["jobs"]["total"] == 0 and server["jobs"]["by_status"]["completed"] == 0


# --- Runtime application --------------------------------------------------------


def test_put_provider_order_rebuilds_the_router_and_the_health_status(settings_file: Path) -> None:
    client = _client()
    assert client.get("/v1/health").json()["assistant"]["provider"] == "gemini"
    assert [p.name for p in get_llm_router().providers] == ["gemini", "groq"]

    body = _put({"assistant": {"provider_order": ["groq", "gemini"]}})

    assert body["assistant"]["provider_order"] == ["groq", "gemini"]
    assert body["assistant"]["source"]["provider_order"] == "override"
    assert [p.name for p in get_llm_router().providers] == ["groq", "gemini"]
    # Within the 30 s cache TTL: the PUT dropped the cached status.
    assert client.get("/v1/health").json()["assistant"]["provider"] == "groq"


def test_put_models_and_enabled_take_effect_at_runtime(settings_file: Path) -> None:
    body = _put(
        {
            "assistant": {
                "providers": {"gemini": {"models": ["gemini-x"]}, "groq": {"models": [" llama-y "], "enabled": False}}
            }
        }
    )

    assert body["assistant"]["providers"]["gemini"]["models"] == ["gemini-x"]
    assert body["assistant"]["providers"]["groq"]["enabled"] is False
    assert body["assistant"]["providers"]["groq"]["available"] is False
    assert [(p.name, p.model) for p in get_llm_router().providers] == [("gemini", "gemini-x")]
    assert json.loads(settings_file.read_text()) == {
        "GEMINI_MODELS": "gemini-x",
        "GROQ_MODEL": "llama-y",
        "LLM_ENABLE_GROQ": "false",
    }


def test_put_allowlists_change_role_resolution_on_the_next_request(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def role(email: str) -> str:
        monkeypatch.delenv("DEV_ROLE", raising=False)
        try:
            return str(_as_user(email).get("/v1/session").json()["role"])
        finally:
            monkeypatch.setenv("DEV_ROLE", "operator")

    assert role("ana@example.org") == "guest"  # not allowlisted: signed out

    body = _put({"access": {"allowed_emails": [" Ana@Example.org "], "allowed_domains": ["@UNIGE.ch"]}})

    assert body["access"]["allowed_emails"] == ["ana@example.org"]
    assert body["access"]["allowed_domains"] == ["unige.ch"]
    assert role("ana@example.org") == "member"
    assert role("bo@unige.ch") == "member"

    _put({"access": {"operator_emails": ["ana@example.org"]}})
    assert role("ana@example.org") == "operator"


def test_put_guest_limits_change_the_quota(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _put({"access": {"guest_analyses_per_hour": 1, "guest_session_hours": 2}})
    assert (body["access"]["guest_analyses_per_hour"], body["access"]["guest_session_hours"]) == (1, 2)

    monkeypatch.delenv("DEV_ROLE")
    info = _client().get("/v1/session").json()
    assert info["analyses"]["limit"] == 1 and info["kept_hours"] == 2
    guest = sessions.new_session()
    sessions.enforce_analysis_quota(guest)
    with pytest.raises(sessions.GuestQuotaError):
        sessions.enforce_analysis_quota(guest)


def test_put_download_policy_changes_who_may_download(settings_file: Path) -> None:
    assert sessions.can_download_models("member") is True

    body = _put({"models": {"download_policy": "operators"}})

    assert body["models"]["download_policy"] == "operators"
    assert body["models"]["source"] == {"download_policy": "override"}
    assert sessions.can_download_models("member") is False
    assert sessions.can_download_models("operator") is True


# --- Persistence and precedence -------------------------------------------------


def _env_file_digest() -> str | None:
    env = ROOT_DIR / ".env"
    return hashlib.sha256(env.read_bytes()).hexdigest() if env.exists() else None


def test_put_persists_atomically_and_survives_a_restart(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_before = _env_file_digest()

    _put({"assistant": {"provider_order": ["groq"]}, "access": {"guest_analyses_per_hour": 3}})

    assert json.loads(settings_file.read_text()) == {"GUEST_ANALYSES_PER_HOUR": "3", "LLM_PROVIDER_ORDER": "groq"}
    assert sorted(p.name for p in settings_file.parent.iterdir()) == ["settings.json"]  # no temp file left
    assert _env_file_digest() == env_before

    # A new process: nothing cached in memory, a new client.
    monkeypatch.setattr(overrides, "_cache", None)
    reset_llm_router()
    body = TestClient(app).get("/v1/settings").json()

    assert body["assistant"]["provider_order"] == ["groq"]
    assert body["access"]["guest_analyses_per_hour"] == 3
    assert body["access"]["source"]["guest_analyses_per_hour"] == "override"
    assert [p.name for p in get_llm_router().providers] == ["groq"]


def test_override_wins_over_env_and_null_restores_env(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUEST_ANALYSES_PER_HOUR", "7")
    assert _client().get("/v1/settings").json()["access"]["source"]["guest_analyses_per_hour"] == "env"

    assert _put({"access": {"guest_analyses_per_hour": 2}})["access"]["guest_analyses_per_hour"] == 2
    assert sessions.guest_analyses_per_hour() == 2

    body = _put({"access": {"guest_analyses_per_hour": None}})

    assert body["access"]["guest_analyses_per_hour"] == 7
    assert body["access"]["source"]["guest_analyses_per_hour"] == "env"
    assert json.loads(settings_file.read_text()) == {}


def test_settings_path_defaults_to_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DT4LC_SETTINGS_PATH")
    assert overrides.settings_path() == CACHE_PATH / "settings.json"


def test_overrides_file_ignores_unknown_and_secret_keys(settings_file: Path) -> None:
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(json.dumps({"GEMINI_API_KEY": "from-file", "GUEST_SESSION_HOURS": "5", "X": "y"}))

    assert overrides.getenv("GEMINI_API_KEY") == FAKE_KEYS["GEMINI_API_KEY"]
    assert overrides.getenv("GUEST_SESSION_HOURS") == "5"
    with pytest.raises(KeyError):
        overrides.write_overrides({"GEMINI_API_KEY": "x"})


# --- Validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "loc"),
    [
        ({"server": {"deployment": "demo"}}, ["body", "server"]),
        ({"data_sources": {}}, ["body", "data_sources"]),
        ({"access": {"bogus": 1}}, ["body", "access", "bogus"]),
        ({"access": {"sign_in": {"configured": True}}}, ["body", "access", "sign_in"]),
        ({"assistant": {"providers": {"gemini": {"key_present": True}}}}, None),
        ({"assistant": {"providers": {"gemini": {"api_key": "x"}}}}, None),
        ({"assistant": {"providers": {"gpt": {"enabled": True}}}}, None),
        ({"access": {"guest_analyses_per_hour": "5"}}, ["body", "access", "guest_analyses_per_hour"]),
        ({"access": {"guest_session_hours": True}}, ["body", "access", "guest_session_hours"]),
        ({"access": {"guest_analyses_per_hour": 1001}}, ["body", "access", "guest_analyses_per_hour"]),
        ({"access": {"guest_analyses_per_hour": -1}}, ["body", "access", "guest_analyses_per_hour"]),
        ({"access": {"guest_session_hours": 0}}, ["body", "access", "guest_session_hours"]),
        ({"access": {"allowed_emails": ["not-an-email"]}}, ["body", "access", "allowed_emails", 0]),
        ({"access": {"operator_emails": ["a@b.org,c@d.org"]}}, ["body", "access", "operator_emails", 0]),
        ({"access": {"allowed_domains": ["exa mple.org"]}}, ["body", "access", "allowed_domains", 0]),
        ({"access": {"allowed_domains": "unige.ch"}}, ["body", "access", "allowed_domains"]),
        ({"assistant": {"provider_order": ["gpt"]}}, ["body", "assistant", "provider_order", 0]),
        ({"assistant": {"provider_order": ["groq", "groq"]}}, ["body", "assistant", "provider_order"]),
        ({"assistant": {"provider_order": []}}, ["body", "assistant", "provider_order"]),
        ({"assistant": {"providers": {"groq": {"models": ["a", "b"]}}}}, ["body", "assistant"]),
        ({"assistant": {"providers": {"gemini": {"models": []}}}}, None),
        ({"assistant": {"providers": {"gemini": {"models": ["a,b"]}}}}, None),
        ({"assistant": {"providers": {"gemini": {"enabled": "yes"}}}}, None),
        ({"models": {"download_policy": "maybe"}}, ["body", "models", "download_policy"]),
    ],
)
def test_invalid_put_is_422_with_field_errors(
    body: dict[str, Any], loc: list[Any] | None, settings_file: Path
) -> None:
    r = _client().put("/v1/settings", json=body)

    assert r.status_code == 422, r.text
    errors = r.json()["detail"]
    assert errors and all("loc" in e and "msg" in e for e in errors)
    if loc is not None:
        assert loc in [e["loc"] for e in errors]
    assert not settings_file.exists()


def test_signed_in_operator_cannot_drop_their_own_operator_email(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEV_ROLE")
    monkeypatch.setenv("OPERATOR_EMAILS", "ops@example.org")
    client = _as_user("ops@example.org")

    r = client.put("/v1/settings", json={"access": {"operator_emails": ["other@example.org"]}})

    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "access", "operator_emails"]
    ok = client.put("/v1/settings", json={"access": {"operator_emails": ["OPS@example.org", "other@example.org"]}})
    assert ok.status_code == 200
    assert ok.json()["access"]["operator_emails"] == ["ops@example.org", "other@example.org"]


def test_signed_in_operator_cannot_reset_operator_emails_to_an_env_without_them(
    settings_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BE77: ``null`` ("Reset to .env") applies the env list, so the caller must be in it."""
    monkeypatch.delenv("DEV_ROLE")
    monkeypatch.setenv("OPERATOR_EMAILS", "other@example.org")
    overrides.write_overrides({"OPERATOR_EMAILS": "ops@example.org"})
    client = _as_user("ops@example.org")

    r = client.put("/v1/settings", json={"access": {"operator_emails": None}})

    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "access", "operator_emails"]
    assert overrides.overrides()["OPERATOR_EMAILS"] == "ops@example.org"
    monkeypatch.setenv("OPERATOR_EMAILS", "other@example.org,OPS@example.org")
    ok = client.put("/v1/settings", json={"access": {"operator_emails": None}})
    assert ok.status_code == 200
    assert ok.json()["access"]["operator_emails"] == ["other@example.org", "ops@example.org"]
    assert ok.json()["access"]["source"]["operator_emails"] == "env"


# --- Assistant test endpoint ----------------------------------------------------


def _provider(name: str, model: str, *, available: bool = True, error: Exception | None = None) -> MagicMock:
    provider = MagicMock()
    provider.name, provider.model = name, model
    provider.is_available.return_value = available
    if error is not None:
        provider.generate.side_effect = error
    else:
        provider.generate.return_value = LLMResponse(text="OK", model=model, provider=name)
    return provider


def test_assistant_test_probes_providers_in_order_with_keys_redacted(
    settings_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    leak = f"401 for key {FAKE_KEYS['GROQ_API_KEY']} (also AIzaSyOTHERKEYVALUE123)"
    router = MagicMock()
    router.providers = [
        _provider("gemini", "gemini-x"),
        _provider("groq", "llama-y", error=RuntimeError(leak)),
        _provider("ollama", "llama3.2", available=False),
    ]

    with caplog.at_level(logging.DEBUG), patch("server.settings.get_llm_router", return_value=router):
        r = _client().post("/v1/settings/assistant/test")

    assert r.status_code == 200
    body = r.json()
    assert body["connected"] is True and body["error"] is None
    assert [(p["name"], p["model"], p["ok"]) for p in body["providers"]] == [
        ("gemini", "gemini-x", True),
        ("groq", "llama-y", False),
        ("ollama", "llama3.2", False),
    ]
    assert all(isinstance(p["latency_ms"], int) and p["latency_ms"] >= 0 for p in body["providers"])
    assert body["providers"][0]["error"] is None
    assert body["providers"][1]["error"] == "RuntimeError: 401 for key *** (also ***)"
    assert body["providers"][2]["error"] == "not available"
    for value in (*FAKE_KEYS.values(), "AIzaSyOTHERKEYVALUE123"):
        assert value not in r.text and value not in caplog.text


def test_assistant_test_times_out_a_slow_provider(settings_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_mod, "PROBE_TIMEOUT_S", 0.05)
    slow = _provider("ollama", "llama3.2")
    slow.generate.side_effect = lambda *a, **k: time.sleep(0.5)
    router = MagicMock()
    router.providers = [slow]

    with patch("server.settings.get_llm_router", return_value=router):
        body = _client().post("/v1/settings/assistant/test").json()

    assert body["connected"] is False
    assert body["providers"][0]["ok"] is False and body["providers"][0]["error"].startswith("no answer in")


# --- Storage --------------------------------------------------------------------


def _tree(root: Path) -> None:
    (root / "a" / "b").mkdir(parents=True)
    (root / "one.bin").write_bytes(b"x" * 100)
    (root / "a" / "two.bin").write_bytes(b"x" * 20)
    (root / "a" / "b" / "three.bin").write_bytes(b"x" * 3)


def test_dir_usage_counts_a_temp_tree_and_caps_the_walk(tmp_path: Path) -> None:
    _tree(tmp_path)

    assert settings_mod.dir_usage(tmp_path) == {"bytes": 123, "files": 3, "truncated": False}
    assert settings_mod.dir_usage(tmp_path, cap=2)["truncated"] is True
    assert settings_mod.dir_usage(tmp_path / "missing") == {"bytes": 0, "files": 0, "truncated": False}


def test_server_storage_and_uploads_on_a_temp_tree(
    settings_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    uploads = cache / "uploads"
    uploads.mkdir(parents=True)
    _tree(cache / "outputs")
    for name, size in (("ab12_x.tif", 50), ("ab12_x.png", 5), ("ab12_x.owner.json", 2)):
        (uploads / name).write_bytes(b"x" * size)
    for name, value in (
        ("CACHE_PATH", cache),
        ("OUTPUTS_PATH", cache / "outputs"),
        ("TEMP_PATH", cache / "tmp"),
        ("UPLOAD_DIR", uploads),
        ("MODELS_PATH", cache / "models"),
    ):
        monkeypatch.setattr(settings_mod, name, value)

    body = _client().get("/v1/settings").json()

    storage = body["server"]["storage"]
    assert storage["cache"] == {"bytes": 180, "files": 6, "truncated": False}
    assert storage["outputs"] == {"bytes": 123, "files": 3, "truncated": False}
    assert storage["temp"]["bytes"] == 0 and storage["models"]["bytes"] == 0
    assert body["server"]["paths"]["uploads"] == str(uploads)
    assert body["data_sources"]["uploads"] == {"path": str(uploads), "files": 1, "bytes": 57}
