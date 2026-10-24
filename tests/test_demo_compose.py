"""The public-demo stack (docker-compose.demo.yml, deploy/): v0.6.0 run 3, Phase 5.

The override must keep nginx the only way in (no published backend port), trust X-Forwarded-For
from the demo network only, pre-install the models before the backend starts, and leave the
default stack alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
from types import ModuleType
from typing import Any

import pytest
import yaml

from dta.dti.models.model_manager import AVAILABLE_MODELS, DownloadProgress, ModelStatus

ROOT = Path(__file__).resolve().parents[1]
SUBNET = "${DEMO_SUBNET:-172.30.60.0/24}"


@dataclass
class Reset:
    """A compose ``!reset`` value: the key is cleared, not merged with the base file."""

    value: Any


class _ComposeLoader(yaml.SafeLoader):
    pass


_ComposeLoader.add_constructor(
    "!reset",
    lambda loader, node: Reset(
        loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_object(node)
    ),
)


def _load(name: str) -> dict[str, Any]:
    return yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)


def _env(service: dict[str, Any]) -> dict[str, str]:
    return dict(entry.split("=", 1) for entry in service.get("environment", []))


def _mounts(service: dict[str, Any]) -> dict[str, str]:
    """``{target: source}`` of the short-syntax volumes."""
    return {v.split(":")[1]: v.split(":")[0] for v in service.get("volumes", [])}


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    return _load("docker-compose.demo.yml")


def test_backend_publishes_no_port(demo: dict[str, Any]) -> None:
    ports = demo["services"]["backend"]["ports"]
    assert isinstance(ports, Reset) and ports.value == [], "the demo must drop the base file's 8000 mapping"


def test_ollama_publishes_no_port(demo: dict[str, Any]) -> None:
    ports = demo["services"]["ollama"]["ports"]
    assert isinstance(ports, Reset) and ports.value == [], "the demo must drop the base file's 11434 mapping"


def test_backend_runs_as_the_demo_and_never_downloads_for_guests(demo: dict[str, Any]) -> None:
    env = _env(demo["services"]["backend"])
    assert env["DT4LC_DEPLOYMENT"] == "demo"
    assert env["MODEL_DOWNLOAD_ON_DEMAND"] in ("none", "operators")


def test_forwarded_ips_trust_only_the_pinned_demo_network(demo: dict[str, Any]) -> None:
    assert demo["networks"]["default"]["ipam"]["config"] == [{"subnet": SUBNET}]
    assert _env(demo["services"]["backend"])["FORWARDED_ALLOW_IPS"] == SUBNET
    assert _env(demo["services"]["frontend"])["DEMO_SUBNET"] == SUBNET


def test_models_are_installed_before_the_backend_starts(demo: dict[str, Any]) -> None:
    services = demo["services"]
    init, backend = services["models-init"], services["backend"]
    assert init["profiles"] == ["demo"]
    assert init["restart"] == "no"
    assert backend["depends_on"]["models-init"] == {"condition": "service_completed_successfully"}

    # Both see the same models volume at their DT4LC_MODEL_CACHE
    for service in (init, backend):
        cache = _env(service)["DT4LC_MODEL_CACHE"]
        assert _mounts(service)[cache] == "models_data"
    assert "models_data" in demo["volumes"]
    script = _mounts(init)["/app/deploy/models_init.py"]
    assert (ROOT / script).is_file()
    assert init["command"] == ["python", "/app/deploy/models_init.py"]


def test_frontend_serves_the_demo_nginx_conf(demo: dict[str, Any]) -> None:
    source = _mounts(demo["services"]["frontend"])["/etc/nginx/templates/default.conf.template"]
    assert source == "./deploy/nginx-demo.conf.template"
    assert (ROOT / source).is_file()


def test_default_stack_is_untouched() -> None:
    base = _load("docker-compose.yml")
    assert "models-init" not in base["services"]
    assert "DT4LC_DEPLOYMENT" not in _env(base["services"]["backend"])
    assert base["services"]["backend"]["ports"] == ["${BACKEND_PORT:-8000}:8000"]
    assert "networks" not in base


def test_demo_nginx_limits_and_forwards_the_client_ip() -> None:
    conf = (ROOT / "deploy" / "nginx-demo.conf.template").read_text(encoding="utf-8")
    assert "set_real_ip_from ${DEMO_SUBNET};" in conf
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    assert "proxy_set_header X-Real-IP $remote_addr;" in conf

    # The strict zone counts POSTs per client IP; the API zone counts every request
    assert re.search(r"POST\s+\$binary_remote_addr;", conf)
    assert re.search(r"limit_req_zone \$dt4lc_post_client zone=dt4lc_post:\S+ rate=10r/m;", conf)
    assert re.search(r"limit_req_zone \$binary_remote_addr zone=dt4lc_api:\S+ rate=\d+r/s;", conf)
    heavy = re.search(r"location ~ \^/v1/\(([^)]*)\)\$ \{(.*?)\}", conf, re.S)
    assert heavy, "no location for the rate-limited POST endpoints"
    assert {"jobs", "chat", "upload", "sample"} <= set(heavy.group(1).split("|"))
    strict = re.compile(rf"^/v1/({heavy.group(1)})$")
    assert strict.match("/v1/jobs/3f2c9a1e/confirm"), "confirming a proposal starts work"
    assert not strict.match("/v1/jobs/3f2c9a1e/events") and not strict.match("/v1/jobs/a/b/confirm")
    assert "limit_req zone=dt4lc_post " in heavy.group(2)
    assert "client_max_body_size 210m;" in heavy.group(2)  # the backend's 200 MB upload limit

    body = re.search(r"return 429 '(.*)';", conf)
    assert body, "no 429 body"
    error = json.loads(body.group(1))  # the project's error schema
    assert error["ok"] is False
    assert error["error"]["code"] == "RATE_LIMITED"
    assert error["error"]["message"]
    assert "limit_req_status 429;" in conf


def _models_init() -> ModuleType:
    spec = importlib.util.spec_from_file_location("models_init", ROOT / "deploy" / "models_init.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeManager:
    def __init__(self, installed: set[str], fail: set[str] | None = None) -> None:
        self.installed = set(installed)
        self.fail = fail or set()
        self.downloaded: list[str] = []

    def is_model_available(self, model_id: str) -> bool:
        return model_id in self.installed

    def get_model_path(self, model_id: str) -> Path:
        return Path("/app/models") / model_id

    def download(self, model_id: str, poll_s: float = 1.0) -> DownloadProgress:
        self.downloaded.append(model_id)
        if model_id in self.fail:
            return DownloadProgress(model_id, status=ModelStatus.FAILED, error="network down")
        self.installed.add(model_id)
        return DownloadProgress(model_id, status=ModelStatus.AVAILABLE)


def test_models_init_installs_missing_models_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_SKIP_MODEL_DOWNLOAD", raising=False)
    main = _models_init().main
    manager = _FakeManager(installed={"prithvi-eo-v1-100m"})

    assert main(manager) == 0
    assert manager.downloaded == [m for m in AVAILABLE_MODELS if m != "prithvi-eo-v1-100m"]
    assert manager.installed == set(AVAILABLE_MODELS)

    assert main(manager) == 0  # a second run skips installed models
    assert len(manager.downloaded) == len(AVAILABLE_MODELS) - 1


def test_models_init_fails_when_a_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_SKIP_MODEL_DOWNLOAD", raising=False)
    manager = _FakeManager(installed=set(), fail={"delineate-anything-small"})
    assert _models_init().main(manager) == 1
    assert manager.downloaded == list(AVAILABLE_MODELS)  # the others are still installed


def test_models_init_can_be_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_SKIP_MODEL_DOWNLOAD", "1")
    manager = _FakeManager(installed=set())
    assert _models_init().main(manager) == 0
    assert manager.downloaded == []
