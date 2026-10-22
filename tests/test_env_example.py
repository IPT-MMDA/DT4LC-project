"""Consistency checks between .env.example and docker-compose.yml."""

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_does_not_override_compose_service_urls() -> None:
    """`cp .env.example .env` must not replace compose's service-hostname defaults.

    docker compose interpolates ``${VAR:-default}`` from the project ``.env``, so
    an active ``VAR=`` line in .env.example overrides a default such as
    ``http://ollama:11434`` with a host-only URL that is wrong inside the container.
    """
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = set(compose["services"])
    service_url_vars = set()
    for service in compose["services"].values():
        for entry in service.get("environment", []):
            m = re.fullmatch(r"(\w+)=\$\{\1:-https?://([\w-]+)[:/].*\}", entry)
            if m and m.group(2) in services:
                service_url_vars.add(m.group(1))
    assert "OLLAMA_BASE_URL" in service_url_vars

    env_lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    active = {line.split("=", 1)[0].strip() for line in env_lines if re.match(r"\s*\w+=", line)}
    assert not active & service_url_vars, f".env.example overrides compose service URLs: {active & service_url_vars}"
