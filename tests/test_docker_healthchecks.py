"""Healthchecks must use tools present in their images (Docker quickstart)."""

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_ollama_healthcheck_uses_ollama_cli() -> None:
    """ollama/ollama ships no curl/wget; a curl check never passes, so ollama-pull never starts."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    test = compose["services"]["ollama"]["healthcheck"]["test"]
    assert test[:2] == ["CMD", "ollama"], test


def test_frontend_healthcheck_targets_ipv4_loopback() -> None:
    """nginx listens on IPv4 only (`listen 80;`); busybox wget resolves localhost to [::1]."""
    dockerfile = (ROOT / "cognitive_ui" / "Dockerfile").read_text(encoding="utf-8")
    healthcheck = re.search(r"^HEALTHCHECK .*?(?<!\\)$", dockerfile, re.M | re.S)
    assert healthcheck, "frontend Dockerfile has no HEALTHCHECK"
    assert "http://127.0.0.1" in healthcheck.group(0), healthcheck.group(0)
