"""The tag-triggered GitHub Release must give working instructions (docker.yml)."""

from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _release_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["release"]["steps"]
    return steps


def test_release_is_a_draft_without_the_source_less_bundle() -> None:
    """The zip held a compose file that builds from source it did not contain; publishing is manual."""
    steps = _release_steps()
    assert "Build deploy bundle" not in [s.get("name") for s in steps]
    create = next(s for s in steps if s.get("name") == "Create GitHub Release")
    assert create["with"]["draft"] is True
    assert "files" not in create["with"]


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("git") is None, reason="needs bash and git")
def test_release_notes_pull_lowercase_images_and_run_from_source(tmp_path: Path) -> None:
    """GHCR image names are lowercase; `docker pull ghcr.io/IPT-MMDA/...` is an invalid reference."""
    notes = next(s for s in _release_steps() if s.get("name") == "Generate release notes")
    script = notes["run"]
    for expr, value in {
        "${{ github.ref_name }}": "v0.5.0",
        "${{ github.repository }}": "IPT-MMDA/DT4LC-project",
        "${{ github.event.repository.name }}": "DT4LC-project",
    }.items():
        script = script.replace(expr, value)
    assert "${{" not in script
    output = tmp_path / "github_output"
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "BACKEND_REF": "ghcr.io/IPT-MMDA/DT4LC-project/backend:0.5.0",
        "FRONTEND_REF": "ghcr.io/IPT-MMDA/DT4LC-project/frontend:0.5.0",
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True, capture_output=True)
    text = output.read_text(encoding="utf-8")

    assert "docker pull ghcr.io/ipt-mmda/dt4lc-project/backend:0.5.0" in text
    assert "docker pull ghcr.io/ipt-mmda/dt4lc-project/frontend:0.5.0" in text
    assert "git clone --branch v0.5.0 https://github.com/IPT-MMDA/DT4LC-project.git" in text
    assert "docker compose --profile ollama up -d" in text
    assert "dt4lc-deploy" not in text
