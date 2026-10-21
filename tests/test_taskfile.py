"""Checks that Taskfile.yml tasks point at code that exists."""

import importlib
from pathlib import Path
import shlex

import yaml

TASKFILE = Path(__file__).resolve().parents[1] / "Taskfile.yml"


def _task_cmds(name: str) -> list[str]:
    tasks = yaml.safe_load(TASKFILE.read_text())["tasks"]
    return [str(cmd) for cmd in tasks[name]["cmds"]]


def test_run_api_dev_uvicorn_target_resolves() -> None:
    """The run:api:dev uvicorn app path is an importable module:attribute."""
    (cmd,) = _task_cmds("run:api:dev")
    args = shlex.split(cmd)
    target = args[args.index("uvicorn") + 1]
    module_name, attr = target.split(":")
    assert hasattr(importlib.import_module(module_name), attr), target
