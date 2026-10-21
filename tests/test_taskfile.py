"""Checks that Taskfile.yml tasks point at code that exists."""

import importlib
from pathlib import Path
import shlex

from fastapi import FastAPI
import yaml

TASKFILE = Path(__file__).resolve().parents[1] / "Taskfile.yml"


def _task_cmds(name: str) -> list[str]:
    tasks = yaml.safe_load(TASKFILE.read_text())["tasks"]
    return [str(cmd) for cmd in tasks[name]["cmds"]]


def test_run_api_dev_uvicorn_target_resolves() -> None:
    """The run:api:dev uvicorn app path resolves to the FastAPI app.

    Checks the type, not just hasattr: once any test has imported server.app, the
    package ``server`` has a submodule attribute ``app``, so ``server:app`` would pass.
    """
    (cmd,) = _task_cmds("run:api:dev")
    args = shlex.split(cmd)
    target = args[args.index("uvicorn") + 1]
    module_name, attr = target.split(":")
    obj = getattr(importlib.import_module(module_name), attr, None)
    assert isinstance(obj, FastAPI), target
