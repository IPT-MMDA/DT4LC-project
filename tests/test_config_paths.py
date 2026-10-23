"""Env overrides for the cache, temp and outputs roots in ``dta.config``.

The paths are computed at import time, so each case imports the modules in a
fresh interpreter with the env set.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from dta.config import RESOURCES_PATH, ROOT_DIR

_PROBE = """
import json
import dta.config as c
import server.layer_metadata_store as lms
import server.paths as sp
print(json.dumps({
    "cache": str(c.CACHE_PATH),
    "temp": str(c.TEMP_PATH),
    "outputs": str(c.OUTPUTS_PATH),
    "models": str(c.MODELS_PATH),
    "uploads": str(c.UPLOADS_PATH),
    "jobs_db": str(c.JOBS_DB_PATH),
    "gee_exports": str(sp.GEE_EXPORT_DIR),
    "layer_metadata": str(lms.METADATA_DIR),
}))
"""

_PATH_VARS = (
    "DT4LC_CACHE_PATH",
    "DT4LC_TEMP_PATH",
    "DT4LC_OUTPUTS_PATH",
    "DT4LC_MODEL_CACHE",
    "DT4LC_UPLOADS_PATH",
    "DT4LC_JOBS_DB",
)


def _probe(**overrides: str) -> dict[str, Path]:
    env = {k: v for k, v in os.environ.items() if k not in _PATH_VARS}
    env.update(overrides)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=ROOT_DIR, env=env, capture_output=True, text=True, check=True
    ).stdout
    return {k: Path(v) for k, v in json.loads(out.strip().splitlines()[-1]).items()}


def test_defaults_stay_under_resources_cache() -> None:
    paths = _probe()
    cache = RESOURCES_PATH / ".cache"
    assert paths["cache"] == cache
    assert paths["temp"] == cache / "tmp"
    assert paths["outputs"] == cache / "outputs"
    assert paths["gee_exports"] == cache / "gee_exports"


def test_cache_override_moves_every_derived_path(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    paths = _probe(DT4LC_CACHE_PATH=str(cache))
    assert paths == {
        "cache": cache,
        "temp": cache / "tmp",
        "outputs": cache / "outputs",
        "models": cache / "models",
        "uploads": cache / "uploads",
        "jobs_db": cache / "jobs.db",
        "gee_exports": cache / "gee_exports",
        "layer_metadata": cache / "layer_metadata",
    }
    assert (cache / "tmp").is_dir() and (cache / "outputs").is_dir()


def test_temp_and_outputs_overrides_win_over_cache(tmp_path: Path) -> None:
    paths = _probe(
        DT4LC_CACHE_PATH=str(tmp_path / "cache"),
        DT4LC_TEMP_PATH=str(tmp_path / "temp"),
        DT4LC_OUTPUTS_PATH=str(tmp_path / "outputs"),
    )
    assert paths["cache"] == tmp_path / "cache"
    assert paths["temp"] == tmp_path / "temp"
    assert paths["outputs"] == tmp_path / "outputs"
    assert (tmp_path / "temp").is_dir() and (tmp_path / "outputs").is_dir()
