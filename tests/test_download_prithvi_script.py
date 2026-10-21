"""Tests for scripts/download_prithvi.py (the download:prithvi task), with the ModelManager mocked."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from dta.dti.models.model_manager import DownloadProgress, ModelStatus

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_prithvi.py"


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_prithvi", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manager(monkeypatch: pytest.MonkeyPatch, script: ModuleType, tmp_path: Path, **kwargs: Any) -> MagicMock:
    manager = MagicMock(**kwargs)
    manager.get_model_path.return_value = tmp_path / "prithvi-eo-v1-100m"
    monkeypatch.setattr(script, "get_model_manager", lambda: manager)
    return manager


def test_downloads_the_model_id_the_app_reads(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    progress = DownloadProgress(model_id="prithvi-eo-v1-100m", status=ModelStatus.DOWNLOADING)
    manager = _manager(monkeypatch, script, tmp_path)
    manager.is_model_available.return_value = False
    manager.start_download.return_value = progress
    manager.get_model_status.return_value = ModelStatus.AVAILABLE

    def finish(_: float) -> None:
        progress.status = ModelStatus.AVAILABLE

    monkeypatch.setattr(script.time, "sleep", finish)

    assert script.main(poll_seconds=0) == 0
    manager.start_download.assert_called_once_with("prithvi-eo-v1-100m")
    assert f"prithvi-eo-v1-100m: available at {tmp_path / 'prithvi-eo-v1-100m'}" in capsys.readouterr().out


def test_skips_download_when_already_available(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = _manager(monkeypatch, script, tmp_path)
    manager.is_model_available.return_value = True

    assert script.main(poll_seconds=0) == 0
    manager.start_download.assert_not_called()
    assert "already downloaded" in capsys.readouterr().out


def test_reports_failure_with_the_error(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    progress = DownloadProgress(model_id="prithvi-eo-v1-100m", status=ModelStatus.FAILED, error="404 Client Error")
    manager = _manager(monkeypatch, script, tmp_path)
    manager.is_model_available.return_value = False
    manager.start_download.return_value = progress
    manager.get_model_status.return_value = ModelStatus.FAILED

    assert script.main(poll_seconds=0) == 1
    assert "prithvi-eo-v1-100m: failed (404 Client Error)" in capsys.readouterr().err
