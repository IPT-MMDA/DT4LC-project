"""Tests for ModelManager.

Tests model listing, status checking, path resolution,
download progress tracking, and deletion.
"""

from pathlib import Path

import pytest

from dta.dti.models.model_manager import (
    AVAILABLE_MODELS,
    DownloadProgress,
    ModelManager,
    ModelStatus,
)


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    """Create a ModelManager with a temp cache dir."""
    return ModelManager(cache_dir=tmp_path / "models")


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache = tmp_path / "test_models"
        ModelManager(cache_dir=cache)
        assert cache.exists()

    def test_has_known_models(self, manager: ModelManager) -> None:
        assert len(AVAILABLE_MODELS) >= 1


class TestModelListing:
    """Tests for list_models."""

    def test_list_models_returns_all(self, manager: ModelManager) -> None:
        models = manager.list_models()
        assert len(models) == len(AVAILABLE_MODELS)

    def test_list_models_has_required_fields(self, manager: ModelManager) -> None:
        models = manager.list_models()
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "status" in m
            assert "size_mb" in m

    def test_list_models_default_not_installed(self, manager: ModelManager) -> None:
        models = manager.list_models()
        for m in models:
            assert m["status"] in ("not_installed", "available")


class TestModelInfo:
    """Tests for get_model_info."""

    def test_get_known_model(self, manager: ModelManager) -> None:
        info = manager.get_model_info("prithvi-eo-v1-100m")
        assert info is not None
        assert info["id"] == "prithvi-eo-v1-100m"
        assert info["name"] == "Prithvi EO v1 (100M)"

    def test_get_unknown_model(self, manager: ModelManager) -> None:
        info = manager.get_model_info("nonexistent-model")
        assert info is None


class TestModelStatus:
    """Tests for get_model_status."""

    def test_unknown_model_not_installed(self, manager: ModelManager) -> None:
        status = manager.get_model_status("nonexistent-xyz")
        assert status == ModelStatus.NOT_INSTALLED

    def test_known_model_not_installed(self, manager: ModelManager) -> None:
        status = manager.get_model_status("prithvi-eo-v1-100m")
        assert status == ModelStatus.NOT_INSTALLED

    def test_available_model(self, manager: ModelManager) -> None:
        """Simulate a model being installed by creating the expected files."""
        model_id = "prithvi-eo-v1-100m"
        model_path = manager.get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)

        # Create the weight file
        (model_path / "Prithvi_EO_V1_100M.pt").write_bytes(b"fake weights")
        (model_path / "inference.py").write_text("# inference")
        (model_path / "prithvi_mae.py").write_text("# mae")
        (model_path / "config.json").write_text("{}")

        status = manager.get_model_status(model_id)
        assert status == ModelStatus.AVAILABLE


class TestModelPath:
    """Tests for get_model_path."""

    def test_known_model_path(self, manager: ModelManager) -> None:
        path = manager.get_model_path("prithvi-eo-v1-100m")
        assert path is not None
        assert path.name == "prithvi-eo-v1-100m"

    def test_unknown_model_path(self, manager: ModelManager) -> None:
        path = manager.get_model_path("nonexistent-xyz")
        assert path is None


class TestIsModelAvailable:
    """Tests for is_model_available."""

    def test_not_available_by_default(self, manager: ModelManager) -> None:
        assert not manager.is_model_available("prithvi-eo-v1-100m")


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_progress_defaults(self) -> None:
        p = DownloadProgress(model_id="test")
        assert p.status == ModelStatus.NOT_INSTALLED
        assert p.progress == 0.0
        assert p.downloaded_mb == 0.0
        assert p.total_mb == 0.0

    def test_progress_mbps(self) -> None:
        p = DownloadProgress(model_id="test", speed_bps=1024 * 1024)
        assert p.speed_mbps == 1.0

    def test_progress_eta(self) -> None:
        p = DownloadProgress(
            model_id="test",
            status=ModelStatus.DOWNLOADING,
            total_bytes=100 * 1024 * 1024,
            downloaded_bytes=50 * 1024 * 1024,
            speed_bps=1024 * 1024,
        )
        eta = p.eta_seconds
        assert eta is not None
        assert eta > 0

    def test_progress_eta_not_downloading(self) -> None:
        p = DownloadProgress(model_id="test", status=ModelStatus.NOT_INSTALLED)
        assert p.eta_seconds is None

    def test_progress_to_dict(self) -> None:
        p = DownloadProgress(
            model_id="test",
            progress=0.5,
            downloaded_bytes=50 * 1024 * 1024,
            total_bytes=100 * 1024 * 1024,
            speed_bps=2 * 1024 * 1024,
        )
        d = p.to_dict()
        assert d["percent"] == 50.0
        assert d["downloaded_mb"] == pytest.approx(50.0, abs=0.1)
        assert d["total_mb"] == pytest.approx(100.0, abs=0.1)


class TestDeleteModel:
    """Tests for model deletion."""

    def test_delete_nonexistent(self, manager: ModelManager) -> None:
        assert not manager.delete_model("nonexistent-xyz")

    def test_delete_not_installed(self, manager: ModelManager) -> None:
        assert not manager.delete_model("prithvi-eo-v1-100m")

    def test_delete_installed(self, manager: ModelManager) -> None:
        """Test deleting an installed model."""
        model_id = "prithvi-eo-v1-100m"
        model_path = manager.get_model_path(model_id)
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "weight.pt").write_bytes(b"fake")

        assert manager.delete_model(model_id) is True
        assert not model_path.exists()


class TestCancelDownload:
    """Tests for download cancellation."""

    def test_cancel_not_downloading(self, manager: ModelManager) -> None:
        assert not manager.cancel_download("prithvi-eo-v1-100m")

    def test_cancel_active(self, manager: ModelManager) -> None:
        """Test cancelling by injecting a downloading status."""
        model_id = "prithvi-eo-v1-100m"
        manager._downloads[model_id] = DownloadProgress(
            model_id=model_id,
            status=ModelStatus.DOWNLOADING,
        )
        assert manager.cancel_download(model_id) is True


class TestStartDownload:
    """Tests for start_download."""

    def test_unknown_model_raises(self, manager: ModelManager) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            manager.start_download("nonexistent-xyz")

    def test_already_downloading_raises(self, manager: ModelManager) -> None:
        model_id = "prithvi-eo-v1-100m"
        manager._downloads[model_id] = DownloadProgress(
            model_id=model_id,
            status=ModelStatus.DOWNLOADING,
        )
        with pytest.raises(ValueError, match="already downloading"):
            manager.start_download(model_id)


class TestModelManagerDownloads:
    """Tests for the actual download thread logic."""

    def test_download_thread_success(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from dta.dti.models.model_manager import ModelManager, ModelStatus

        manager = ModelManager(cache_dir=tmp_path)
        model_id = "prithvi-eo-v1-100m"

        # Create fake downloaded files so get_model_status returns AVAILABLE
        def fake_hf_hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
            dest = tmp_path / model_id / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake weights")
            return str(dest)

        with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_hub_download):
            progress = manager.start_download(model_id)
            # Wait for thread to finish
            manager._download_threads[model_id].join()

        assert manager.get_model_status(model_id) == ModelStatus.AVAILABLE
        assert progress.status == ModelStatus.AVAILABLE
        assert progress.progress == 1.0

    def test_download_thread_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from dta.dti.models.model_manager import ModelManager, ModelStatus

        manager = ModelManager(cache_dir=tmp_path)
        model_id = "prithvi-eo-v1-100m"

        with patch("huggingface_hub.hf_hub_download", side_effect=Exception("Network error")):
            progress = manager.start_download(model_id)
            manager._download_threads[model_id].join()

        assert manager.get_model_status(model_id) == ModelStatus.FAILED
        assert progress.error is not None
        assert len(progress.error) > 0

    def test_download_thread_cancel(self, tmp_path: Path) -> None:
        import time
        from unittest.mock import patch

        from dta.dti.models.model_manager import ModelManager, ModelStatus

        manager = ModelManager(cache_dir=tmp_path)
        model_id = "prithvi-eo-v1-100m"

        def slow_hf_download(repo_id: str, filename: str, **kwargs: object) -> str:
            for _ in range(20):
                if manager._cancel_flags.get(model_id, False):
                    raise Exception("Cancelled")
                time.sleep(0.05)
            dest = tmp_path / model_id / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake")
            return str(dest)

        with patch("huggingface_hub.hf_hub_download", side_effect=slow_hf_download):
            progress = manager.start_download(model_id)
            time.sleep(0.1)
            manager.cancel_download(model_id)
            manager._download_threads[model_id].join()

        # After cancellation, model should not be in AVAILABLE state
        assert progress.status in (ModelStatus.NOT_INSTALLED, ModelStatus.FAILED)


class TestModelManagerUtils:
    """Tests for model manager utility functions."""

    def test_get_gdal_package_spec(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from dta.dti.models.model_manager import ModelManager

        manager = ModelManager(cache_dir=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "3.10.3\n"

        with patch("subprocess.run", return_value=mock_proc):
            assert manager._get_gdal_package_spec() == "gdal==3.10.3"

        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            assert manager._get_gdal_package_spec() is None

    def test_install_pip_dependencies(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from dta.dti.models.model_manager import ModelManager

        manager = ModelManager(cache_dir=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with patch("subprocess.run", return_value=mock_proc):
            with patch.object(manager, "_get_gdal_package_spec", return_value="gdal==3.10.3"):
                manager._install_pip_dependencies(["numpy", "gdal"])

    def test_install_pip_dependencies_error(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from dta.dti.models.model_manager import ModelManager

        manager = ModelManager(cache_dir=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "pip error"

        with patch("subprocess.run", return_value=mock_proc):
            import pytest

            with pytest.raises(RuntimeError, match="Failed to install"):
                manager._install_pip_dependencies(["numpy"])

    def test_download_github_repo(self, tmp_path: Path) -> None:
        import io
        from unittest.mock import MagicMock, patch
        import zipfile

        from dta.dti.models.model_manager import ModelManager

        manager = ModelManager(cache_dir=tmp_path)
        dest = tmp_path / "repo"

        # Create a mock zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("repo-main/", "")
            zf.writestr("repo-main/test.txt", "content")
            zf.writestr("repo-main/subdir/", "")
            zf.writestr("repo-main/subdir/test2.txt", "content2")

        mock_response = MagicMock()
        mock_response.read.return_value = zip_buffer.getvalue()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock_response
            manager._download_github_repo("owner/repo", "main", dest)

        assert (dest / "test.txt").exists()
        assert (dest / "subdir" / "test2.txt").exists()
