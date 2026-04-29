"""Model Manager - handles download, deletion, and status of ML models.

Downloads models from HuggingFace Hub on demand. Provides progress tracking
for non-blocking downloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model availability status."""

    NOT_INSTALLED = "not_installed"
    DOWNLOADING = "downloading"
    AVAILABLE = "available"
    FAILED = "failed"


@dataclass
class ModelInfo:
    """Model metadata and download configuration."""

    id: str
    name: str
    description: str
    huggingface_repo: str
    files: list[str]
    size_mb: int
    license: str
    license_url: str
    # Optional: specific revision/commit
    revision: str | None = None
    # Optional: GitHub repo for additional code files
    github_repo: str | None = None
    github_tag: str | None = None  # Tag/branch to download (default: main)
    # Optional: pip dependencies to install when model is downloaded
    pip_dependencies: list[str] | None = None


@dataclass
class DownloadProgress:
    """Tracks download progress for a model."""

    model_id: str
    status: ModelStatus = ModelStatus.NOT_INSTALLED
    progress: float = 0.0  # 0.0 to 1.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0  # bytes per second
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def speed_mbps(self) -> float:
        return self.speed_bps / (1024 * 1024)

    @property
    def eta_seconds(self) -> int | None:
        if self.speed_bps <= 0 or self.status != ModelStatus.DOWNLOADING:
            return None
        remaining = self.total_bytes - self.downloaded_bytes
        return int(remaining / self.speed_bps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "percent": round(self.progress * 100, 1),
            "downloaded_mb": round(self.downloaded_mb, 1),
            "total_mb": round(self.total_mb, 1),
            "speed_mbps": round(self.speed_mbps, 2),
            "eta_seconds": self.eta_seconds,
        }


# Available models manifest
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "prithvi-eo-v1-100m": ModelInfo(
        id="prithvi-eo-v1-100m",
        name="Prithvi EO v1 (100M)",
        description="NASA/IBM geospatial foundation model for satellite imagery reconstruction and analysis",
        huggingface_repo="ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        files=[
            "Prithvi_EO_V1_100M.pt",
            "inference.py",
            "prithvi_mae.py",
            "config.json",  # Contains pretrained_cfg needed by inference.py
        ],
        size_mb=450,
        license="Apache-2.0",
        license_url="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
    ),
    "delineate-anything-small": ModelInfo(
        id="delineate-anything-small",
        name="Delineate-Anything (Small)",
        description="Field boundary detection model for agricultural parcels using YOLO-based segmentation",
        huggingface_repo="MykolaL/DelineateAnything",
        files=["DelineateAnything-S.pt"],
        size_mb=85,
        license="AGPL-3.0",
        license_url="https://github.com/Lavreniuk/Delineate-Anything",
        github_repo="Lavreniuk/Delineate-Anything",
        github_tag="main",
    ),
}


class ModelManager:
    """Manages model downloads, deletions, and status tracking.

    Models are downloaded to a cache directory (configurable via DT4LC_MODEL_CACHE
    environment variable or defaults to ~/.cache/dt4lc/models).

    Downloads run in background threads and don't block the main application.
    Progress can be tracked via get_download_progress().
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize model manager.

        Args:
            cache_dir: Directory for model cache. Defaults to MODELS_PATH
                       from config (resources/.cache/models/)
        """
        if cache_dir is None:
            from dta.config import MODELS_PATH

            cache_dir = MODELS_PATH

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track active downloads
        self._downloads: dict[str, DownloadProgress] = {}
        self._download_threads: dict[str, threading.Thread] = {}
        self._cancel_flags: dict[str, bool] = {}
        self._lock = threading.Lock()

        logger.info(f"ModelManager initialized with cache_dir: {self.cache_dir}")

    def list_models(self) -> list[dict[str, Any]]:
        """List all available models with their current status.

        Returns:
            List of model info dicts with status
        """
        models = []
        for model_id, info in AVAILABLE_MODELS.items():
            status = self.get_model_status(model_id)
            model_dict = {
                "id": info.id,
                "name": info.name,
                "description": info.description,
                "size_mb": info.size_mb,
                "license": info.license,
                "license_url": info.license_url,
                "status": status.value,
                "path": str(self.get_model_path(model_id)) if status == ModelStatus.AVAILABLE else None,
            }

            # Add download progress if downloading
            if status == ModelStatus.DOWNLOADING:
                progress = self.get_download_progress(model_id)
                if progress:
                    model_dict["download_progress"] = progress.to_dict()

            models.append(model_dict)

        return models

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get detailed info for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model info dict or None if not found
        """
        if model_id not in AVAILABLE_MODELS:
            return None

        info = AVAILABLE_MODELS[model_id]
        status = self.get_model_status(model_id)

        result = {
            "id": info.id,
            "name": info.name,
            "description": info.description,
            "size_mb": info.size_mb,
            "license": info.license,
            "license_url": info.license_url,
            "huggingface_repo": info.huggingface_repo,
            "status": status.value,
            "path": str(self.get_model_path(model_id)) if status == ModelStatus.AVAILABLE else None,
        }

        # Add download progress if downloading
        if status == ModelStatus.DOWNLOADING:
            progress = self.get_download_progress(model_id)
            if progress:
                result["download_progress"] = progress.to_dict()
                if progress.error:
                    result["error"] = progress.error

        return result

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Check if a model is installed, downloading, or not installed.

        Args:
            model_id: Model identifier

        Returns:
            Current model status
        """
        if model_id not in AVAILABLE_MODELS:
            return ModelStatus.NOT_INSTALLED

        # Check if currently downloading
        with self._lock:
            if model_id in self._downloads:
                return self._downloads[model_id].status

        # Check if files exist
        model_path = self.get_model_path(model_id)
        if model_path and model_path.exists():
            info = AVAILABLE_MODELS[model_id]

            # Check HuggingFace weight files
            required_files = [f for f in info.files if f.endswith((".pt", ".pth", ".bin"))]
            if not required_files:
                required_files = info.files[:1]  # At least first file

            hf_files_exist = all((model_path / f).exists() for f in required_files)

            # Check GitHub code files if specified
            github_files_exist = True
            if info.github_repo:
                # For GitHub repos, check for key entry point files
                github_entry_points = ["delineate.py", "inference.py", "main.py"]
                github_files_exist = any((model_path / f).exists() for f in github_entry_points)

            if hf_files_exist and github_files_exist:
                return ModelStatus.AVAILABLE

        return ModelStatus.NOT_INSTALLED

    def get_model_path(self, model_id: str) -> Path | None:
        """Get the path where a model is/would be stored.

        Args:
            model_id: Model identifier

        Returns:
            Path to model directory or None if unknown model
        """
        if model_id not in AVAILABLE_MODELS:
            return None
        return self.cache_dir / model_id

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is downloaded and ready to use.

        Args:
            model_id: Model identifier

        Returns:
            True if model is available
        """
        return self.get_model_status(model_id) == ModelStatus.AVAILABLE

    def get_download_progress(self, model_id: str) -> DownloadProgress | None:
        """Get current download progress for a model.

        Args:
            model_id: Model identifier

        Returns:
            Download progress or None if not downloading
        """
        with self._lock:
            return self._downloads.get(model_id)

    def start_download(
        self,
        model_id: str,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> DownloadProgress:
        """Start downloading a model in the background.

        Args:
            model_id: Model identifier
            progress_callback: Optional callback for progress updates

        Returns:
            Initial download progress object

        Raises:
            ValueError: If model_id is unknown or already downloading
        """
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        with self._lock:
            if model_id in self._downloads and self._downloads[model_id].status == ModelStatus.DOWNLOADING:
                raise ValueError(f"Model {model_id} is already downloading")

            # Initialize progress
            info = AVAILABLE_MODELS[model_id]
            progress = DownloadProgress(
                model_id=model_id,
                status=ModelStatus.DOWNLOADING,
                total_bytes=info.size_mb * 1024 * 1024,
                started_at=datetime.now(),
            )
            self._downloads[model_id] = progress
            self._cancel_flags[model_id] = False

        # Start background download
        thread = threading.Thread(
            target=self._download_worker,
            args=(model_id, progress_callback),
            daemon=True,
        )
        self._download_threads[model_id] = thread
        thread.start()

        logger.info(f"Started download for {model_id}")
        return progress

    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download.

        Args:
            model_id: Model identifier

        Returns:
            True if cancellation was requested
        """
        with self._lock:
            if model_id not in self._downloads:
                return False
            if self._downloads[model_id].status != ModelStatus.DOWNLOADING:
                return False
            self._cancel_flags[model_id] = True

        logger.info(f"Cancellation requested for {model_id}")
        return True

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model.

        Args:
            model_id: Model identifier

        Returns:
            True if model was deleted
        """
        if model_id not in AVAILABLE_MODELS:
            return False

        model_path = self.get_model_path(model_id)
        if model_path and model_path.exists():
            try:
                shutil.rmtree(model_path)
                logger.info(f"Deleted model {model_id} from {model_path}")

                # Clear any download state
                with self._lock:
                    self._downloads.pop(model_id, None)
                    self._cancel_flags.pop(model_id, None)

                return True
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
                return False

        return False

    def _download_worker(
        self,
        model_id: str,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> None:
        """Background worker for downloading a model.

        Downloads model weights from HuggingFace and optionally code from GitHub.

        Args:
            model_id: Model identifier
            progress_callback: Optional callback for progress updates
        """
        try:
            from huggingface_hub import hf_hub_download

            info = AVAILABLE_MODELS[model_id]
            model_path = self.get_model_path(model_id)
            if model_path is None:
                raise ValueError(f"Unknown model: {model_id}")
            model_path.mkdir(parents=True, exist_ok=True)

            # Calculate total steps (HF files + GitHub if present)
            total_steps = len(info.files) + (1 if info.github_repo else 0)
            completed_steps = 0

            # Step 1: Download GitHub repo if specified
            if info.github_repo:
                if self._cancel_flags.get(model_id, False):
                    self._handle_cancellation(model_id, model_path)
                    return

                logger.info(f"Downloading GitHub repo {info.github_repo} for {model_id}")
                start_time = time.time()
                self._download_github_repo(info.github_repo, info.github_tag or "main", model_path)
                elapsed = time.time() - start_time

                completed_steps += 1
                self._update_progress(model_id, completed_steps, total_steps, elapsed, progress_callback)

            # Step 2: Install pip dependencies if specified
            if info.pip_dependencies:
                if self._cancel_flags.get(model_id, False):
                    self._handle_cancellation(model_id, model_path)
                    return

                logger.info(f"Installing pip dependencies for {model_id}: {info.pip_dependencies}")
                self._install_pip_dependencies(info.pip_dependencies)

            # Step 3: Download HuggingFace files (model weights)
            for filename in info.files:
                if self._cancel_flags.get(model_id, False):
                    self._handle_cancellation(model_id, model_path)
                    return

                logger.info(f"Downloading {filename} for {model_id}")

                start_time = time.time()
                downloaded_path = hf_hub_download(
                    repo_id=info.huggingface_repo,
                    filename=filename,
                    revision=info.revision,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                )
                elapsed = time.time() - start_time

                completed_steps += 1
                file_size = Path(downloaded_path).stat().st_size
                self._update_progress(model_id, completed_steps, total_steps, elapsed, progress_callback, file_size)

            # Mark as complete
            with self._lock:
                progress = self._downloads[model_id]
                progress.status = ModelStatus.AVAILABLE
                progress.progress = 1.0
                progress.downloaded_bytes = progress.total_bytes
                progress.completed_at = datetime.now()

            logger.info(f"Download complete for {model_id}")

            if progress_callback:
                progress_callback(self._downloads[model_id])

            # Clean up download tracking after a short delay
            time.sleep(2)
            with self._lock:
                self._downloads.pop(model_id, None)
                self._cancel_flags.pop(model_id, None)

        except Exception as e:
            logger.error(f"Download failed for {model_id}: {e}")
            with self._lock:
                if model_id in self._downloads:
                    self._downloads[model_id].status = ModelStatus.FAILED
                    self._downloads[model_id].error = str(e)

            if progress_callback:
                progress_callback(self._downloads[model_id])

    def _handle_cancellation(self, model_id: str, model_path: Path) -> None:
        """Handle download cancellation."""
        logger.info(f"Download cancelled for {model_id}")
        with self._lock:
            self._downloads[model_id].status = ModelStatus.NOT_INSTALLED
            self._downloads[model_id].error = "Cancelled by user"
        if model_path.exists():
            shutil.rmtree(model_path)

    def _update_progress(
        self,
        model_id: str,
        completed: int,
        total: int,
        elapsed: float,
        callback: Callable[[DownloadProgress], None] | None,
        file_size: int | None = None,
    ) -> None:
        """Update download progress."""
        with self._lock:
            progress = self._downloads[model_id]
            progress.progress = completed / total
            progress.downloaded_bytes = int(progress.progress * progress.total_bytes)
            if elapsed > 0 and file_size:
                progress.speed_bps = file_size / elapsed

        if callback:
            callback(self._downloads[model_id])

    def _install_pip_dependencies(self, dependencies: list[str]) -> None:
        """Install pip dependencies for a model.

        Args:
            dependencies: List of pip package specifications (e.g., ["gdal", "numpy>=1.20"])

        Raises:
            RuntimeError: If installation fails
        """
        import subprocess
        import sys

        for dep in dependencies:
            # Handle GDAL specially - must match system libgdal version
            pkg = dep
            if dep.lower() == "gdal":
                gdal_spec = self._get_gdal_package_spec()
                if gdal_spec is None:
                    raise RuntimeError("GDAL system library not found. Install libgdal-dev first.")
                pkg = gdal_spec

            logger.info(f"Installing {pkg}...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per package
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to install {pkg}: {result.stderr}")
                logger.info(f"Successfully installed {pkg}")
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"Timeout installing {pkg}") from e

    def _get_gdal_package_spec(self) -> str | None:
        """Get GDAL pip package spec matching system libgdal version.

        Returns:
            Package spec like "gdal==3.10.3" or None if gdal-config not found
        """
        import subprocess

        try:
            result = subprocess.run(
                ["gdal-config", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"Detected system GDAL version: {version}")
                return f"gdal=={version}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    def _download_github_repo(self, repo: str, tag: str, dest_path: Path) -> None:
        """Download and extract GitHub repository as ZIP.

        Args:
            repo: GitHub repository in format "owner/repo"
            tag: Git tag or branch name
            dest_path: Destination directory
        """
        import io
        import urllib.request
        import zipfile

        zip_url = f"https://github.com/{repo}/archive/refs/heads/{tag}.zip"
        logger.info(f"Downloading {zip_url}")

        with urllib.request.urlopen(zip_url, timeout=60) as response:
            zip_data = response.read()

        # Extract ZIP contents
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # ZIP contains a top-level folder like "Delineate-Anything-main/"
            # We need to extract contents without this prefix
            prefix = None
            for name in zf.namelist():
                if prefix is None:
                    # First entry is the top-level directory
                    prefix = name.split("/")[0] + "/"
                    continue

                # Skip the prefix directory itself
                if name == prefix:
                    continue

                # Get relative path without prefix
                rel_path = name[len(prefix) :]
                if not rel_path:
                    continue

                target_path = dest_path / rel_path

                if name.endswith("/"):
                    # Directory
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    # File
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

        logger.info(f"Extracted GitHub repo to {dest_path}")


# Global singleton instance
_manager: ModelManager | None = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance.

    Returns:
        Singleton ModelManager instance
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelManager()
    return _manager


def reset_model_manager() -> None:
    """Reset the global ModelManager instance (mainly for testing)."""
    global _manager
    _manager = None
