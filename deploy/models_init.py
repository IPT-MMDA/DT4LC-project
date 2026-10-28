"""Pre-install the model weights for the public demo (the ``models-init`` service).

Downloads every managed model (Prithvi EO v1 and Delineate-Anything) through the project's
ModelManager into ``DT4LC_MODEL_CACHE`` (the ``models_data`` volume), so guests never trigger a
download. A model already installed is skipped, so running it again is quick and harmless.
``DEMO_SKIP_MODEL_DOWNLOAD=1`` skips every download (smoke tests of the demo stack).
"""

from __future__ import annotations

import os
import sys

from dta.dti.models.model_manager import AVAILABLE_MODELS, ModelManager, ModelStatus, get_model_manager


def main(manager: ModelManager | None = None) -> int:
    """Install the missing models; 0 when all are available (or skipped), 1 if any failed."""
    if os.environ.get("DEMO_SKIP_MODEL_DOWNLOAD", "").strip().lower() in ("1", "true", "yes"):
        print("models-init: DEMO_SKIP_MODEL_DOWNLOAD is set, skipping model downloads")
        return 0
    manager = manager or get_model_manager()
    failed = 0
    for model_id in AVAILABLE_MODELS:
        if manager.is_model_available(model_id):
            print(f"models-init: {model_id} already installed at {manager.get_model_path(model_id)}")
            continue
        print(f"models-init: downloading {model_id} to {manager.get_model_path(model_id)} ...", flush=True)
        progress = manager.download(model_id, poll_s=2.0)
        if progress.status == ModelStatus.AVAILABLE:
            print(f"models-init: {model_id} installed")
        else:
            failed += 1
            print(f"models-init: {model_id} failed: {progress.error or progress.status.value}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
