"""Download the Prithvi EO v1 (100M) weights through the project's ModelManager.

Files land in MODELS_PATH/prithvi-eo-v1-100m (resources/.cache/models/ by
default; DT4LC_MODEL_CACHE overrides it), which is where the app reads them.
"""

import sys
import time

from dta.dti.models.model_manager import ModelStatus, get_model_manager

MODEL_ID = "prithvi-eo-v1-100m"


def main(poll_seconds: float = 2.0) -> int:
    """Download the model if needed, wait for the download, and report the final status."""
    manager = get_model_manager()
    path = manager.get_model_path(MODEL_ID)

    if manager.is_model_available(MODEL_ID):
        print(f"{MODEL_ID}: available at {path} (already downloaded)")
        return 0

    print(f"Downloading {MODEL_ID} to {path} ...")
    progress = manager.start_download(MODEL_ID)
    last_percent = None
    while progress.status == ModelStatus.DOWNLOADING:
        percent = progress.to_dict()["percent"]
        if percent != last_percent:
            print(f"  {percent}%")
            last_percent = percent
        time.sleep(poll_seconds)

    status = manager.get_model_status(MODEL_ID)
    if status == ModelStatus.AVAILABLE:
        print(f"{MODEL_ID}: available at {path}")
        return 0
    print(f"{MODEL_ID}: {status.value} ({progress.error or 'download did not complete'})", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
