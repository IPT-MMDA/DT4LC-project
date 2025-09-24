import os
from pathlib import Path
import sys


def run_prithvi(data_files: list[str], input_indices: list[int] | None = None) -> dict:
    prithvi_dir = Path(os.environ.get("PRITHVI_DIR", "dta/dti/coe/resources/prithvi_eo_v1_100m"))
    sys.path.insert(0, str(prithvi_dir))  # so `import inference` works
    import inference  # from the HF repo
    # The HF script expects GeoTIFFs (same location, chronological order) with 6 HLS bands.
    # Many teams just call its main/func; adapt below to whatever API inference.py exposes.
    # If it only provides an argparse main, fall back to subprocess.

    # Example: if inference exposes a function like `run(data_files, input_indices=None)`
    result = inference.run(data_files=data_files, input_indices=input_indices)
    return result  # e.g., feature tensors / recon output (shape depends on their script)
