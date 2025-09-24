from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
TARGET = Path("dta/dti/models/third_party/prithvi_eo_v1_100m")

TARGET.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=REPO,
    local_dir=str(TARGET),
    local_dir_use_symlinks=False,  # real files, no cache symlinks
    resume_download=True,
    allow_patterns=[
        "Prithvi_EO_V1_100M.pt",
        "Prithvi_100M.pt",
        "inference.py",
        "prithvi_mae.py",
        "config.json",
        "config.yaml",
        "examples/*",
    ],
)
print(f"Prithvi files ready at: {TARGET}")
