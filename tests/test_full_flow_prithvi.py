import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pytest

from dta.dti.coe.orchestrator import orchestrate
from dta.dti.registry import get_item, load_registry
from dta.dti.schemas import Attachment, ChatRequest

REGISTRY = load_registry()

PRITHVI_DIR = Path(get_item(REGISTRY, "models/prithvi_features").runner.env["PRITHVI_DIR"]).resolve()
EXAMPLES = [
    "examples/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
    "examples/HLS.L30.T13REN.2018029T172738.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
    "examples/HLS.L30.T13REN.2018061T172724.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
]
WEIGHTS = "Prithvi_EO_V1_100M.pt"  # or "Prithvi_100M.pt"
CFG_JSON = "config.json"


def test_fullflow_prithvi_example(tmp_path: Path) -> None:
    """
    Simulates a user chat + file attachment, runs Context->Planner->DM,
    then executes Prithvi inference.py on the downloaded example HLS GeoTIFFs.
    Produces RGB outputs and asserts that expected files are created.
    """

    # --- Pre-flight checks / skip if resources missing --------------------
    if not PRITHVI_DIR.exists():
        pytest.skip(f"Prithvi bundle not found at {PRITHVI_DIR}. Run scripts/download_prithvi.py first.")

    for rel in EXAMPLES + [WEIGHTS, CFG_JSON, "inference.py"]:
        if not (PRITHVI_DIR / rel).exists():
            pytest.skip(f"Missing required Prithvi file: {rel}")

    # Torch is required by inference.py
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("PyTorch not available in this environment.")

    # --- 1) Simulate a user chat entry with an 'attachment' ---------------
    # We only attach the first tif; the runner will still use all three examples.
    first_tif = (PRITHVI_DIR / EXAMPLES[0]).as_posix()
    req = ChatRequest(
        prompt="Generate features using the Prithvi 100M model for my area (temporal HLS stack).",
        attachments=[
            Attachment(
                id="att-1",
                filename=Path(first_tif).name,
                mime_type="image/tiff",
                path=first_tif,
                size_bytes=os.path.getsize(first_tif),
            )
        ],
        metadata={"test_mode": True},
    )

    # --- 2) Orchestrate: Context → Planner → Decision Maker ---------------
    result = orchestrate(req)
    assert result["ok"], f"Planner/DM failed: {result}"

    plan = result["plan"]
    step_ids = [s["uses"] for s in plan["steps"]]
    # Ensure our prithvi step is selected
    assert any("models/prithvi_features" in s for s in step_ids), f"Plan missing prithvi step: {step_ids}"

    # --- 3) Execute the Prithvi step (import the downloaded entrypoint) ---
    # We directly import the downloaded inference.py from PRITHVI_DIR to avoid path headaches.
    entrypoint = PRITHVI_DIR / "inference.py"
    sys.path.insert(0, str(PRITHVI_DIR))  # so prithvi_mae.py resolves
    spec = importlib.util.spec_from_file_location("prithvi_infer", str(entrypoint))
    m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(m)

    # Build absolute paths to the three sample GeoTIFFs
    data_files = [str(PRITHVI_DIR / p) for p in EXAMPLES]
    # Output directory
    out_dir = tmp_path / "prithvi_out"

    # Set env hints (optional; your registry runner uses these too)
    os.environ["PRITHVI_DIR"] = str(PRITHVI_DIR)
    os.environ["PRITHVI_WEIGHTS"] = str(PRITHVI_DIR / WEIGHTS)
    os.environ["PRITHVI_CONFIG"] = str(PRITHVI_DIR / CFG_JSON)

    # Call the entrypoint's main() with explicit args (mirrors CLI).
    # Defaults & args are documented in the model card and the script:
    # --data_files ... --config_path config.json --checkpoint weights --output_dir ... --rgb_outputs
    # (Three HLS GeoTIFFs, chronological order; HLS bands Blue, Green, Red, Narrow NIR, SWIR1, SWIR2.)
    # Sources: model card "Inference and demo" and inference.py defaults.  # cites below
    m.main(
        data_files=data_files,
        config_path=str(PRITHVI_DIR / CFG_JSON),
        checkpoint=str(PRITHVI_DIR / WEIGHTS),
        output_dir=str(out_dir),
        rgb_outputs=True,
        mask_ratio=0.75,
        input_indices=None,  # examples already have the 6 HLS bands in the expected order
    )

    # --- 4) Validate that expected outputs were written -------------------
    # If rgb_outputs=True, inference.py writes per-timestep:
    # original_rgb_t{t}.tiff, predicted_rgb_t{t}.tiff, masked_rgb_t{t}.tiff
    expected = [
        "original_rgb_t0.tiff",
        "predicted_rgb_t0.tiff",
        "masked_rgb_t0.tiff",
        "original_rgb_t1.tiff",
        "predicted_rgb_t1.tiff",
        "masked_rgb_t1.tiff",
        "original_rgb_t2.tiff",
        "predicted_rgb_t2.tiff",
        "masked_rgb_t2.tiff",
    ]
    for name in expected:
        p = out_dir / name
        assert p.exists() and p.stat().st_size > 0, f"Missing/empty output: {p}"

    # Optional: leave artifacts for inspection if test fails; else cleanup.
    try:
        shutil.rmtree(out_dir)
    except Exception:
        pass
