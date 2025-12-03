from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

import pytest
import rasterio

from dta.config import TEMP_PATH
from dta.dti.coe.orchestrator import orchestrate
from dta.dti.models import get_model_manager
from dta.dti.schemas import Attachment, ChatRequest

# ----- Resolve paths from ModelManager (models downloaded on-demand) ------
MODEL_MANAGER = get_model_manager()
PRITHVI_MODEL_ID = "prithvi-eo-v1-100m"
CFG_JSON = "config.json"
WEIGHTS = "Prithvi_EO_V1_100M.pt"  # or "Prithvi_100M.pt"

# Example files (these would need to be downloaded separately or skipped)
EXAMPLES = [
    "examples/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
    "examples/HLS.L30.T13REN.2018029T172738.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
    "examples/HLS.L30.T13REN.2018061T172724.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif",
]


def test_fullflow_prithvi_example_persist_to_temp() -> None:
    # --- Pre-flight: Check if model is available via ModelManager ---------
    if not MODEL_MANAGER.is_model_available(PRITHVI_MODEL_ID):
        pytest.skip(
            f"Prithvi model not installed. Use the Models page in UI or "
            f'run: python -c "from dta.dti.models import get_model_manager; '
            f"get_model_manager().start_download('{PRITHVI_MODEL_ID}')\""
        )

    # Get model path from ModelManager
    prithvi_path = MODEL_MANAGER.get_model_path(PRITHVI_MODEL_ID)
    if prithvi_path is None:
        pytest.skip("Prithvi model path not found after availability check")

    prithvi_dir = Path(prithvi_path)

    # Check required files
    need = ["inference.py", "prithvi_mae.py", WEIGHTS, CFG_JSON]
    missing = [p for p in need if not (prithvi_dir / p).exists()]
    if missing:
        pytest.skip(f"Prithvi files missing in {prithvi_dir}. Missing: {missing}")

    # Examples are not downloaded by ModelManager (too large), skip if missing
    example_missing = [e for e in EXAMPLES if not (prithvi_dir / e).exists()]
    if example_missing:
        pytest.skip(f"Example files not available. These are not downloaded by default: {example_missing}")

    try:
        import torch  # required at runtime

        _ = torch.__version__
    except Exception:
        pytest.skip("PyTorch not available in this environment.")

    # --- Output location under TEMP_PATH ----------------------------------
    temp_root = Path(TEMP_PATH).resolve()
    run_id = f"prithvi_out_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = temp_root / "tests" / "prithvi_out" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Simulate a user chat (attach TIFF; agent will PNG-preview it) --
    tif0 = prithvi_dir / EXAMPLES[0]
    req = ChatRequest(
        prompt="Generate temporal features with Prithvi 100M for the selected AOI.",
        attachments=[
            Attachment(
                id="att-1",
                filename=tif0.name,
                mime_type="image/tiff",
                path=str(tif0),
                size_bytes=tif0.stat().st_size,
            )
        ],
        metadata={"ui_bbox": [0, 0, 1, 1], "test_mode": True},
    )

    # --- 2) Orchestrate: Context → Planner → DM ---------------------------
    result = orchestrate(req)
    assert result["ok"], f"Plan/DM failed: {result}"
    plan = result["plan"]
    step_ids = [s["uses"] for s in plan["steps"]]
    assert any("models/prithvi_features" in s for s in step_ids), f"Plan missing prithvi step: {step_ids}"

    # sanity: model path exists and has inference.py
    assert (prithvi_dir / "inference.py").exists(), f"inference.py not found in {prithvi_dir}"

    # --- 3) Execute downloaded inference.py directly ----------------------
    entry = prithvi_dir / "inference.py"
    sys.path.insert(0, str(prithvi_dir))  # import sibling prithvi_mae.py
    spec = importlib.util.spec_from_file_location("prithvi_infer", str(entry))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    data_files = [str(prithvi_dir / e) for e in EXAMPLES]

    # Provide envs (mirrors what your executor would do)
    os.environ["PRITHVI_DIR"] = str(prithvi_dir)
    os.environ["PRITHVI_CONFIG"] = str(prithvi_dir / CFG_JSON)
    os.environ["PRITHVI_WEIGHTS"] = str(prithvi_dir / WEIGHTS)

    module.main(
        data_files=data_files,
        config_path=str(prithvi_dir / CFG_JSON),
        checkpoint=str(prithvi_dir / WEIGHTS),
        output_dir=str(out_dir),
        rgb_outputs=True,
        mask_ratio=0.75,
        input_indices=None,
    )

    # --- 4) Validate outputs & write manifest -----------------------------
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
    produced = []
    for name in expected:
        p = out_dir / name
        assert p.exists() and p.stat().st_size > 0, f"Missing/empty output: {p}"
        produced.append(p)

    # Deeper checks with rasterio (shape/CRS/bands consistent)
    metas = []
    for p in produced:
        with rasterio.open(p) as src:
            metas.append(
                {
                    "name": p.name,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "crs": str(src.crs) if src.crs else None,
                    "dtype": src.dtypes[0],
                }
            )
            assert src.width > 0 and src.height > 0
            assert src.count in (3,)  # RGB outputs
    # All outputs should share dimensions
    w0, h0 = metas[0]["width"], metas[0]["height"]
    assert all(m["width"] == w0 and m["height"] == h0 for m in metas)

    # Save plan + metadata manifest
    manifest = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "plan": plan,
        "prithvi_dir": str(prithvi_dir),
        "inputs": data_files,
        "outputs": [str(p) for p in produced],
        "raster_meta": metas,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Keep artifacts in TEMP_PATH for inspection; uncomment to clean up:
    # shutil.rmtree(out_dir, ignore_errors=True)

    # Final assertion: manifest exists and is parseable
    loaded = json.loads((out_dir / "manifest.json").read_text())
    assert loaded["run_id"] == run_id
    assert len(loaded["outputs"]) == len(expected)
