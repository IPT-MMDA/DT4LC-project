"""Job artifact contract (docs/ARCHITECTURE.md "Job artifacts"; #65, #67 bug 2).

Results carry ``{kind, path, url, stats}`` references to GeoTIFF/PNG files,
never pixel arrays or base64 images, so job records, list responses and
SQLite rows stay small.
"""

from __future__ import annotations

from collections.abc import Iterator
import functools
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from dta.dti.coe import planner
from dta.dti.schemas import ContextUnderstanding

_BASE64 = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def _pixel_data(obj: Any, path: str = "") -> list[str]:
    """Return the paths of every array, base64-looking string (> 1 KB) or nested numeric list."""
    found: list[str] = []
    if isinstance(obj, np.ndarray):
        found.append(f"{path}: numpy.ndarray")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += _pixel_data(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        if any(isinstance(v, list | tuple) for v in obj) and all(
            isinstance(x, int | float | type(None)) for v in obj if isinstance(v, list | tuple) for x in v
        ):
            found.append(f"{path}: nested list of numbers")
        elif len(obj) > 256 and all(isinstance(v, int | float) for v in obj):
            found.append(f"{path}: list of {len(obj)} numbers")
        else:
            for i, v in enumerate(obj):
                found += _pixel_data(v, f"{path}[{i}]")
    elif isinstance(obj, str) and len(obj) > 1024 and _BASE64.match(obj):
        found.append(f"{path}: base64-looking string of {len(obj)} chars")
    return found


def _assert_ref(ref: dict[str, Any], kind: str) -> None:
    from urllib.parse import quote

    assert set(ref) == {"kind", "path", "url", "stats"}, ref
    assert ref["kind"] == kind
    assert Path(ref["path"]).is_absolute() and Path(ref["path"]).exists()
    assert ref["url"] == "/v1/download?path=" + quote(ref["path"], safe="")
    if kind == "geotiff":
        assert set(ref["stats"]) == {"min", "max", "mean", "valid_pixels"}
    else:
        assert ref["stats"] == {}


# --- Every algorithm's result ------------------------------------------------


@pytest.mark.parametrize(
    ("module", "raster_keys", "viz_keys"),
    [
        ("ndvi", {"ndvi"}, {"ndvi_map"}),
        ("ndwi", {"ndwi"}, {"ndwi_map"}),
        ("ndsi", {"ndsi", "snow_mask"}, {"ndsi_map"}),
        ("evi", {"evi"}, {"evi_map"}),
        ("snow_classifier", {"snow_mask", "ndsi", "ndvi", "brightness"}, {"snow_classification", "criteria_analysis"}),
        ("lulc_classifier", {"classification"}, {"classification_map", "distribution_chart"}),
    ],
)
def test_single_raster_algorithm_results_hold_refs(
    module: str, raster_keys: set[str], viz_keys: set[str], synthetic_raster_path: str
) -> None:
    import importlib

    result = importlib.import_module(f"dta.dti.algorithms.{module}").run(synthetic_raster_path)

    assert _pixel_data(result) == []
    assert set(result["rasters"]) == raster_keys
    assert set(result["visualizations"]) == viz_keys
    for ref in result["rasters"].values():
        _assert_ref(ref, "geotiff")
    for ref in result["visualizations"].values():
        _assert_ref(ref, "png")


def test_change_detection_result_holds_refs(synthetic_raster_pair: tuple[str, str]) -> None:
    from dta.dti.algorithms import change_detection

    before, after = synthetic_raster_pair
    result = change_detection.run(before, after, "ndvi")

    assert _pixel_data(result) == []
    assert set(result["rasters"]) == {"change", "index_before", "index_after"}
    assert set(result["visualizations"]) == {"change_map", "ndvi_before", "ndvi_after"}
    with rasterio.open(result["rasters"]["change"]["path"]) as src, rasterio.open(after) as ref:
        assert src.crs == ref.crs and src.transform == ref.transform
        assert src.dtypes[0] == "float32" and np.isnan(src.nodata)
        assert np.nanmean(src.read(1)) == pytest.approx(result["statistics"]["mean_change"], abs=1e-5)


def test_spectral_index_run_change_holds_refs(synthetic_raster_pair: tuple[str, str]) -> None:
    from dta.dti.algorithms import ndvi, spectral_index

    before, after = synthetic_raster_pair
    result = spectral_index.run_change(before, after, ndvi._config())

    assert _pixel_data(result) == []
    assert set(result["rasters"]) == {"change"}
    _assert_ref(result["rasters"]["change"], "geotiff")
    assert result["rasters"]["change"]["stats"]["mean"] == pytest.approx(result["statistics"]["mean_change"], abs=1e-5)


def test_ndvi_change_reads_rasters_from_files(synthetic_raster_pair: tuple[str, str]) -> None:
    from dta.dti.algorithms import ndvi

    before, after = (ndvi.calculate_ndvi(p) for p in synthetic_raster_pair)
    result = ndvi.ndvi_change(before, after)

    assert _pixel_data(result) == []
    assert result["statistics"]["mean_change"] < 0  # vegetation loss in the fixture pair
    _assert_ref(result["rasters"]["change"], "geotiff")


def test_model_step_result_holds_png_refs_next_to_tiffs(tmp_path: Path) -> None:
    from dta.dti.executor import PipelineExecutor

    tif = tmp_path / "predicted_rgb.tiff"
    with rasterio.open(tif, "w", driver="GTiff", width=16, height=16, count=3, dtype="uint8") as dst:
        dst.write(np.full((3, 16, 16), 128, dtype=np.uint8))

    result = PipelineExecutor()._collect_model_outputs("mock-model", str(tmp_path), {"RasterPath": "/x.tif"})

    assert _pixel_data(result) == []
    assert result["output_files"] == [str(tif)]
    ref = result["visualizations"]["predicted_rgb"]
    _assert_ref(ref, "png")
    assert Path(ref["path"]) == tif.with_suffix(".png")


def test_executor_writes_step_files_under_its_output_dir(synthetic_raster_path: str, tmp_path: Path) -> None:
    from dta.dti.executor import PipelineExecutor
    from dta.dti.schemas import ExecutionPlan, PlanStep

    plan = ExecutionPlan(
        flow="ndvi",
        steps=[
            PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
            PlanStep(uses="algorithms/ndvi"),
        ],
        outputs=["NDVIMap"],
    )
    out = PipelineExecutor().execute(plan, output_dir=tmp_path / "job1")["artifacts"]["NDVIMap"]

    assert out["rasters"]["ndvi"]["path"] == str(tmp_path / "job1" / "1_NDVIMap_ndvi.tif")
    assert out["visualizations"]["ndvi_map"]["path"] == str(tmp_path / "job1" / "1_NDVIMap_ndvi_map.png")


def test_executor_outside_job_queue_writes_to_run_dir(synthetic_raster_path: str) -> None:
    from dta.dti import artifact_files as artifacts
    from dta.dti.executor import PipelineExecutor
    from dta.dti.schemas import ExecutionPlan, PlanStep

    plan = ExecutionPlan(
        flow="ndvi",
        steps=[
            PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
            PlanStep(uses="algorithms/ndvi"),
        ],
        outputs=["NDVIMap"],
    )
    path = Path(PipelineExecutor().execute(plan)["artifacts"]["NDVIMap"]["rasters"]["ndvi"]["path"])

    assert path.parent.parent == artifacts.OUTPUTS_PATH
    assert re.fullmatch(r"run-[0-9a-f]{32}", path.parent.name)


# --- Job API sizes (1000x1000 uint16 NDVI job) --------------------------------


@pytest.fixture
def stub_llm() -> Iterator[MagicMock]:
    """No LLM on the job path: fixed NDVI context, template planner, mock summary router."""
    ctx = ContextUnderstanding(
        goal="calculate ndvi",
        desired_outputs=["NDVIMap"],
        required_inputs=["RasterPath"],
        hints={"keywords": ["ndvi"]},
    )
    router = MagicMock()
    router.generate.return_value = MagicMock(text="Stub summary.", provider="mock")
    with (
        patch("dta.dti.coe.orchestrator.analyze", return_value=ctx),
        patch("dta.dti.coe.orchestrator.plan", functools.partial(planner.plan, use_llm=False)),
        patch("dta.dti.executor.get_llm_router", return_value=router),
    ):
        yield router


@pytest.fixture
def raster_1000(tmp_path: Path) -> str:
    path = tmp_path / "u16_1000.tif"
    rng = np.random.default_rng(0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1000,
        height=1000,
        count=5,
        dtype="uint16",
        crs="EPSG:32636",
        transform=from_origin(500000, 5000000, 30, 30),
    ) as dst:
        dst.write(rng.integers(100, 6000, size=(5, 1000, 1000), dtype=np.uint16))
    return str(path)


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Path]]:
    """TestClient whose job queue persists to a temp SQLite DB."""
    from server.app import app
    import server.jobs as jobs_mod

    db = tmp_path / "jobs.db"
    monkeypatch.setattr(jobs_mod, "_job_queue", jobs_mod.JobQueue(max_workers=1, store=jobs_mod.SQLiteJobStore(db)))
    with TestClient(app) as client:
        yield client, db


def test_ndvi_job_record_list_and_row_are_small(
    api: tuple[TestClient, Path], raster_1000: str, stub_llm: MagicMock
) -> None:
    from dta.dti import artifact_files as artifacts
    import server.jobs as jobs_mod

    client, db = api
    att = {"id": "a1", "filename": "u16_1000.tif", "path": raster_1000, "mime_type": "image/tiff"}
    job_id = client.post("/v1/jobs", json={"prompt": "calculate ndvi", "attachments": [att]}).json()["id"]
    for _ in range(240):
        if client.get(f"/v1/jobs/{job_id}").json()["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.25)

    r = client.get(f"/v1/jobs/{job_id}")
    job = r.json()
    assert job["status"] == "completed", job["error"]
    assert len(r.content) < 1_000_000, f"GET /v1/jobs/{{id}} is {len(r.content)} bytes"
    assert _pixel_data(job) == []

    # 20 jobs in the list: the real one plus 19 copies of its record.
    store = jobs_mod._job_queue._store  # type: ignore[union-attr]
    real = store.get(job_id)
    for i in range(19):
        real.id = f"copy{i:04d}"
        store.save(real)
    r = client.get("/v1/jobs?limit=20")
    assert len(r.json()["jobs"]) == 20
    assert len(r.content) < 200_000, f"GET /v1/jobs?limit=20 is {len(r.content)} bytes"
    assert all("result" not in j for j in r.json()["jobs"])

    con = sqlite3.connect(db)
    for (row_result,) in con.execute("SELECT result FROM jobs"):
        assert len(row_result) < 1_000_000
        assert _pixel_data(json.loads(row_result)) == []
    con.close()

    ndvi_map = job["result"]["execution"]["artifacts"]["NDVIMap"]
    assert "ndvi_array" not in ndvi_map
    tif_ref = ndvi_map["rasters"]["ndvi"]
    png_ref = ndvi_map["visualizations"]["ndvi_map"]
    assert Path(tif_ref["path"]).parent == artifacts.OUTPUTS_PATH / job_id
    assert tif_ref["stats"]["valid_pixels"] == 1_000_000
    assert -1.0 <= tif_ref["stats"]["min"] <= tif_ref["stats"]["max"] <= 1.0

    tif = client.get(tif_ref["url"])
    assert tif.status_code == 200
    with MemoryFile(tif.content) as mem, mem.open() as src:
        assert src.crs.to_string() == "EPSG:32636"
        assert (src.count, src.height, src.width) == (1, 1000, 1000)
        assert src.dtypes[0] == "float32" and np.isnan(src.nodata)
    png = client.get(png_ref["url"])
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
