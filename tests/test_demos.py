"""Demo-flow regression tests (Layer A).

Pins the algorithm + executor + registry pipelines exercised by the team's
demo scenario. These tests bypass the LLM (no key required) and instead build
``ExecutionPlan`` objects by hand, mirroring what the planner would emit for
each demo prompt. The Layer-B counterpart in ``test_demos_llm.py`` covers the
full intent → context → plan → execute path.

Each test class corresponds to one step of the rehearsed demo flow. Tests that
need real Kahovka samples skip when the directory is empty (the default in
this repo); the synthetic-raster fixtures from ``conftest.py`` provide CI-safe
coverage that exercises the same pipeline shape on Landsat-8-like data.
"""

from pathlib import Path

import pytest

from dta.config import ROOT_DIR
from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ExecutionPlan, PlanStep

KAHOVKA_DIR = ROOT_DIR / "resources/kahovka_data"


def _kahovka_pair() -> tuple[Path, Path] | None:
    """Return (before, after) Kahovka GeoTIFFs if vendored, else None."""
    if not KAHOVKA_DIR.exists():
        return None
    tifs = sorted(list(KAHOVKA_DIR.glob("*.tif")) + list(KAHOVKA_DIR.glob("*.tiff")))
    if len(tifs) < 2:
        return None
    return tifs[0], tifs[1]


class TestKahovkaVegetationChange:
    """Demo 1 step 3: 'analyze vegetation changes' on Kahovka before/after.

    Pins the NDVI change-detection pipeline (input/file-before +
    input/file-after + algorithms/change-detection with IndexType=ndvi).
    Future spectral-index refactors must keep this output shape.
    """

    def test_synthetic_pair_runs_change_detection(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="vegetation-change",
            steps=[
                PlanStep(uses="input/file-before", binds={"RasterPathBefore": before}),
                PlanStep(uses="input/file-after", binds={"RasterPathAfter": after}),
                PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndvi"}),
            ],
            outputs=["ChangeMap"],
        )

        result = executor.execute(plan)

        assert "ChangeMap" in result["artifacts"]
        change_map = result["artifacts"]["ChangeMap"]
        assert change_map is not None
        # The pipeline returns a dict with statistics + visualizations.
        assert isinstance(change_map, dict)
        assert "statistics" in change_map

    def test_kahovka_real_data_runs(self) -> None:
        """Same pipeline on real Kahovka samples when vendored locally."""
        pair = _kahovka_pair()
        if pair is None:
            pytest.skip("Kahovka samples not present; run on a machine with resources/kahovka_data/")
        before, after = pair

        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow="vegetation-change",
            steps=[
                PlanStep(uses="input/file-before", binds={"RasterPathBefore": str(before)}),
                PlanStep(uses="input/file-after", binds={"RasterPathAfter": str(after)}),
                PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndvi"}),
            ],
            outputs=["ChangeMap"],
        )
        result = executor.execute(plan)
        assert "ChangeMap" in result["artifacts"]


class TestKahovkaLulcClassification:
    """Demo 1 steps 4-5: 'Analyze land-cover classes for this image'.

    Pins the LULC pipeline (input/file + algorithms/lulc-classifier).
    """

    def test_synthetic_runs_lulc(self, synthetic_raster_path: str) -> None:
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="lulc",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
                PlanStep(uses="algorithms/lulc-classifier", binds={}),
            ],
            outputs=["LULCMap"],
        )

        result = executor.execute(plan)

        assert "LULCMap" in result["artifacts"]
        lulc = result["artifacts"]["LULCMap"]
        assert lulc is not None


class TestGlacierIceCoverChange:
    """Demo 2.2: 'Analyze ice cover changes' on glacier before/after.

    Pins the NDSI change-detection pipeline. The user does NOT type 'NDSI' —
    the planner must map 'ice cover' to NDSI. This test pins the algorithmic
    half (the planner mapping is pinned in test_demos_llm.py).
    """

    def test_synthetic_pair_runs_ndsi_change(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        executor = PipelineExecutor()

        plan = ExecutionPlan(
            flow="ice-change",
            steps=[
                PlanStep(uses="input/file-before", binds={"RasterPathBefore": before}),
                PlanStep(uses="input/file-after", binds={"RasterPathAfter": after}),
                PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndsi"}),
            ],
            outputs=["ChangeMap"],
        )
        result = executor.execute(plan)
        assert "ChangeMap" in result["artifacts"]
        change_map = result["artifacts"]["ChangeMap"]
        assert isinstance(change_map, dict)
        assert "statistics" in change_map


class TestForestNdviChangeExplicit:
    """Demo 2.1: 'Analyze NDVI changes for these images' (Svyaty Hory forest).

    Same pipeline shape as the Kahovka vegetation-change demo — the user just
    phrases it explicitly. Layer-B test pins that both phrasings reach this
    same plan; this test pins the plan's algorithmic outcome.
    """

    def test_synthetic_pair_runs(self, synthetic_raster_pair: tuple[str, str]) -> None:
        before, after = synthetic_raster_pair
        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow="ndvi-change-explicit",
            steps=[
                PlanStep(uses="input/file-before", binds={"RasterPathBefore": before}),
                PlanStep(uses="input/file-after", binds={"RasterPathAfter": after}),
                PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndvi"}),
            ],
            outputs=["ChangeMap"],
        )
        result = executor.execute(plan)
        assert "ChangeMap" in result["artifacts"]


class TestSpectralIndicesIndividually:
    """Pins per-index calculation paths.

    A future refactor may collapse ndvi.py / ndwi.py / ndsi.py / evi.py into
    one parameterized module — these tests must keep working with byte-
    identical output keys across that change.
    """

    @pytest.mark.parametrize(
        ("step_id", "output_type"),
        [
            ("algorithms/ndvi", "NDVIMap"),
            ("algorithms/ndwi", "NDWIMap"),
            ("algorithms/ndsi", "NDSIMap"),
            ("algorithms/evi", "EVIMap"),
        ],
    )
    def test_index_runs_on_synthetic(self, step_id: str, output_type: str, synthetic_raster_path: str) -> None:
        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow=f"single-{step_id}",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
                PlanStep(uses=step_id, binds={}),
            ],
            outputs=[output_type],
        )
        result = executor.execute(plan)
        assert output_type in result["artifacts"]
        assert result["artifacts"][output_type] is not None


class TestStatistics:
    """Pins the algorithms/statistics pipeline."""

    def test_synthetic_runs_statistics(self, synthetic_raster_path: str) -> None:
        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow="stats",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
                PlanStep(uses="algorithms/statistics", binds={}),
            ],
            outputs=["Statistics"],
        )
        result = executor.execute(plan)
        assert "Statistics" in result["artifacts"]
        stats = result["artifacts"]["Statistics"]
        assert stats is not None


class TestSentinel2NdwiFetch:
    """Demo 1 step 1: 'Get data' against Sentinel-2 / GEE.

    Hits real Google Earth Engine, requires GEE credentials, and is gated
    behind ``@pytest.mark.network``. CI never runs this — engineers do it
    manually before refactor merges that touch ``data_sources/gee_*.py``.
    """

    @pytest.mark.network
    @pytest.mark.slow
    def test_small_bbox_fetch(self) -> None:
        from dta.dti.data_sources.gee_sentinel2 import fetch_sentinel2_indices

        bbox = [30.27, 46.78, 30.30, 46.80]  # tiny ~3km box near Kahovka
        result = fetch_sentinel2_indices(
            bbox=bbox,
            start_date="2023-06-01",
            end_date="2023-06-15",
            index_type="ndwi",
        )
        assert isinstance(result, dict)
        assert result.get("ok") is True
        assert "tile_url" in result


class TestPrithviReconstruction:
    """Demo 2.3 (optional): 'Reconstruct this image' via Prithvi MAE.

    Heavy ML; gated behind gpu+slow markers and a runtime check that the
    Prithvi model is actually downloaded. CI never runs this.
    """

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_prithvi_runs_when_model_installed(self, synthetic_raster_path: str) -> None:
        from dta.dti.models import get_model_manager

        manager = get_model_manager()
        if not manager.is_model_available("prithvi-eo-v1-100m"):
            pytest.skip("prithvi-eo-v1-100m not installed; run model download first")

        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow="prithvi-reconstruction",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
                PlanStep(uses="models/prithvi-reconstruction", binds={}),
            ],
            outputs=["Reconstruction"],
        )
        result = executor.execute(plan)
        assert "Reconstruction" in result["artifacts"]


class TestDelineateAnything:
    """Field-boundary detection (not in main demo but available capability)."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_delineate_runs_when_model_installed(self, synthetic_raster_path: str) -> None:
        from dta.dti.models import get_model_manager

        manager = get_model_manager()
        if not manager.is_model_available("delineate-anything-small"):
            pytest.skip("delineate-anything-small not installed; run model download first")

        executor = PipelineExecutor()
        plan = ExecutionPlan(
            flow="delineate",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": synthetic_raster_path}),
                PlanStep(uses="models/delineate-anything", binds={}),
            ],
            outputs=["FieldBoundaries"],
        )
        result = executor.execute(plan)
        assert "FieldBoundaries" in result["artifacts"]
