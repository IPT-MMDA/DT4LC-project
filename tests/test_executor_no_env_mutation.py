"""Regression guard: the executor must not mutate process-global ``os.environ``.

Concurrent jobs run through one ``PipelineExecutor`` per request via the
``JobQueue``'s thread pool. If ``_run_python`` mutates ``os.environ`` (or
``sys.path`` in a way that varies by item), two threads can clobber each
other's runtime config. Algorithms should receive everything they need
through ``runner.args_map`` kwargs instead.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import ExecutionPlan, PlanStep


def _ndvi_plan(raster_path: str) -> ExecutionPlan:
    return ExecutionPlan(
        flow="env-isolation-test",
        steps=[
            PlanStep(uses="input/file", binds={"RasterPath": raster_path}),
            PlanStep(uses="algorithms/ndvi", binds={}),
        ],
        outputs=["NDVIMap"],
    )


class TestExecutorDoesNotMutateEnviron:
    def test_environ_unchanged_after_single_pipeline(self, synthetic_raster_path: str) -> None:
        """A successful pipeline run must leave ``os.environ`` byte-for-byte intact."""
        before = dict(os.environ)
        executor = PipelineExecutor()
        executor.execute(_ndvi_plan(synthetic_raster_path))
        after = dict(os.environ)
        assert before == after, (
            "executor mutated os.environ during _run_python — concurrent jobs would clobber each other"
        )

    def test_environ_unchanged_under_concurrent_execution(
        self, synthetic_raster_path: str, synthetic_raster_pair: tuple[str, str]
    ) -> None:
        """Two pipelines running concurrently must leave ``os.environ`` intact."""
        before = dict(os.environ)
        before_path, after_path = synthetic_raster_pair

        def _run_ndvi(path: str) -> None:
            PipelineExecutor().execute(_ndvi_plan(path))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(_run_ndvi, synthetic_raster_path),
                pool.submit(_run_ndvi, before_path),
                pool.submit(_run_ndvi, after_path),
            ]
            for f in futures:
                f.result()  # surface any exception

        after = dict(os.environ)
        assert before == after, "executor mutated os.environ when 3 threads ran NDVI concurrently"


class TestDelineateConfigViaArgsMap:
    """Delineate-Anything's model variant ('small' / 'large') must travel through
    ``runner.args_map``, not ``runner.env`` — otherwise concurrent delineate
    invocations with different sizes would race on a global env var.
    """

    def test_registry_passes_delineate_model_via_args_map(self) -> None:
        from dta.dti.registry import get_item, load_registry

        item = get_item(load_registry(), "models/delineate-anything")
        assert item.runner is not None
        assert not item.runner.env, (
            f"delineate runner.env should be empty (use args_map instead); got {item.runner.env}"
        )
        args_map = item.runner.args_map or {}
        assert "delineate_model" in args_map, (
            "delineate runner.args_map should include 'delineate_model'; "
            "the inference.py run() signature reads it as a kwarg"
        )

    def test_inference_run_accepts_delineate_model_kwarg(self) -> None:
        """The third-party inference.py contract: ``run(RasterPath, delineate_model="...")``."""
        import importlib.util
        import inspect

        from dta.config import ROOT_DIR

        path = Path(ROOT_DIR) / "dta/dti/models/third_party/delineate_anything/inference.py"
        spec = importlib.util.spec_from_file_location("delineate_inference_probe", str(path))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sig = inspect.signature(module.run)
        assert "delineate_model" in sig.parameters, (
            f"run() should accept delineate_model kwarg; got {list(sig.parameters)}"
        )
