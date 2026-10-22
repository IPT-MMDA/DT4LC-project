"""Regression guard: ``_run_python`` must support both load routes.

The executor accepts two entrypoint forms in ``runner.entrypoint``:

* **Dotted module path** (``dta.dti.algorithms.ndvi``) — bundled in-package
  code; loaded via ``importlib.import_module`` so Python's normal module cache
  applies and no ``sys.path`` mutation is needed.
* **Filesystem path** (``dta/.../foo.py``, ``${MODEL_PATH}/inference.py``) —
  third-party / runtime-resolved plugins; loaded via ``spec_from_file_location``.

Detection: ``str.endswith(".py")`` or contains ``/`` / ``\\`` ⇒ filesystem;
otherwise module path.

The filesystem branch is reserved for plugin loading; this test exercises it
synthetically so it doesn't bitrot before the third-party plugin loader lands.
"""

from __future__ import annotations

import logging
from pathlib import Path
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from dta.dti.executor import ExecutionError, PipelineExecutor
from dta.dti.schemas import (
    ExecutionPlan,
    PlanStep,
    PreprocessorRef,
    Registry,
    RegistryItem,
    Runner,
)


class TestRegistryUsesBothLoadRoutes:
    """Pin the registry's intended split so a future bulk edit doesn't silently
    regress every entrypoint to one form.
    """

    def test_in_package_algorithms_use_module_path(self) -> None:
        from dta.dti.registry import get_item, load_registry

        reg = load_registry()
        for item_id in (
            "algorithms/ndvi",
            "algorithms/evi",
            "algorithms/ndwi",
            "algorithms/ndsi",
            "algorithms/snow-classifier",
            "algorithms/statistics",
            "algorithms/change-detection",
            "algorithms/lulc-classifier",
            "preprocessors/scale-to-hls",
        ):
            item = get_item(reg, item_id)
            assert item.runner is not None and item.runner.entrypoint is not None
            ep = item.runner.entrypoint
            assert not ep.endswith(".py") and "/" not in ep, (
                f"{item_id} entrypoint {ep!r} looks like a filesystem path; "
                "in-package code should use a dotted module path so importlib's "
                "module cache applies and sys.path stays untouched"
            )

    def test_third_party_models_keep_filesystem_paths(self) -> None:
        """Prithvi (``${MODEL_PATH}/...``) and Delineate-Anything (vendored
        third-party) intentionally stay on the filesystem branch.
        """
        from dta.dti.registry import get_item, load_registry

        reg = load_registry()
        for item_id in ("models/prithvi-reconstruction", "models/delineate-anything"):
            item = get_item(reg, item_id)
            assert item.runner is not None and item.runner.entrypoint is not None
            ep = item.runner.entrypoint
            assert ep.endswith(".py"), (
                f"{item_id} entrypoint {ep!r} should remain a filesystem path; "
                "the dynamic-load route is what supports third-party plugins"
            )


class TestFilesystemPathRoute:
    """The dynamic ``spec_from_file_location`` route must keep working — it's
    the abstraction we'll rely on for third-party plugin loading.
    """

    def test_synthetic_filesystem_entrypoint_executes(self, tmp_path: Path) -> None:
        plugin = tmp_path / "fake_plugin.py"
        plugin.write_text(
            textwrap.dedent(
                """
                def run(value):
                    return int(value) * 2
                """
            )
        )

        registry = Registry(
            version="1.0",
            types=["Value", "Doubled"],
            instances=[
                RegistryItem(
                    id="plugins/doubler",
                    kind="algorithm",
                    inputs=["Value"],
                    outputs=["Doubled"],
                    runner=Runner(
                        type="python",
                        entrypoint=str(plugin),
                        args_map={"value": "${Value}"},
                        function="run",
                    ),
                ),
            ],
        )

        plan = ExecutionPlan(
            flow="filesystem-route-test",
            steps=[PlanStep(uses="plugins/doubler", binds={"Value": "21"})],
            outputs=["Doubled"],
        )

        result = PipelineExecutor(registry=registry).execute(plan)
        assert result["artifacts"]["Doubled"] == 42


class TestModulePathRoute:
    """The ``importlib.import_module`` route is what the in-package algorithms
    use. Spot-check the cache property that motivates this route.
    """

    def test_module_import_is_cached(self) -> None:
        """A repeat ``import_module`` returns the same module object — confirms
        we're getting cache benefit (vs ``spec_from_file_location`` which
        creates a fresh module per call).
        """
        import importlib

        a = importlib.import_module("dta.dti.algorithms.ndvi")
        b = importlib.import_module("dta.dti.algorithms.ndvi")
        assert a is b


def _write_raster(path: Path, count: int) -> str:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(30.0, 47.0, 0.0001, 0.0001),
    ) as dst:
        dst.write(np.full((count, 8, 8), 0.1, dtype=np.float32))
    return str(path)


class TestPreprocessorRoute:
    """Preprocessors load through the same dotted/filesystem loader (#60).

    ``preprocessors/scale-to-hls`` is a dotted module path; it used to be
    resolved as a filesystem path only, logged "entrypoint not found" and was
    skipped, so Prithvi got unscaled input.
    """

    @staticmethod
    def _prithvi_plan(raster: str) -> ExecutionPlan:
        return ExecutionPlan(
            flow="prithvi-reconstruction",
            steps=[
                PlanStep(uses="input/file", binds={"RasterPath": raster}),
                PlanStep(uses="models/prithvi-reconstruction", binds={}),
            ],
            outputs=["Reconstruction"],
        )

    @staticmethod
    def _mock_prithvi(model_dir: Path) -> MagicMock:
        """Stand-in for the downloaded Prithvi model: inference.py echoes its input."""
        model_dir.mkdir()
        (model_dir / "inference.py").write_text(
            "def main(data_files, **kwargs):\n    return {'data_files': data_files}\n"
        )
        manager = MagicMock()
        manager.is_model_available.return_value = True
        manager.get_model_path.return_value = model_dir
        return manager

    def test_dotted_preprocessor_runs_before_model(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        raster = _write_raster(tmp_path / "six.tif", 6)
        manager = self._mock_prithvi(tmp_path / "model")

        with (
            patch("dta.dti.models.get_model_manager", return_value=manager),
            caplog.at_level(logging.INFO, logger="dta.dti.executor"),
        ):
            result = PipelineExecutor().execute(self._prithvi_plan(raster))

        (model_input,) = result["artifacts"]["Reconstruction"]["data_files"]
        assert model_input != raster
        assert Path(model_input).name == "hls_scaled_input.tif"
        assert "Preprocessor preprocessors/scale-to-hls applied to RasterPath" in caplog.text
        assert "not found" not in caplog.text

    def test_wrong_band_count_fails_before_model(self, tmp_path: Path) -> None:
        raster = _write_raster(tmp_path / "seven.tif", 7)
        manager = self._mock_prithvi(tmp_path / "model")

        with (
            patch("dta.dti.models.get_model_manager", return_value=manager),
            pytest.raises(ExecutionError, match="expected 6 bands .*, got 7") as excinfo,
        ):
            PipelineExecutor().execute(self._prithvi_plan(raster))
        assert isinstance(excinfo.value.__cause__, ValueError)

    @pytest.mark.parametrize(
        ("entrypoint", "error"),
        [
            ("dta.dti.preprocessors.does_not_exist", "Failed to import module"),
            ("dta/dti/preprocessors/does_not_exist.py", "Entrypoint not found"),
        ],
    )
    def test_unloadable_preprocessor_is_an_error(self, tmp_path: Path, entrypoint: str, error: str) -> None:
        registry = Registry(
            version="1.0",
            types=["RasterPath"],
            instances=[
                RegistryItem(
                    id="preprocessors/broken",
                    kind="preprocessor",
                    runner=Runner(type="python", entrypoint=entrypoint, function="run"),
                ),
                RegistryItem(
                    id="models/needs-preprocessing",
                    kind="model",
                    inputs=["RasterPath"],
                    preprocessors=[PreprocessorRef(id="preprocessors/broken", apply_to="RasterPath")],
                    runner=Runner(type="python", entrypoint="dta.dti.algorithms.ndvi"),
                ),
            ],
        )
        executor = PipelineExecutor(registry=registry)
        item = registry.instances[1]

        with pytest.raises(ExecutionError, match=error):
            executor._apply_preprocessors(item, {"RasterPath": "in.tif"}, str(tmp_path))

    def test_preprocessor_missing_from_registry_is_an_error(self, tmp_path: Path) -> None:
        # A typo in a ``preprocessors:`` id fails the step (get_item raises
        # KeyError); it must not run the model on unscaled input.
        from dta.dti.registry import get_item, load_registry

        registry = load_registry()
        get_item(registry, "models/prithvi-reconstruction").preprocessors[0].id = "preprocessors/typo"
        raster = _write_raster(tmp_path / "six.tif", 6)
        manager = self._mock_prithvi(tmp_path / "model")

        with (
            patch("dta.dti.models.get_model_manager", return_value=manager),
            pytest.raises(ExecutionError, match="Registry item not found: preprocessors/typo"),
        ):
            PipelineExecutor(registry=registry).execute(self._prithvi_plan(raster))
