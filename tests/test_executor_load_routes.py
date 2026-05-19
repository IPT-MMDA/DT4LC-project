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

from pathlib import Path
import textwrap

from dta.dti.executor import PipelineExecutor
from dta.dti.schemas import (
    ExecutionPlan,
    PlanStep,
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
