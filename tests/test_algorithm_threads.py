"""Algorithm PNG renderers are safe on concurrent job worker threads (v0.5.1 item 7).

pyplot keeps one global "current figure"; when renders ran on several threads,
``plt.colorbar``/``plt.savefig`` could act on another thread's figure, silently
writing the wrong PNG. Each render now owns its ``Figure``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from PIL import Image
import pytest

from dta.dti import artifact_files
from dta.dti.algorithms import change_detection, spectral_index
from dta.dti.registry import get_item, load_registry

INDICES = ("evi", "ndvi", "ndwi", "ndsi")
TASKS = [("si", i) for i in INDICES] + [("cd", i) for i in INDICES]
ALGORITHMS_DIR = Path(__file__).resolve().parent.parent / "dta" / "dti" / "algorithms"


def _render(kind: str, index: str, before: str, after: str, outdir: Path) -> dict[str, str]:
    """Run one task and return its PNG paths by visualization key."""
    with artifact_files.step_outputs(outdir, f"{kind}_{index}"):
        if kind == "si":
            config = get_item(load_registry(), f"algorithms/{index}").config or {}
            result = spectral_index.run(before, config)
        else:
            result = change_detection.run(before, after, index)
    return {key: ref["path"] for key, ref in result["visualizations"].items()}


def _pixels(path: str) -> tuple[str, tuple[int, int], bytes]:
    with Image.open(path) as img:
        img.load()
        return img.mode, img.size, img.tobytes()


def _check_concurrent_renders(
    pair: tuple[str, str], tmp_path: Path, recwarn: pytest.WarningsRecorder, iterations: int
) -> None:
    before, after = pair
    serial = {
        (kind, index, key): _pixels(path)
        for kind, index in TASKS
        for key, path in _render(kind, index, before, after, tmp_path / "serial").items()
    }
    assert len(serial) == 32  # 4 index maps + 4 x (change map, before, after), each with its bare map

    for it in range(iterations):
        outdir = tmp_path / f"it{it}"
        with ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
            futures = {(k, i): pool.submit(_render, k, i, before, after, outdir / f"{k}_{i}") for k, i in TASKS}
        for (kind, index), future in futures.items():
            pngs = future.result()  # re-raises a worker exception
            assert len(pngs) == (2 if kind == "si" else 6)
            for key, path in pngs.items():
                assert _pixels(path) == serial[(kind, index, key)], f"iteration {it}: {path} != serial render"

    colorbar = [str(w.message) for w in recwarn if "Adding colorbar to a different Figure" in str(w.message)]
    assert colorbar == []


def test_concurrent_renders_match_serial(
    synthetic_raster_pair: tuple[str, str], tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """4 spectral_index + 4 change_detection runs at once write the same PNGs as serial runs."""
    _check_concurrent_renders(synthetic_raster_pair, tmp_path, recwarn, iterations=1)


@pytest.mark.slow
def test_concurrent_renders_match_serial_20_iterations(
    synthetic_raster_pair: tuple[str, str], tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """The Definition-of-done load: 20 iterations of 4 + 4 concurrent runs."""
    _check_concurrent_renders(synthetic_raster_pair, tmp_path, recwarn, iterations=20)


def test_algorithms_do_not_import_pyplot() -> None:
    """pyplot's global figure state is not thread-safe; algorithms use the OO Figure API."""
    pattern = re.compile(r"matplotlib\.pyplot|import pyplot")
    offenders = [
        f"{path.relative_to(ALGORITHMS_DIR)}:{n}: {line.strip()}"
        for path in sorted(ALGORITHMS_DIR.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert offenders == []
