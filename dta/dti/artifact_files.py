"""Job artifact files (docs/ARCHITECTURE.md "Job artifacts").

Results never carry pixel data: rasters are written as single-band float32
GeoTIFFs (nodata NaN) and visualizations as PNG files, and the result holds a
``{kind, path, url, stats}`` reference per file.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import quote
import uuid

import numpy as np
import rasterio

from dta.config import OUTPUTS_PATH as OUTPUTS_PATH

# (output directory, file-name prefix) of the pipeline step being executed.
_step: ContextVar[tuple[Path, str] | None] = ContextVar("dt4lc_artifact_step", default=None)


def output_dir(job_id: str | None = None) -> Path:
    """``OUTPUTS_PATH/<job_id>/``, or ``OUTPUTS_PATH/run-<uuid hex>/`` outside the job queue."""
    return OUTPUTS_PATH / (job_id or f"run-{uuid.uuid4().hex}")


@contextmanager
def step_outputs(directory: Path, prefix: str) -> Iterator[None]:
    """Files saved inside this block go to ``directory`` as ``<prefix>_<key>.<tif|png>``."""
    token = _step.set((directory, prefix))
    try:
        yield
    finally:
        _step.reset(token)


def ref(kind: str, path: Path | str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Artifact reference for a file the API serves through ``/v1/download``."""
    path = str(path)
    return {"kind": kind, "path": path, "url": "/v1/download?path=" + quote(path, safe=""), "stats": stats or {}}


def write_png(path: Path, png: bytes) -> dict[str, Any]:
    """Write PNG bytes to ``path`` and return its ``png`` reference."""
    path.write_bytes(png)
    return ref("png", path)


def write_geotiff(path: Path, array: np.ndarray, crs: Any, transform: Any) -> dict[str, Any]:
    """Write ``array`` as a single-band float32 GeoTIFF (nodata NaN) and return its ``geotiff`` reference."""
    data = np.asarray(array, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        nodata=np.nan,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    valid = data[np.isfinite(data)].astype(np.float64)
    stats: dict[str, Any] = {"min": None, "max": None, "mean": None, "valid_pixels": 0}
    if valid.size:
        stats = {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "valid_pixels": int(valid.size),
        }
    return ref("geotiff", path, stats)


def save(
    result: dict[str, Any],
    rasters: dict[str, np.ndarray],
    visualizations: dict[str, bytes | None] | None,
    *,
    crs: Any,
    transform: Any,
) -> dict[str, Any]:
    """Write ``rasters`` and ``visualizations`` to files and put their references into ``result``.

    Sets ``result["rasters"]`` and, unless ``visualizations`` is None,
    ``result["visualizations"]`` (same keys; a None image stays None).
    """
    step = _step.get()
    directory, prefix = step if step else (output_dir(), "")
    directory.mkdir(parents=True, exist_ok=True)

    def name(key: str, ext: str) -> str:
        return f"{prefix}_{key}.{ext}" if prefix else f"{key}.{ext}"

    result["rasters"] = {
        key: write_geotiff(directory / name(key, "tif"), array, crs, transform) for key, array in rasters.items()
    }
    if visualizations is not None:
        result["visualizations"] = {
            key: write_png(directory / name(key, "png"), png) if png is not None else None
            for key, png in visualizations.items()
        }
    return result
