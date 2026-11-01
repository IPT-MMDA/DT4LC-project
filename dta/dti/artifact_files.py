"""Job artifact files (docs/ARCHITECTURE.md "Job artifacts").

Results never carry pixel data: rasters are written as single-band float32
GeoTIFFs (nodata NaN) and visualizations as PNG files, and the result holds a
``{kind, path, url, stats, size_bytes}`` reference per file. A finished job adds each
reference's ``download_name`` and puts it in the ``url`` (``server.library.name_downloads``).

A raster saved with a *style* carries it in its GeoTIFF tags (``DT4LC_KIND``
``change|index|class``; ``DT4LC_VMIN``, ``DT4LC_VMAX``, ``DT4LC_RAMP`` for change
and index; ``DT4LC_CLASSES`` for class), so ``/v1/tiles`` draws it in the same
colours as its bare map PNG (:func:`colourize`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import io
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote
import uuid

import numpy as np
from PIL import Image
import rasterio

from dta.config import OUTPUTS_PATH as OUTPUTS_PATH

# Map legend colours: water lost / drier, no change, water gained / wetter.
CHANGE_RAMP = ("#8c4a2f", "#c4cad6", "#2d5f9e")
# Long side of a bare map PNG, in pixels.
BARE_MAX_SIDE = 1024

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


def download_url(path: str, name: str | None = None) -> str:
    """``/v1/download?path=<path>``, plus ``&name=<name>`` (the file name the download gets)."""
    url = "/v1/download?path=" + quote(path, safe="")
    return f"{url}&name={quote(name, safe='')}" if name else url


def ref(kind: str, path: Path | str, stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Artifact reference for a file the API serves through ``/v1/download``.

    ``size_bytes`` is the file's size when the reference is made (right after writing it),
    ``None`` if there is no file yet.
    """
    path = str(path)
    size = os.path.getsize(path) if os.path.isfile(path) else None
    return {
        "kind": kind,
        "path": path,
        "url": download_url(path),
        "stats": stats or {},
        "size_bytes": size,
    }


def hex_colour(colour: str | tuple[float, float, float]) -> str:
    """``(r, g, b)`` in 0-1 (or a hex string) -> ``"#rrggbb"``."""
    if isinstance(colour, str):
        return colour.lower()
    return "#" + "".join(f"{round(c * 255):02x}" for c in colour)


def change_style(limit: float) -> dict[str, Any]:
    """A change raster: :data:`CHANGE_RAMP` over ``[-limit, +limit]`` (negative lost, positive gained)."""
    limit = abs(float(limit)) or 1.0
    return {"kind": "change", "range": [-limit, limit], "ramp": list(CHANGE_RAMP)}


def index_style(colours: Iterable[str | tuple[float, float, float]], vmin: float, vmax: float) -> dict[str, Any]:
    """An index raster: evenly spaced colour stops over ``[vmin, vmax]``."""
    return {"kind": "index", "range": [float(vmin), float(vmax)], "ramp": [hex_colour(c) for c in colours]}


def class_style(classes: dict[int, str]) -> dict[str, Any]:
    """A class raster: one colour per class value; other values are transparent."""
    return {"kind": "class", "classes": {int(k): hex_colour(v) for k, v in classes.items()}}


def style_tags(style: dict[str, Any]) -> dict[str, str]:
    """GeoTIFF tags that carry ``style``."""
    if style["kind"] == "class":
        return {"DT4LC_KIND": "class", "DT4LC_CLASSES": ",".join(f"{k}:{v}" for k, v in style["classes"].items())}
    lo, hi = style["range"]
    return {
        "DT4LC_KIND": style["kind"],
        "DT4LC_VMIN": repr(lo),
        "DT4LC_VMAX": repr(hi),
        "DT4LC_RAMP": ",".join(style["ramp"]),
    }


def read_style(tags: dict[str, str]) -> dict[str, Any] | None:
    """The style in GeoTIFF ``tags`` (see :func:`style_tags`), or ``None`` for an untagged or malformed file."""
    kind = tags.get("DT4LC_KIND")
    try:
        if kind == "class":
            pairs = (item.split(":", 1) for item in tags["DT4LC_CLASSES"].split(","))
            return class_style({int(k): v for k, v in pairs})
        if kind in ("change", "index"):
            ramp = tags["DT4LC_RAMP"].split(",")
            _rgb(ramp)
            return {"kind": kind, "range": [float(tags["DT4LC_VMIN"]), float(tags["DT4LC_VMAX"])], "ramp": ramp}
    except (KeyError, ValueError):
        pass
    return None


def _rgb(colours: list[str]) -> np.ndarray:
    """``["#rrggbb", ...]`` -> ``(n, 3)`` float array in 0-255."""
    if not colours or any(len(c) != 7 or not c.startswith("#") for c in colours):
        raise ValueError(f"Not a list of #rrggbb colours: {colours!r}")
    return np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in colours], dtype=np.float64)


def colourize(data: np.ndarray, style: dict[str, Any]) -> np.ndarray:
    """RGBA uint8 image of a single-band array in ``style``; NaN pixels are transparent.

    Change and index values are clipped to the style's range and interpolated
    linearly between its evenly spaced colour stops (as matplotlib's
    ``LinearSegmentedColormap.from_list``); class values take their class colour.
    """
    data = np.asarray(data, dtype=np.float64)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    valid = np.isfinite(data)
    if style["kind"] == "class":
        for value, colour in style["classes"].items():
            hit = valid & (data == value)
            rgba[hit, :3] = _rgb([colour])[0]
            rgba[hit, 3] = 255
        return rgba
    stops = _rgb(style["ramp"])
    lo, hi = style["range"]
    t = np.clip((data[valid] - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.full(int(valid.sum()), 0.5)
    x = np.linspace(0.0, 1.0, len(stops))
    for channel in range(3):
        rgba[valid, channel] = np.round(np.interp(t, x, stops[:, channel])).astype(np.uint8)
    rgba[valid, 3] = 255
    return rgba


def write_bare_png(path: Path, array: np.ndarray, style: dict[str, Any]) -> dict[str, Any]:
    """Write ``array`` as a map-only RGBA PNG: no title, colour bar or margin, NaN transparent.

    The image keeps the raster's aspect, at most :data:`BARE_MAX_SIDE` pixels on
    the long side (never upscaled). The reference's ``stats`` hold the image's
    ``width``, ``height`` and the ``style`` it was drawn in.
    """
    img = Image.fromarray(colourize(array, style), mode="RGBA")
    scale = BARE_MAX_SIDE / max(img.size)
    if scale < 1:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        resample = Image.Resampling.NEAREST if style["kind"] == "class" else Image.Resampling.BOX
        img = img.resize(size, resample)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())
    return ref("png", path, {"width": img.width, "height": img.height, "style": style})


def write_png(path: Path, png: bytes) -> dict[str, Any]:
    """Write PNG bytes to ``path`` and return its ``png`` reference."""
    path.write_bytes(png)
    return ref("png", path)


def write_geotiff(
    path: Path, array: np.ndarray, crs: Any, transform: Any, style: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write ``array`` as a single-band float32 GeoTIFF (nodata NaN), ``style`` in its tags; return its reference."""
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
        if style is not None:
            dst.update_tags(**style_tags(style))
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
    styles: dict[str, dict[str, Any]] | None = None,
    bare: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``rasters`` and ``visualizations`` to files and put their references into ``result``.

    Sets ``result["rasters"]`` and, unless ``visualizations`` is None,
    ``result["visualizations"]`` (same keys; a None image stays None).
    ``styles`` maps raster keys to the style written into their GeoTIFF tags.
    ``bare`` maps visualization keys to styled raster keys: each adds a map-only
    PNG ``visualizations["<key>_bare"]`` of that raster (:func:`write_bare_png`).
    """
    styles = styles or {}
    step = _step.get()
    directory, prefix = step if step else (output_dir(), "")
    directory.mkdir(parents=True, exist_ok=True)

    def name(key: str, ext: str) -> str:
        return f"{prefix}_{key}.{ext}" if prefix else f"{key}.{ext}"

    result["rasters"] = {
        key: write_geotiff(directory / name(key, "tif"), array, crs, transform, styles.get(key))
        for key, array in rasters.items()
    }
    if visualizations is not None:
        result["visualizations"] = {
            key: write_png(directory / name(key, "png"), png) if png is not None else None
            for key, png in visualizations.items()
        }
        for key, raster_key in (bare or {}).items():
            result["visualizations"][f"{key}_bare"] = write_bare_png(
                directory / name(f"{key}_bare", "png"), rasters[raster_key], styles[raster_key]
            )
    return result
