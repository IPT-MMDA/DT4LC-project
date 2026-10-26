"""Earth Engine feature builder of the LULC change-hazard model.

Builds, for a bounding box and an origin year ``t0``, the per-cell features the model is trained on, on a 30 m
grid, with the recipes of the training data (the ``ukr-lulc`` dataset: v1.0 chips and the ext1 30 m layers), so
that a model trained on that data sees the same numbers here:

- ``start_class`` (not a feature: it selects the per-from-class model): Esri/IO annual LULC of ``t0`` at 10 m in
  its own legend (:data:`ESRI_CLASSES`, clouds = no data), 3 x 3 block mode to 30 m, no data ignored, ties to the
  centre pixel's class, else the lowest class index.
- ``esri_nbr{3,7,15}_<class>``: share of each class among the valid 30 m cells of the r x r square around a cell.
- ``elevation_m``, ``slope_deg``: Copernicus GLO-30 bilinear to 10 m (slope: ``ee.Terrain.slope`` on the 30 m grid,
  then bilinear to 10 m), 3 x 3 block mean. ``aspect_sin_east``, ``aspect_cos_north``: the downslope direction from
  the gradient of the 30 m elevation. The DEM release is chosen by key (:data:`DEM_ASSETS`): ``glo30_2024_1`` by
  default; ``glo30``, the release Earth Engine deprecated, is the one dataset v1.0 was built from.
- ``ae_A00`` .. ``ae_A63``: AlphaEarth annual embedding of ``t0``, nearest onto the 10 m grid, then mean onto the
  30 m grid in Earth Engine (``reduceResolution``), float32 as ext1's ``alphaearth/mean30m``. The v1.0 cube stored
  the same mean as float16.

The grid has a halo of :data:`HALO` cells (600 m), so the largest neighbourhood (15 cells) and the gradient never
reach its edge inside the core; features are returned for the core only. Nothing here initialises Earth Engine:
the caller does (``dta.dti.data_sources.gee_common.initialize_gee``, or ``ee.Initialize`` in the CLI).

    python -m dta.dti.models.lulc_hazard.features --bbox 30.40 50.35 30.55 50.45 --out cube.npz [--t0 2024]

Without ``--t0`` the CLI, like the runner, uses the latest year with both an Esri map and an AlphaEarth embedding
(:func:`latest_t0`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import logging
import math
import time
from typing import Any
import warnings

import numpy as np

logger = logging.getLogger(__name__)

RES = 30.0  # model grid, metres
FINE = 10.0  # grid of the 10 m inputs (Esri, DEM): 3 x 3 cells per model cell
HALO = 20  # cells around the core; >= the largest neighbourhood radius (7) and the training chips' halo
RADII = (3, 7, 15)  # neighbourhood square side, in 30 m cells
NODATA = -9999.0

ESRI_ASSET = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ESRI_CLASSES = ("Water", "Trees", "FloodedVeg", "Crops", "Built", "Bare", "Snow", "Rangeland")  # index 1..8
_ESRI_CODES = {1: "Water", 2: "Trees", 4: "FloodedVeg", 5: "Crops", 7: "Built", 8: "Bare", 9: "Snow", 11: "Rangeland"}
ESRI_LUT = np.zeros(256, np.uint8)  # Esri code -> class index; 0 (no data) for clouds (10) and anything else
for _code, _name in _ESRI_CODES.items():
    ESRI_LUT[_code] = ESRI_CLASSES.index(_name) + 1

DEM_ASSETS = {
    "glo30_2024_1": "COPERNICUS/DEM/GLO30_2024_1",
    "glo30": "COPERNICUS/DEM/GLO30",  # deprecated in Earth Engine; dataset v1.0 (and its cube) used it
}
DEM_DEFAULT = "glo30_2024_1"
T0_FIRST = 2017  # first year with both Esri and AlphaEarth
AE_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AE_BANDS = tuple(f"A{i:02d}" for i in range(64))

NBR_COLUMNS = tuple(f"esri_nbr{r}_{c}" for r in RADII for c in ESRI_CLASSES)
TERRAIN_COLUMNS = ("elevation_m", "slope_deg", "aspect_sin_east", "aspect_cos_north")
AE_COLUMNS = tuple(f"ae_{b}" for b in AE_BANDS)
COLUMNS = NBR_COLUMNS + TERRAIN_COLUMNS + AE_COLUMNS

THREADS = 4  # concurrent computePixels requests
TILE = {"esri": 1024, "dem": 1024, "ae": 256}  # tile side in pixels: each response stays well under 48 MB


class FeatureUnavailableError(RuntimeError):
    """Earth Engine has no input for this area and year (e.g. no AlphaEarth embedding of ``t0``)."""


# ----------------------------------------------------------------- grid


@dataclass(frozen=True)
class Grid:
    """A 30 m grid: ``nx`` x ``ny`` core cells with :data:`HALO` cells around them, north-up, in ``crs`` metres."""

    crs: str
    x0: float  # west edge of the full grid (core + halo)
    y0: float  # north edge of the full grid
    nx: int
    ny: int
    halo: int = HALO

    @property
    def width(self) -> int:
        return self.nx + 2 * self.halo

    @property
    def height(self) -> int:
        return self.ny + 2 * self.halo

    @property
    def core(self) -> tuple[slice, slice]:
        return slice(self.halo, self.halo + self.ny), slice(self.halo, self.halo + self.nx)

    @property
    def area_km2(self) -> float:
        return self.nx * self.ny * RES * RES / 1e6

    def transform(self) -> Any:
        """Affine transform of the core (the rasters the model writes)."""
        from rasterio.transform import from_origin

        return from_origin(self.x0 + self.halo * RES, self.y0 - self.halo * RES, RES, RES)


def utm_crs(lon: float, lat: float) -> str:
    """EPSG code of the WGS84 UTM zone holding ``lon``, ``lat``."""
    zone = min(60, int((lon + 180) // 6) + 1)
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def grid_for_bbox(
    bbox: Sequence[float], crs: str = "auto", origin: Sequence[float] = (0.0, 0.0), halo: int = HALO
) -> Grid:
    """The 30 m grid covering ``bbox`` ([west, south, east, north], degrees) in ``crs``.

    ``crs="auto"`` is the UTM zone of the box centre. The core is snapped outwards to the lattice through
    ``origin`` (the training grid's origin, so cells line up with the training chips).
    """
    from rasterio.warp import transform_bounds

    west, south, east, north = (float(v) for v in bbox)
    if crs == "auto":
        crs = utm_crs((west + east) / 2, (south + north) / 2)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    ox, oy = (float(v) for v in origin)
    cx0 = ox + math.floor((minx - ox) / RES) * RES
    cx1 = ox + math.ceil((maxx - ox) / RES) * RES
    cy0 = oy + math.floor((miny - oy) / RES) * RES
    cy1 = oy + math.ceil((maxy - oy) / RES) * RES
    nx, ny = round((cx1 - cx0) / RES), round((cy1 - cy0) / RES)
    return Grid(crs, cx0 - halo * RES, cy1 + halo * RES, nx, ny, halo)


# ----------------------------------------------------------------- local reductions (copies of the training code)


def mode3(a: np.ndarray, k: int) -> np.ndarray:
    """3 x 3 block mode of a class-index raster (0 = no data, ignored). Tie -> centre class, else lowest index."""
    h, w = a.shape[0] // 3, a.shape[1] // 3
    b = a[: h * 3, : w * 3].reshape(h, 3, w, 3).transpose(0, 2, 1, 3).reshape(h, w, 9)
    cen = b[..., 4]
    cnt = np.empty((k, h, w), np.float32)
    for c in range(1, k + 1):
        cnt[c - 1] = (b == c).sum(-1) + 0.5 * (cen == c)
    m = cnt.argmax(0).astype(np.uint8) + 1
    m[cnt.max(0) == 0] = 0
    return m


def mean3(a: np.ndarray) -> np.ndarray:
    """NaN-aware 3 x 3 block mean -> float32 (an all-NaN block stays NaN)."""
    *lead, h, w = a.shape
    h, w = h // 3, w // 3
    b = a[..., : h * 3, : w * 3].reshape(*lead, h, 3, w, 3).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice" on all-NaN blocks
        return np.nanmean(b, axis=(-3, -1)).astype(np.float32)


def neighbourhood_shares(
    cls: np.ndarray, radii: Iterable[int] = RADII, k: int = len(ESRI_CLASSES)
) -> dict[int, np.ndarray]:
    """Per radius, ``(k, h, w)`` share of each class among the valid cells of the r x r square (NaN with none)."""
    from scipy.ndimage import uniform_filter

    valid = (cls > 0).astype(np.float32)
    out = {}
    for r in radii:
        den = uniform_filter(valid, r, mode="constant", cval=0.0)
        shares = np.empty((k, *cls.shape), np.float32)
        for c in range(1, k + 1):
            num = uniform_filter((cls == c).astype(np.float32), r, mode="constant", cval=0.0)
            shares[c - 1] = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
        out[r] = shares
    return out


def aspect(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sin, cos)`` of the downslope azimuth of a north-up 30 m elevation raster; 0 if flat, NaN if unknown."""
    gy, gx = np.gradient(elevation, RES)  # gy = d/drow (southward), gx = d/dcol (eastward)
    g_e, g_n = gx, -gy
    mag = np.hypot(g_e, g_n)
    flat = ~(mag > 1e-6)
    east = np.where(flat, 0.0, -g_e / np.where(flat, 1, mag)).astype(np.float32)
    north = np.where(flat, 0.0, -g_n / np.where(flat, 1, mag)).astype(np.float32)
    east[np.isnan(mag)] = np.nan
    north[np.isnan(mag)] = np.nan
    return east, north


# ----------------------------------------------------------------- Earth Engine pulls


def _request(ee: Any, image: Any, crs: str, x0: float, y0: float, w: int, h: int, res: float) -> dict[str, Any]:
    return {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": int(w), "height": int(h)},
            "affineTransform": {
                "scaleX": res,
                "shearX": 0,
                "translateX": x0,
                "shearY": 0,
                "scaleY": -res,
                "translateY": y0,
            },
            "crsCode": crs,
        },
    }


def _pull_once(
    ee: Any,
    image: Any,
    bands: Sequence[str],
    crs: str,
    x0: float,
    y0: float,
    w: int,
    h: int,
    res: float,
    tries: int = 6,
) -> np.ndarray:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            raw = ee.data.computePixels(_request(ee, image, crs, x0, y0, w, h, res))
            rec = np.load(io.BytesIO(raw), allow_pickle=False)
            return np.stack([rec[b] for b in bands]).astype(np.float32)
        except Exception as e:  # noqa: BLE001 - Earth Engine raises plain exceptions for quota and size errors
            text = str(e).lower()
            if any(k in text for k in ("too many pixels", "must be less than", "exceed", "memory")):
                raise MemoryError(str(e)) from e
            last = e
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Earth Engine computePixels failed: {last}")


def _pull_tile(
    ee: Any, image: Any, bands: Sequence[str], crs: str, x0: float, y0: float, w: int, h: int, res: float
) -> np.ndarray:
    """One tile; split in four while Earth Engine says it is too large."""
    try:
        return _pull_once(ee, image, bands, crs, x0, y0, w, h, res)
    except MemoryError:
        if w < 64 and h < 64:
            raise
        w1, h1 = max(w // 2, 1), max(h // 2, 1)
        out = np.empty((len(bands), h, w), np.float32)
        for dx, dy, ww, hh in ((0, 0, w1, h1), (w1, 0, w - w1, h1), (0, h1, w1, h - h1), (w1, h1, w - w1, h - h1)):
            if ww and hh:
                out[:, dy : dy + hh, dx : dx + ww] = _pull_tile(
                    ee, image, bands, crs, x0 + dx * res, y0 - dy * res, ww, hh, res
                )
        return out


def pull(
    ee: Any, image: Any, bands: Sequence[str], grid: Grid, res: float, tile: int, nodata: float = NODATA
) -> np.ndarray:
    """``(bands, h, w)`` float32 of ``image`` on the full ``grid`` (halo included) at ``res``; ``nodata`` -> NaN."""
    scale = round(RES / res)
    w, h = grid.width * scale, grid.height * scale
    expr = image.select(list(bands)).unmask(nodata)
    out = np.empty((len(bands), h, w), np.float32)

    def one(rc: tuple[int, int]) -> None:
        r, c = rc
        tw, th = min(tile, w - c), min(tile, h - r)
        out[:, r : r + th, c : c + tw] = _pull_tile(
            ee, expr, bands, grid.crs, grid.x0 + c * res, grid.y0 - r * res, tw, th, res
        )

    jobs = [(r, c) for r in range(0, h, tile) for c in range(0, w, tile)]
    with ThreadPoolExecutor(min(THREADS, len(jobs))) as ex:
        list(ex.map(one, jobs))
    out[out == nodata] = np.nan
    return out


def _rect(ee: Any, grid: Grid) -> Any:
    return ee.Geometry.Rectangle(
        [grid.x0, grid.y0 - grid.height * RES, grid.x0 + grid.width * RES, grid.y0], proj=grid.crs, geodesic=False
    )


def esri_collection(ee: Any, year: int) -> Any:
    """Esri/IO tiles of ``year`` (2017-2024 assets end in ``_<year>``; 2025 assets are ``<tile>_<year>0101-...``)."""
    return ee.ImageCollection(ESRI_ASSET).filter(
        ee.Filter.Or(
            ee.Filter.stringEndsWith("system:index", f"_{year}"),
            ee.Filter.stringContains("system:index", f"_{year}0101-"),
        )
    )


def ae_collection(ee: Any, year: int) -> Any:
    return ee.ImageCollection(AE_ASSET).filterDate(f"{year}-01-01", f"{year + 1}-01-01")


def esri_image(ee: Any, grid: Grid, year: int) -> Any:
    fine = ee.Projection(grid.crs, [FINE, 0, grid.x0, 0, -FINE, grid.y0])
    return esri_collection(ee, year).filterBounds(_rect(ee, grid)).mosaic().setDefaultProjection(fine).select(["b1"])


def check_dem(dem: str) -> str:
    if dem not in DEM_ASSETS:
        raise ValueError(f"Unknown DEM {dem!r}; one of {', '.join(DEM_ASSETS)}.")
    return dem


def dem_image(ee: Any, grid: Grid, release: str = DEM_DEFAULT) -> Any:
    coarse = ee.Projection(grid.crs, [RES, 0, grid.x0, 0, -RES, grid.y0])
    dem = (
        ee.ImageCollection(DEM_ASSETS[check_dem(release)])
        .select("DEM")
        .mosaic()
        .setDefaultProjection(ee.Projection("EPSG:4326", [1 / 3600, 0, -180, 0, -1 / 3600, 90]))
    )
    slope30 = ee.Terrain.slope(dem.resample("bilinear").reproject(coarse))
    return ee.Image.cat(
        [dem.resample("bilinear").rename("elevation_m"), slope30.resample("bilinear").rename("slope_deg")]
    )


def ae_image(ee: Any, grid: Grid, year: int) -> Any:
    img = ae_collection(ee, year).filterBounds(_rect(ee, grid)).mosaic().select(list(AE_BANDS))
    img10 = img.reproject(crs=grid.crs, crsTransform=[FINE, 0, grid.x0, 0, -FINE, grid.y0])
    return img10.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)


def available_years(ee: Any, grid: Grid, years: Iterable[int]) -> dict[str, list[int]]:
    """Years of ``years`` with Esri tiles and with AlphaEarth images over ``grid`` (one Earth Engine call)."""
    rect = _rect(ee, grid)
    years = list(years)
    sizes = ee.Dictionary(
        {
            **{f"esri_{y}": esri_collection(ee, y).filterBounds(rect).size() for y in years},
            **{f"ae_{y}": ae_collection(ee, y).filterBounds(rect).size() for y in years},
        }
    ).getInfo()
    return {p: [y for y in years if sizes[f"{p}_{y}"] > 0] for p in ("esri", "ae")}


def latest_t0(ee: Any, grid: Grid, first: int = T0_FIRST, last: int | None = None, need_ae: bool = True) -> int:
    """The latest year in ``first`` .. ``last`` (default: this year) with an Esri map (and an AlphaEarth embedding)."""
    import datetime

    last = last or datetime.date.today().year
    have = available_years(ee, grid, range(first, last + 1))
    years = set(have["esri"]) & set(have["ae"]) if need_ae else set(have["esri"])
    if not years:
        raise FeatureUnavailableError(
            f"Earth Engine has no Esri map{' with an AlphaEarth embedding' if need_ae else ''} "
            f"of this area for {first}-{last}."
        )
    return max(years)


# ----------------------------------------------------------------- the cube


@dataclass
class FeatureCube:
    """Core-only features of one grid and year."""

    grid: Grid
    t0: int
    start_class: np.ndarray  # (ny, nx) uint8: 0 = no data, i = ESRI_CLASSES[i - 1]
    features: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (ny, nx) float32
    seconds: dict[str, float] = field(default_factory=dict)  # wall time per input

    def matrix(self, columns: Sequence[str]) -> np.ndarray:
        """``(ny * nx, len(columns))`` float32, row-major cells, NaN where an input is missing."""
        return np.stack([self.features[c].ravel() for c in columns], axis=1).astype(np.float32, copy=False)


def check_columns(columns: Iterable[str]) -> list[str]:
    """``columns`` as a list; ValueError naming any this builder does not make."""
    columns = list(columns)
    unknown = [c for c in columns if c not in COLUMNS]
    if unknown:
        raise ValueError(
            f"The feature builder makes no {', '.join(unknown)}; it makes {len(COLUMNS)} columns "
            f"(esri_nbr*, terrain, ae_A00..ae_A63)."
        )
    return columns


def build_features(
    grid: Grid,
    t0: int,
    columns: Iterable[str] = COLUMNS,
    ee: Any = None,
    check_years: bool = True,
    dem: str = DEM_DEFAULT,
) -> FeatureCube:
    """Features of ``columns`` at ``t0`` on ``grid`` (Earth Engine must be initialised)."""
    if ee is None:
        import ee as ee_module

        ee = ee_module
    columns = check_columns(columns)
    check_dem(dem)
    need_ae = any(c in AE_COLUMNS for c in columns)
    need_dem = any(c in TERRAIN_COLUMNS for c in columns)
    if check_years:
        have = available_years(ee, grid, [t0])
        if t0 not in have["esri"]:
            raise FeatureUnavailableError(f"Esri land cover has no {t0} map of this area.")
        if need_ae and t0 not in have["ae"]:
            raise FeatureUnavailableError(f"AlphaEarth has no {t0} embedding of this area.")

    cube = FeatureCube(grid, t0, np.zeros((grid.ny, grid.nx), np.uint8))
    core = grid.core

    t = time.time()
    codes = pull(ee, esri_image(ee, grid, t0), ["b1"], grid, FINE, TILE["esri"], nodata=0)[0]
    codes = np.nan_to_num(codes, nan=0).astype(np.uint8)
    cls = mode3(ESRI_LUT[codes], len(ESRI_CLASSES))
    cube.start_class = cls[core].copy()
    if any(c in NBR_COLUMNS for c in columns):
        for r, shares in neighbourhood_shares(cls).items():
            for i, name in enumerate(ESRI_CLASSES):
                cube.features[f"esri_nbr{r}_{name}"] = shares[i][core]
    cube.seconds["esri"] = round(time.time() - t, 1)

    if need_dem:
        t = time.time()
        elev10, slope10 = pull(ee, dem_image(ee, grid, dem), ["elevation_m", "slope_deg"], grid, FINE, TILE["dem"])
        elev = mean3(elev10)
        east, north = aspect(elev)
        cube.features["elevation_m"] = elev[core]
        cube.features["slope_deg"] = mean3(slope10)[core]
        cube.features["aspect_sin_east"] = east[core]
        cube.features["aspect_cos_north"] = north[core]
        cube.seconds["dem"] = round(time.time() - t, 1)

    if need_ae:
        t = time.time()
        ae = pull(ee, ae_image(ee, grid, t0), AE_BANDS, grid, RES, TILE["ae"])
        for i, b in enumerate(AE_BANDS):
            cube.features[f"ae_{b}"] = ae[i][core]
        cube.seconds["ae"] = round(time.time() - t, 1)

    cube.features = {c: cube.features[c] for c in columns}
    return cube


def save_npz(cube: FeatureCube, path: str) -> None:
    """The cube as ``.npz``: ``start_class``, ``X`` (columns, ny, nx), ``columns`` and the grid."""
    g = cube.grid
    np.savez_compressed(
        path,
        start_class=cube.start_class,
        X=np.stack(list(cube.features.values())) if cube.features else np.zeros((0, g.ny, g.nx), np.float32),
        columns=np.array(list(cube.features)),
        grid=np.array([g.crs, g.x0, g.y0, g.nx, g.ny, g.halo], dtype=object).astype(str),
        t0=cube.t0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--t0", type=int, help="origin year (default: the latest with Esri and AlphaEarth)")
    ap.add_argument("--crs", default="auto")
    ap.add_argument("--origin", nargs=2, type=float, default=(0.0, 0.0))
    ap.add_argument("--dem", choices=list(DEM_ASSETS), default=DEM_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default=os.environ.get("GEE_PROJECT_ID"))
    a = ap.parse_args(argv)
    import ee

    ee.Initialize(project=a.project)
    grid = grid_for_bbox(a.bbox, a.crs, a.origin)
    t0 = a.t0 or latest_t0(ee, grid)
    cube = build_features(grid, t0, ee=ee, check_years=a.t0 is not None, dem=a.dem)
    save_npz(cube, a.out)
    print(f"{a.out}: t0 {t0}, {grid.nx} x {grid.ny} cells, {len(cube.features)} columns, {cube.seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
