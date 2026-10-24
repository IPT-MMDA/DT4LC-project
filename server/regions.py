"""Region chip for ``POST /v1/jobs`` with ``region`` (DESIGN §4): area, bounds and the display line.

The request model is :class:`server.schemas.JobRegion`: its validator calls :func:`region_shape` and
:func:`area_km2`, its ``echo()`` :func:`region_line`. The area is geodesic on the WGS84 ellipsoid (pyproj,
installed with geopandas and rasterio).
"""

from __future__ import annotations

from typing import Any

from pyproj import Geod
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from dta.dti.data_sources.gee_region import date_range

# Largest region a job may fetch (50 km × 50 km). A fetch can still be too large for Earth Engine's
# direct download at fine resolution (Sentinel-2 at 10 m); the fetch step then fails and says so.
MAX_REGION_KM2 = 2500.0
MAX_COORDINATES = 10_000

_GEOD = Geod(ellps="WGS84")


def _positions(coordinates: Any) -> list[Any]:
    """Every ``[lon, lat]`` position in a Polygon's or MultiPolygon's nested coordinate lists."""
    if isinstance(coordinates, list | tuple) and coordinates and isinstance(coordinates[0], int | float):
        return [coordinates]
    if not isinstance(coordinates, list | tuple):
        raise ValueError("region geometry coordinates must be nested lists of [lon, lat]")
    return [p for c in coordinates for p in _positions(c)]


def region_shape(geometry: dict[str, Any] | None, bbox: list[float] | None) -> BaseGeometry:
    """The region as a shapely geometry; ValueError if it is not a sane WGS84 polygon or bbox."""
    if (geometry is None) == (bbox is None):
        raise ValueError("region needs exactly one of geometry and bbox")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("region bbox must be [west, south, east, north]")
        west, south, east, north = bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(
                "region bbox must be [west, south, east, north] in degrees with west < east, south < north"
            )
        return box(west, south, east, north)
    assert geometry is not None
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("region geometry must be a GeoJSON Polygon or MultiPolygon")
    positions = _positions(geometry.get("coordinates"))
    if len(positions) > MAX_COORDINATES:
        raise ValueError(f"region geometry has more than {MAX_COORDINATES} positions")
    for p in positions:
        if len(p) < 2 or not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90):
            raise ValueError(f"region geometry position {list(p)} is not [lon, lat] in degrees")
    try:
        geom = shape(geometry)
    except Exception as e:  # shapely raises several types for malformed rings
        raise ValueError(f"region geometry is not a valid polygon: {e}") from e
    if geom.is_empty or not geom.is_valid or geom.area == 0:
        raise ValueError("region geometry is not a valid polygon (empty, zero area or self-intersecting)")
    return geom


def area_km2(geom: BaseGeometry) -> float:
    """Geodesic area of a WGS84 geometry in km²."""
    area_m2, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(float(area_m2)) / 1e6


def format_area(km2: float) -> str:
    """``"412 km²"``; one decimal below 10 km²."""
    return f"{km2:.1f} km²" if km2 < 10 else f"{km2:.0f} km²"


def region_line(region: dict[str, Any]) -> str:
    """``"Region · Dnipro floodplain · Kherson · 412 km² · HLS L30 · 2023-06-01 → 06-09 · ≤20 % cloud"``.

    ``name`` and ``place`` are left out when missing.
    """
    parts = ["Region", region.get("name"), region.get("place"), format_area(region["area_km2"]), region["dataset"]]
    parts += [date_range(region["start_date"], region["end_date"]), f"≤{region['cloud_max']} % cloud"]
    return " · ".join(p for p in parts if p)
