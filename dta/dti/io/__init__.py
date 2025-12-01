"""DT4LC IO utilities for raster and vector data."""

from .raster import RasterData, load_raster_as_rgb, validate_geotiff

__all__ = ["load_raster_as_rgb", "validate_geotiff", "RasterData"]
