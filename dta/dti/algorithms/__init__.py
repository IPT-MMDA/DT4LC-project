"""Geospatial algorithms for DT4LC.

This package provides algorithms for processing satellite imagery and
geospatial data, including vegetation indices, change detection, and
statistical analysis.
"""

from .ndvi import calculate_ndvi, ndvi_change
from .statistics import calculate_statistics

__all__ = ["calculate_ndvi", "ndvi_change", "calculate_statistics"]
