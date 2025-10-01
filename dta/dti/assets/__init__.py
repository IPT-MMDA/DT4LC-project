"""Data asset management for DT4LC.

This package handles loading, caching, and managing data sources including
local files, remote URLs, and in-memory data.
"""

from .manager import DataAssetManager

__all__ = ["DataAssetManager"]
