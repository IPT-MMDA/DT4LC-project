"""Data Asset Manager - handles data loading and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rasterio


class DataAssetManager:
    """Manages data sources and provides unified access to assets.

    The manager handles:
    - Local file paths (absolute and relative)
    - Remote URLs (future)
    - Cached data (future)
    - Metadata extraction

    Example:
        >>> manager = DataAssetManager()
        >>> path = manager.resolve("kahovka_raster")
        >>> print(path)
        /path/to/kahovka_data/raster.tif
    """

    def __init__(self, data_root: Path | None = None) -> None:
        """Initialize asset manager.

        Args:
            data_root: Root directory for relative paths. Defaults to project resources/
        """
        if data_root is None:
            from dta.config import ROOT_DIR

            data_root = ROOT_DIR / "resources"

        self.data_root = Path(data_root)
        self.cache: dict[str, Any] = {}

    def resolve(self, asset_id: str, params: dict[str, Any] | None = None) -> str:
        """Resolve an asset ID to a file path.

        Args:
            asset_id: Asset identifier (e.g., "kahovka_raster", "file://path/to/data.tif")
            params: Optional parameters for asset resolution

        Returns:
            Absolute path to the asset

        Raises:
            FileNotFoundError: If asset cannot be found
            ValueError: If asset_id format is invalid
        """
        params = params or {}

        # Handle file:// URIs
        if asset_id.startswith("file://"):
            path = Path(asset_id[7:])  # Remove "file://"
            if not path.is_absolute():
                path = self.data_root / path
            if not path.exists():
                raise FileNotFoundError(f"Asset not found: {path}")
            return str(path.resolve())

        # Handle absolute paths
        if asset_id.startswith("/") or (len(asset_id) > 1 and asset_id[1] == ":"):
            path = Path(asset_id)
            if not path.exists():
                raise FileNotFoundError(f"Asset not found: {path}")
            return str(path.resolve())

        # Handle named assets (lookup from known locations)
        resolved = self._resolve_named_asset(asset_id, params)
        if resolved:
            return resolved

        # Try as relative path from data_root
        path = self.data_root / asset_id
        if path.exists():
            return str(path.resolve())

        raise FileNotFoundError(
            f"Asset '{asset_id}' not found. Tried:\n"
            f"  - file:// URI\n"
            f"  - Absolute path\n"
            f"  - Named asset\n"
            f"  - Relative to {self.data_root}"
        )

    def _resolve_named_asset(self, name: str, params: dict[str, Any]) -> str | None:
        """Resolve well-known asset names.

        Args:
            name: Asset name (e.g., "kahovka_raster", "prithvi_example")
            params: Additional parameters

        Returns:
            Resolved path or None if not found
        """
        # Kahovka data
        if name == "kahovka_raster":
            search_dirs = [
                self.data_root / "kahovka_data",
                self.data_root,
            ]
            for search_dir in search_dirs:
                if search_dir.exists():
                    # Look for .tif files
                    tif_files = list(search_dir.glob("*.tif")) + list(search_dir.glob("*.tiff"))
                    if tif_files:
                        return str(tif_files[0].resolve())

        # Prithvi example data
        if name == "prithvi_example":
            prithvi_dir = self.data_root / "prithvi_eo_v1_100m" / "examples"
            if prithvi_dir.exists():
                examples = list(prithvi_dir.glob("*.tif"))
                if examples:
                    # Return first example or specific one from params
                    idx = params.get("index", 0)
                    if idx < len(examples):
                        return str(examples[idx].resolve())

        return None

    def get_metadata(self, asset_path: str) -> dict[str, Any]:
        """Extract metadata from a raster asset.

        Args:
            asset_path: Path to raster file

        Returns:
            Dictionary with metadata (CRS, bounds, bands, etc.)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be opened as raster
        """
        path = Path(asset_path)
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {asset_path}")

        # Cache key
        cache_key = f"metadata:{asset_path}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            with rasterio.open(asset_path) as src:
                metadata = {
                    "path": str(asset_path),
                    "driver": src.driver,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]) if src.dtypes else None,
                    "crs": src.crs.to_string() if src.crs else None,
                    "bounds": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top,
                    },
                    "transform": list(src.transform)[:6] if src.transform else None,
                    "nodata": src.nodata,
                    "resolution": src.res if hasattr(src, "res") else None,
                }

                # Cache metadata
                self.cache[cache_key] = metadata
                return metadata

        except Exception as e:
            raise ValueError(f"Failed to read raster metadata: {e}") from e

    def list_assets(self, pattern: str = "*") -> list[str]:
        """List available assets in data_root.

        Args:
            pattern: Glob pattern (e.g., "*.tif", "kahovka_data/*.tif")

        Returns:
            List of asset paths
        """
        if not self.data_root.exists():
            return []

        matches = list(self.data_root.glob(f"**/{pattern}"))
        return [str(p.resolve()) for p in matches if p.is_file()]

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self.cache.clear()
