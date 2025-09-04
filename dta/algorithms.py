from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio


@dataclass
class Algorithms:
    def ndvi(self, raster_path: str) -> dict[str, Any]:
        # Simple NDVI from GeoTIFF bands if available: (NIR - Red) / (NIR + Red)
        # We handle common band orders loosely; this is a minimal placeholder.
        with rasterio.open(raster_path) as src:
            count = src.count
            red_idx = 3 if count >= 4 else 1
            nir_idx = 4 if count >= 5 else count
            red = src.read(red_idx).astype("float32")
            nir = src.read(nir_idx).astype("float32")
        denom = (nir + red)
        denom[denom == 0] = 1e-6
        ndvi = (nir - red) / denom
        return {"ndvi": ndvi}

    def ndvi_change(self, ndvi_a: dict[str, Any], ndvi_b: dict[str, Any]) -> dict[str, Any]:
        a = ndvi_a.get("ndvi")
        b = ndvi_b.get("ndvi")
        if a is None or b is None:
            return {"change": None}
        # Align shapes if needed by minimal cropping
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        change = (b[:h, :w] - a[:h, :w]).astype("float32")
        return {"change": change}

    def stats_basic(self, raster_path: str) -> dict[str, Any]:
        # Compute basic per-band stats and an overall histogram for the first band
        with rasterio.open(raster_path) as src:
            count = src.count
            stats = []
            for i in range(1, count + 1):
                band = src.read(i).astype("float32")
                stats.append({
                    "band": i,
                    "min": float(np.nanmin(band)),
                    "max": float(np.nanmax(band)),
                    "mean": float(np.nanmean(band)),
                    "std": float(np.nanstd(band)),
                })
            first = src.read(1).astype("float32")
        hist, bin_edges = np.histogram(first[~np.isnan(first)], bins=32)
        return {"stats": stats, "histogram": {"bins": bin_edges.tolist(), "counts": hist.tolist()}}
