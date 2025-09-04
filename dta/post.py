from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from cognitive_ui.core.visualization import enhance_raster_with_current_settings, load_raster
from pathlib import Path

from pathlib import Path


@dataclass
class PostProcessor:
    """Transforms raw outputs into UI-friendly artifacts.

    For now, visualization via WMS is deferred. We return a concise summary only.
    """

    def summarize(self, data: Any | None = None) -> Dict[str, Any]:
        # Produce a concise, context-aware textual summary for common outputs
        if isinstance(data, dict):
            # Stats summary (land cover distribution proxy)
            if "stats" in data:
                bands = data.get("stats", [])
                parts: list[str] = []
                for b in bands[:3]:  # keep concise in chat
                    parts.append(
                        f"Band {b.get('band')}: min={b.get('min'):.3f}, mean={b.get('mean'):.3f}, "
                        f"max={b.get('max'):.3f}, std={b.get('std'):.3f}"
                    )
                more = "" if len(bands) <= 3 else f" (+{len(bands)-3} more bands)"
                meta = {
                    "bands": bands,
                    "histogram": data.get("histogram"),
                }
                return {"summary": "Distribution summary (per-band) — " + "; ".join(parts) + more, "meta": meta}

            # NDVI summary
            if "ndvi" in data:
                ndvi = data["ndvi"]
                ndvi_min = float(np.nanmin(ndvi))
                ndvi_mean = float(np.nanmean(ndvi))
                ndvi_max = float(np.nanmax(ndvi))
                msg = (
                    f"NDVI range: {ndvi_min:.3f} to {ndvi_max:.3f} (mean {ndvi_mean:.3f}). "
                    "Higher values indicate denser/healthier vegetation."
                )
                meta = {"ndvi_min": ndvi_min, "ndvi_mean": ndvi_mean, "ndvi_max": ndvi_max}
                return {"summary": msg, "meta": meta}

            # NDVI change summary
            if "change" in data:
                ch = data["change"]
                pos = float(np.mean(ch > 0)) if ch is not None else 0.0
                neg = float(np.mean(ch < 0)) if ch is not None else 0.0
                msg = (
                    f"NDVI change map: {pos*100:.1f}% increase, {neg*100:.1f}% decrease. "
                    "Positive values imply greening, negative imply browning."
                )
                meta = {"increase_ratio": pos, "decrease_ratio": neg}
                return {"summary": msg, "meta": meta}

        # LLM text summary
        if isinstance(data, dict) and "text" in data:
            return {"summary": str(data.get("text"))}

        # Fallback
        return {"summary": "Processing complete. Results available above."}

    def visualize_ndvi(self, ndvi_map: Dict[str, Any]) -> Dict[str, Any]:
        # Convert NDVI to a quick RGB heatmap (0..1 range mapped to colors)
        arr = ndvi_map.get("ndvi")
        if arr is None:
            return {"image": None}
        ndvi = arr
        ndvi = (ndvi + 1.0) / 2.0
        ndvi = np.nan_to_num(ndvi, nan=0.5)
        ndvi = np.clip(ndvi, 0.0, 1.0)
        # Simple colormap: low=red, mid=yellow, high=green
        r = np.clip(2 * (1 - ndvi), 0, 1)
        g = np.clip(2 * ndvi, 0, 1)
        b = np.zeros_like(ndvi)
        rgb = np.stack([r, g, b], axis=-1).astype("float32")
        return {"image": rgb}

    def visualize_raster(self, raster_path: str) -> Dict[str, Any]:
        arr = load_raster(Path(raster_path))
        img = enhance_raster_with_current_settings(arr)
        return {"image": img}
