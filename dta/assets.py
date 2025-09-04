from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataAssetManager:
    """Resolves logical asset references to actual file paths.

    For now, this implementation only knows about the bundled Prithvi example
    assets so the end-to-end flow can run without extra setup.
    """

    project_root: Path

    def resolve(self, ref: str, *, options: dict[str, Any] | None = None) -> Path:
        if ref == "load/kahovka_raster":
            # Prefer a specific file if provided via options, otherwise default to first known tif
            base = self.project_root / "resources" / "kahovka_data"
            if options and "filename" in options:
                return base / str(options["filename"])
            # Default to a stable known file
            default = base / "hlsl_20230601.tif"
            return default
        if ref == "load/prithvi_example_raster":
            return (
                self.project_root
                / "resources"
                / "prithvi_eo_v1_100m"
                / "examples"
                / "HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
            )
        raise ValueError(f"Unknown asset reference: {ref}")
