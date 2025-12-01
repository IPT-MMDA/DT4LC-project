"""Delineate-Anything model wrapper.

Wrapper for the Delineate-Anything field boundary detection model.
See: https://github.com/IPT-MMDA/Delineate-Anything
"""

from __future__ import annotations

import logging
from pathlib import Path
import random
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class DelineateAnythingModel:
    """Delineate-Anything field boundary detection model wrapper.

    Uses YOLO-based segmentation for agricultural field boundary detection.
    Outputs GeoPackage files with polygon geometries and area statistics.

    Falls back to stub mode if dependencies are not available.
    """

    def __init__(self, model_variant: str = "small") -> None:
        """Initialize Delineate-Anything model.

        Args:
            model_variant: Model variant - "small" (faster) or "large" (more accurate)
        """
        self._name = "delineate-anything"
        self._version = "v1.0"
        self._model_variant = model_variant
        self._deps_available: bool | None = None

    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @property
    def version(self) -> str:
        """Model version."""
        return self._version

    @property
    def required_inputs(self) -> list[str]:
        """Required input types."""
        return ["Raster"]

    @property
    def outputs(self) -> list[str]:
        """Output types."""
        return ["GeoPackage", "FieldBoundaries"]

    def _check_dependencies(self) -> bool:
        """Check if full model dependencies are available.

        Returns:
            True if all dependencies are installed
        """
        if self._deps_available is not None:
            return self._deps_available

        try:
            import geopandas  # noqa: F401
            import rasterio  # noqa: F401
            import shapely  # noqa: F401
            import ultralytics  # noqa: F401

            self._deps_available = True
        except ImportError as e:
            logger.debug(f"Delineate-Anything dependency missing: {e}")
            self._deps_available = False

        return self._deps_available

    def is_available(self) -> bool:
        """Check if model is available.

        Returns:
            True if dependencies are installed
        """
        return self._check_dependencies()

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run field boundary detection.

        Args:
            inputs: Input dictionary with 'raster_path'

        Returns:
            Dictionary with detection results
        """
        raster_path = inputs.get("raster_path")
        if not raster_path:
            raise ValueError("raster_path is required")

        # Check if full dependencies are available
        if self._check_dependencies():
            return self._predict_with_model(inputs)
        else:
            return self._predict_stub(inputs)

    def _predict_with_model(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run actual model prediction when dependencies are available.

        Args:
            inputs: Input dictionary with 'raster_path'

        Returns:
            Dictionary with detection results
        """
        from .third_party.delineate_anything import delineate_fields

        raster_path = inputs.get("raster_path")
        if not raster_path:
            raise ValueError("raster_path is required")

        output_path = inputs.get("output_path")
        config = inputs.get("config")

        result = delineate_fields(
            raster_path=str(raster_path),
            output_path=str(output_path) if output_path else None,
            model=self._model_variant,
            config=config,
        )

        return result

    def _predict_stub(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Stub prediction when dependencies are not available.

        Generates realistic demo output for field boundary detection.

        Args:
            inputs: Input dictionary with 'raster_path'

        Returns:
            Stub detection results
        """
        logger.info("Running Delineate-Anything in stub mode (demo)")

        raster_path = inputs.get("raster_path", "input.tif")
        output_path = inputs.get("output_path")

        if not output_path:
            output_dir = Path(tempfile.gettempdir()) / "dt4lc_delineate"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{Path(raster_path).stem}_fields.gpkg")

        # Generate realistic demo results
        num_fields = random.randint(5, 15)
        total_area = sum(random.uniform(2500, 50000) for _ in range(num_fields))

        return {
            "output_path": output_path,
            "num_fields": num_fields,
            "total_area_m2": round(total_area, 2),
            "crs": "EPSG:32633",
            "model": self._model_variant,
            "mode": "stub",
            "fields": [
                {
                    "id": i + 1,
                    "area_m2": round(random.uniform(2500, 50000), 2),
                    "perimeter_m": round(random.uniform(200, 1000), 2),
                }
                for i in range(num_fields)
            ],
        }

    def get_missing_requirements(self) -> list[str]:
        """Get list of missing requirements.

        Returns:
            List of missing package names
        """
        missing = []

        try:
            import ultralytics  # noqa: F401
        except ImportError:
            missing.append("ultralytics")

        try:
            import rasterio  # noqa: F401
        except ImportError:
            missing.append("rasterio")

        try:
            import geopandas  # noqa: F401
        except ImportError:
            missing.append("geopandas")

        try:
            import shapely  # noqa: F401
        except ImportError:
            missing.append("shapely")

        try:
            import cv2  # noqa: F401
        except ImportError:
            missing.append("opencv-python")

        return missing
