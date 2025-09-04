from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from cognitive_ui.sample_model import model as prithvi_mae_model


Prediction = Dict[str, Any]


@dataclass
class ModelRegistry:
    """Resolves logical model identifiers to callables.

    The registry keeps adapters thin so we can reuse the sample Prithvi model
    bundled with the project.
    """

    def get(self, ref: str) -> Callable[..., Any]:
        if ref == "models/prithvi_features":
            # Return a callable that computes features via Prithvi encoder
            def extract_features(raster_path: str) -> Prediction:
                # For now, we do not perform a true forward pass; we mimic output
                # to keep the scaffolding runnable without GPU.
                _ = raster_path
                return {"features": "prithvi_mae_encoder_output_stub"}

            return extract_features
        raise ValueError(f"Unknown model reference: {ref}")
