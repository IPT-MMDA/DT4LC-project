import io
from typing import Any

import numpy as np
import rasterio

from dta.dti.coe.llm import LLMMessage, LLMRouter
from dta.dti.coe.llm.config import create_router_from_env
from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding

# Lazy router initialization
_router: LLMRouter | None = None


def _get_router() -> LLMRouter:
    """Lazy initialization of LLM router.

    Returns:
        Configured LLM router with fallback
    """
    global _router
    if _router is None:
        _router = create_router_from_env()
    return _router


SYS = (
    "You are a Context Understanding Agent for a geospatial DT. "
    "Extract: (1) goal (2) desired output types from registry types "
    "(3) required input types (4) keywords. "
    "Be concise and return JSON with keys goal, desired_outputs, required_inputs, hints.keywords."
)


def _tiff_to_png_bytes(tif_path: str) -> bytes:
    # Minimal RGB preview from a multi-band GeoTIFF
    # Prefer rasterio if available; else fallback to PIL
    try:
        with rasterio.open(tif_path) as src:
            # Heuristic: use bands 3,2,1 (or 4,3,2) if available; else first 3
            pick = [3, 2, 1] if src.count >= 3 else [1, 1, 1]
            bands = []
            for b in pick:
                b = min(b, src.count)
                x = src.read(b).astype("float32")
                # simple min-max to 0..255 for preview
                lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
                x = np.clip((x - lo) / max(1e-6, (hi - lo)), 0, 1) * 255
                bands.append(x.astype("uint8"))
            rgb = np.dstack(bands)
        from PIL import Image

        im = Image.fromarray(rgb)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # Fallback: PIL-only, assumes 3-band compatible TIFF
        from PIL import Image

        im = Image.open(tif_path).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _image_part(att: Attachment) -> Any:
    """Convert attachment to image part.

    TODO: Implement multimodal support for LLM router.
    For now, returns None as we handle text-only.
    """
    # TODO: Support images in router
    return None


def analyze(req: ChatRequest, registry_types: list[str]) -> ContextUnderstanding:
    """Analyze user request and extract structured context.

    Args:
        req: User chat request
        registry_types: Available types from registry

    Returns:
        Structured context understanding

    Raises:
        Exception: If all LLM providers fail
    """
    router = _get_router()

    # Build prompt with registry types and system instructions
    system_msg = f"{SYS}\n\n[REGISTRY_TYPES]={registry_types}"
    user_msg = req.prompt

    # TODO: Handle image attachments for multimodal providers
    # For now, just use text

    messages = [LLMMessage(role="system", content=system_msg), LLMMessage(role="user", content=user_msg)]

    # Generate with router (will try Gemini, fallback to Ollama)
    response = router.generate(messages, temperature=0.3)  # Lower temp for structured output

    # Parse JSON from response
    import json
    import re

    m = re.search(r"\{.*\}", response.text, re.S)
    data = (
        json.loads(m.group(0))
        if m
        else {"goal": req.prompt, "desired_outputs": [], "required_inputs": [], "hints": {"keywords": []}}
    )
    return ContextUnderstanding(**data)
