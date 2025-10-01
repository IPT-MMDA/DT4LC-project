import io
import os

from google import genai
from google.genai import types
import numpy as np
import rasterio

from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding

MODEL = "gemini-2.0-flash-exp"  # Use flash-exp for better availability


def _get_client() -> genai.Client:
    """Lazy initialization of Gemini client.

    Returns:
        Gemini client

    Raises:
        ValueError: If GEMINI_API_KEY is not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. Set it via: export GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


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


def _image_part(att: Attachment) -> types.Part:
    if att.mime_type.lower() in {"image/tiff", "image/tif"}:
        png_bytes = _tiff_to_png_bytes(att.path)
        return types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    # JPEG/PNG/WebP are fine inline under ~20 MB
    with open(att.path, "rb") as fh:
        return types.Part.from_bytes(data=fh.read(), mime_type=att.mime_type)


def analyze(req: ChatRequest, registry_types: list[str]) -> ContextUnderstanding:
    """Analyze user request and extract structured context.

    Args:
        req: User chat request
        registry_types: Available types from registry

    Returns:
        Structured context understanding

    Raises:
        ValueError: If GEMINI_API_KEY is not set
    """
    client = _get_client()

    parts: list[types.Part | str] = [f"[REGISTRY_TYPES]={registry_types}", SYS, req.prompt]
    for att in req.attachments:
        if att.mime_type.startswith("image/"):
            parts.append(_image_part(att))
    resp = client.models.generate_content(
        model=MODEL,
        contents=parts,
        # TODO: disable "thinking" for speed/budget
        # generation_config=types.GenerationConfig(thinking={'thinking_budget': 0})
    )
    # let Gemini produce a small JSON block; simple parse:
    import json
    import re

    m = re.search(r"\{.*\}", resp.text, re.S)
    data = (
        json.loads(m.group(0))
        if m
        else {"goal": req.prompt, "desired_outputs": [], "required_inputs": [], "hints": {"keywords": []}}
    )
    return ContextUnderstanding(**data)
