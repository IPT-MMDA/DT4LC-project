import io
from typing import Any

import numpy as np
import rasterio

from dta.dti.coe.llm import LLMMessage, get_llm_router
from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding

SYS = (
    "You are a Context Understanding Agent for a geospatial Digital Twin. "
    "Extract structured information from the user's request.\n\n"
    "Return ONLY valid JSON with this EXACT structure:\n"
    "{\n"
    '  "goal": "brief description of what user wants to accomplish",\n'
    '  "desired_outputs": ["OutputType1"],\n'
    '  "required_inputs": ["InputType1"],\n'
    '  "hints": {"keywords": ["keyword1", "keyword2"]}\n'
    "}\n\n"
    "CRITICAL RULES:\n"
    "- desired_outputs MUST be an array of TYPE NAME STRINGS only\n"
    "- required_inputs MUST be an array of TYPE NAME STRINGS only\n"
    "- Do NOT use objects or nested structures for these arrays\n"
    "- Only use types from the registry list provided below\n"
    "- ONLY include outputs that the user EXPLICITLY requests\n"
    "- If unsure, use empty arrays []\n\n"
    "DOMAIN KNOWLEDGE - Match user intent to SINGLE output type:\n"
    '- "field boundaries", "parcels", "delineate", "agricultural plots" → ["FieldBoundaries"]\n'
    '- "vegetation health", "greenness", "ndvi" → ["NDVIMap"]\n'
    '- "water index", "water detection", "ndwi" → ["NDWIMap"]\n'
    '- "snow index", "ndsi", "snow detection" → ["NDSIMap"]\n'
    '- "land cover", "land use", "lulc", "classify" → ["LULCMap"]\n'
    '- "statistics", "distribution", "histogram" → ["Statistics"]\n'
    '- "change detection", "compare images", "before/after" → ["ChangeMap"]\n'
    '- "vegetation change", "ndvi change" → ["ChangeMap"] with hints.index_type="ndvi"\n'
    '- "snow change", "ndsi change", "glacier change" → ["ChangeMap"] with hints.index_type="ndsi"\n'
    '- "water change", "ndwi change", "flood detection" → ["ChangeMap"] with hints.index_type="ndwi"\n'
    '- "prithvi", "features", "embeddings", "foundation model" → ["Features"]\n\n'
    "IMPORTANT: Do NOT add multiple outputs unless user explicitly asks for multiple analyses.\n"
    '"Detect parcels" → ONLY ["FieldBoundaries"], NOT ["FieldBoundaries", "Features"]\n'
    '"Calculate NDVI" → ONLY ["NDVIMap"], NOT ["NDVIMap", "Features"]'
)


def _tiff_to_png_bytes(tif_path: str) -> bytes:
    """Convert GeoTIFF to PNG preview image.

    Uses rasterio for multi-band GeoTIFFs with proper band selection,
    falls back to PIL for simple RGB conversion on any error.

    Args:
        tif_path: Path to GeoTIFF file

    Returns:
        PNG image as bytes
    """
    from PIL import Image

    try:
        with rasterio.open(tif_path) as src:
            # Use bands 3,2,1 for RGB if available, otherwise use first band repeated
            band_indices = [3, 2, 1] if src.count >= 3 else [1, 1, 1]
            bands = []
            for band_num in band_indices:
                actual_band = min(band_num, src.count)
                data = src.read(actual_band).astype("float32")
                # Percentile stretch to 0-255 for preview
                lo, hi = np.nanpercentile(data, 2), np.nanpercentile(data, 98)
                normalized = np.clip((data - lo) / max(1e-6, (hi - lo)), 0, 1) * 255
                bands.append(normalized.astype("uint8"))
            rgb = np.dstack(bands)

        im = Image.fromarray(rgb)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    except (rasterio.errors.RasterioIOError, OSError):
        # Fallback: PIL-only for simple RGB TIFFs
        im = Image.open(tif_path).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _image_part(att: Attachment) -> Any:
    """Convert attachment to image part for multimodal LLM providers.

    Note: Multimodal image analysis is not yet implemented.
    Context understanding currently relies on text prompts and file metadata.

    Args:
        att: File attachment with path to image

    Returns:
        None (image content not currently used in context analysis)
    """
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
    router = get_llm_router()

    # Build prompt with registry types and system instructions
    system_msg = f"{SYS}\n\n[REGISTRY_TYPES]={registry_types}"
    user_msg = req.prompt

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
