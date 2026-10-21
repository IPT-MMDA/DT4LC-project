import io
import logging
from typing import Any

import numpy as np
from pydantic import ValidationError
import rasterio

from dta.dti.coe.llm import LLMMessage, get_llm_router
from dta.dti.coe.triggers import get_trigger_index
from dta.dti.registry import load_registry
from dta.dti.schemas import Attachment, ChatRequest, ContextUnderstanding, Registry

logger = logging.getLogger(__name__)

# Static portions of the system prompt — generic instruction format that
# doesn't depend on which algorithms exist. The DOMAIN KNOWLEDGE bullet list
# is rendered separately from the registry in `_render_domain_knowledge()`.
_SYS_PREAMBLE = (
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
)
_SYS_EPILOGUE = (
    "\n\nIMPORTANT: Do NOT add multiple outputs unless user explicitly asks for multiple analyses.\n"
    '"Detect parcels" → ONLY ["FieldBoundaries"], NOT ["FieldBoundaries", "Features"]\n'
    '"Calculate NDVI" → ONLY ["NDVIMap"], NOT ["NDVIMap", "Features"]'
)


def _render_domain_knowledge(registry: Registry) -> str:
    """Render the DOMAIN KNOWLEDGE bullet list from registry triggers + outputs.

    Each user-runnable item becomes one bullet of the form
    ``- "kw1", "kw2", "kw3" → ["OutputType"]``. The change-detection item
    additionally renders one bullet per ``config.index_keyword_map`` entry
    so the LLM learns the per-index hint mapping.
    """
    lines: list[str] = []
    for item in registry.instances:
        if item.kind in ("input", "preprocessor", "postprocess"):
            continue
        if item.id == "algorithms/change-detection":
            continue  # rendered specially below
        if item.triggers is None or not item.outputs:
            continue
        # Take the first 4 keywords for prompt brevity; quote each.
        kws = [f'"{k}"' for k in item.triggers.keywords[:4]]
        outputs = "[" + ", ".join(f'"{o}"' for o in item.outputs[:1]) + "]"
        lines.append(f"- {', '.join(kws)} → {outputs}")

    # Change-detection: a base line + one per index_keyword_map entry.
    cd_item = next((it for it in registry.instances if it.id == "algorithms/change-detection"), None)
    if cd_item is not None:
        if cd_item.triggers is not None:
            base_kws = [f'"{k}"' for k in cd_item.triggers.keywords[:4]]
            lines.append(f'- {", ".join(base_kws)} → ["ChangeMap"]')
        index_map = (cd_item.config or {}).get("index_keyword_map", {})
        for index_name, kws in index_map.items():
            quoted = [f'"{k}"' for k in kws[:4]]
            lines.append(f'- {", ".join(quoted)} → ["ChangeMap"] with hints.index_type="{index_name}"')
    return "\n".join(lines)


def _build_sys_prompt(registry: Registry, registry_types: list[str]) -> str:
    """Compose the full system prompt: preamble + registry-rendered knowledge + epilogue."""
    return f"{_SYS_PREAMBLE}{_render_domain_knowledge(registry)}{_SYS_EPILOGUE}\n\n[REGISTRY_TYPES]={registry_types}"


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


def _image_part(att: Attachment) -> str | None:
    """Convert attachment to base64 PNG thumbnail for multimodal LLM providers.

    Args:
        att: File attachment with path to image

    Returns:
        Base64-encoded PNG string (512x512 max), or None if conversion fails
    """
    import base64

    from PIL import Image

    if att.path is None or not att.mime_type.startswith("image/"):
        return None
    try:
        png_bytes = _tiff_to_png_bytes(att.path)
        im = Image.open(io.BytesIO(png_bytes))
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _accept(data: Any, registry_types: list[str]) -> ContextUnderstanding | None:
    """Structural check on a parsed context reply.

    Accepts a reply that fits the ContextUnderstanding schema, has a non-blank
    goal, names only registry types and carries string keywords; extra keys are
    ignored. Logs the reason and returns None otherwise.
    """
    try:
        ctx: ContextUnderstanding = ContextUnderstanding.model_validate(data)
    except ValidationError as e:
        reason = f"does not match the ContextUnderstanding schema: {e.error_count()} error(s)"
    else:
        unknown = [t for t in ctx.desired_outputs + ctx.required_inputs if t not in registry_types]
        keywords = ctx.hints.get("keywords", [])
        if not ctx.goal.strip():
            reason = "blank goal"
        elif unknown:
            reason = f"types not in the registry: {unknown}"
        elif not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            reason = "hints.keywords is not a list of strings"
        else:
            return ctx
    logger.warning(f"Context agent reply rejected: {reason}")
    return None


def _deterministic_context(req: ChatRequest) -> ContextUnderstanding:
    """Context from the registry triggers, used when no LLM reply is usable.

    Two or more attachments, or a prompt matching a ChangeMap producer, ask for a
    ChangeMap from before/after rasters with no keywords, so the planner takes the
    index type from the goal. Another matched item gives its first output, its
    inputs and its trigger keywords. No match keeps only the uploaded raster.
    """
    item = get_trigger_index().find_matching_item(req.prompt)
    if len(req.attachments) >= 2 or (item is not None and "ChangeMap" in item.outputs):
        return ContextUnderstanding(
            goal=req.prompt,
            desired_outputs=["ChangeMap"],
            required_inputs=["RasterPathBefore", "RasterPathAfter"],
            hints={"keywords": []},
        )
    if item is not None:
        return ContextUnderstanding(
            goal=req.prompt,
            desired_outputs=item.outputs[:1],
            required_inputs=list(item.inputs),
            hints={"keywords": list(item.triggers.keywords) if item.triggers else []},
        )
    return ContextUnderstanding(
        goal=req.prompt,
        desired_outputs=[],
        required_inputs=["RasterPath"] if req.attachments else [],
        hints={"keywords": []},
    )


def analyze(req: ChatRequest, registry_types: list[str]) -> ContextUnderstanding:
    """Analyze user request and extract structured context.

    Falls back to the deterministic context when every LLM provider fails or
    the reply is not accepted, so this never raises for LLM reasons.

    Args:
        req: User chat request
        registry_types: Available types from registry

    Returns:
        Structured context understanding
    """
    router = get_llm_router()

    # The SYS prompt's DOMAIN KNOWLEDGE block is derived from each registry
    # item's triggers + outputs + (for change-detection) config — adding a
    # new item to registry.yaml updates this prompt automatically.
    registry = load_registry()
    system_msg = _build_sys_prompt(registry, registry_types)
    user_msg = req.prompt

    vision_available = any(p.supports_images for p in router.providers)
    image_b64_list: list[str] = []
    if vision_available:
        for att in req.attachments:
            b64 = _image_part(att)
            if b64 is not None:
                image_b64_list.append(b64)

    messages = [
        LLMMessage(role="system", content=system_msg),
        LLMMessage(role="user", content=user_msg, images=image_b64_list or None),
    ]

    # Generate with router (will try Gemini, fallback to Ollama)
    try:
        response = router.generate(messages, temperature=0.3)  # Lower temp for structured output
    except Exception as e:
        logger.warning(f"Context agent LLM call failed ({e}); using deterministic context")
        return _deterministic_context(req)

    # Parse JSON from response
    import json
    import re

    m = re.search(r"\{.*\}", response.text, re.S)
    if not m:
        logger.warning("Context agent reply contains no JSON object; using deterministic context")
        return _deterministic_context(req)
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"Context agent returned malformed JSON ({e}); using deterministic context")
        return _deterministic_context(req)
    ctx = _accept(data, registry_types)
    if ctx is None:
        logger.warning("Context agent reply failed validation; using deterministic context")
        return _deterministic_context(req)
    return ctx
