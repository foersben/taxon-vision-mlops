# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Visual explanation endpoint."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from taxon_vision.config import get_settings
from taxon_vision.inference.cam import SaliencyExtractionError, generate_cam_overlay, generate_heatmap
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


class ExplainResponse(BaseModel):
    """Response containing CAM explanation generation status and latency.

    Attributes:
        heatmap_status: Operational generation status indicator (e.g. 'generated', 'cached').
        latency_ms: Overhead latency incurred during attribution calculation.
        heatmap_shape: Spatial dimensions [height, width] of the generated attribution map.
        cam_heatmap_base64: Base64-encoded PNG image of the CAM saliency heatmap overlaid on the input image.
    """

    heatmap_status: str
    latency_ms: float
    heatmap_shape: list[int] | None = None
    cam_heatmap_base64: str | None = None


def _resolve_target_class_index(target_class_index: int | None) -> int | None:
    """Map external DarwinCore taxon ID or 0-indexed class index to classification head index.

    Why:
        Operators and HTMX buttons may submit either canonical DarwinCore taxon IDs
        (e.g. 48662) or linear output channel indices (0..K-1). This helper ensures
        both formats resolve to the appropriate linear head weight vector.

    Args:
        target_class_index: Optional submitted index or taxon identifier.

    Returns:
        Mapped 0-indexed class index, or None if unmapped.
    """
    if target_class_index is None:
        return None
    catalog = load_taxa_catalog()
    for idx, item in enumerate(catalog):
        if item.get("id") == target_class_index:
            return idx
    if 0 <= target_class_index < len(catalog):
        return target_class_index
    return None


def _run_heatmap_generation(
    image_bytes: bytes | None,
    target_class: int | None = None,
) -> tuple[str, list[int], str | None]:
    """Execute synchronous Class Activation Map generation in worker thread.

    Why:
        Forward-only CAM executes spatial projection and bilinear resizing synchronously.
        Offloading execution prevents blocking the asynchronous HTTP event loop thread.

    How:
        Decodes raw bytes into PIL Image or synthesizes a blank test canvas, invokes
        `generate_heatmap` and `generate_cam_overlay`, and catches `SaliencyExtractionError`
        to translate into HTTP 422.

    Complexity:
        O(1) forward-pass feature extraction plus O(H * W) bilinear interpolation.

    Args:
        image_bytes: Optional raw image buffer.
        target_class: Optional resolved target class index.

    Returns:
        Tuple of (status_string, spatial_shape, base64_overlay_png).

    Raises:
        HTTPException: If image cannot be decoded or saliency extraction fails.
    """
    if image_bytes is not None and len(image_bytes) > 0:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot decode upload as valid image for CAM: {exc}",
            ) from exc
    else:
        # Default probe image for headless health and smoke tests
        settings = get_settings()
        image = Image.new("RGB", (settings.model.image_size, settings.model.image_size), color="green")

    try:
        heatmap = generate_heatmap(image, target_class=target_class)
        overlay_b64 = generate_cam_overlay(image, target_class=target_class)
        return "generated", list(heatmap.shape), overlay_b64
    except SaliencyExtractionError as err:
        logger.warning("CAM generation failed: %s", err)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)) from err


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    target_class_index: int | None = None,
) -> Any:
    """Generate Class Activation Map (CAM) saliency visualization.

    Why:
        High-stakes biodiversity monitoring requires interpretable predictions so ecologists can verify
        that classifications are based on genuine morphometric organism characteristics (e.g. wing venation,
        dorsal markings) rather than background context artifacts (e.g. foliage, ruler bars, museum pins).
        As specified in Strategy Report Chapter 4 (§4.4), classical backward-pass Grad-CAM is prohibited
        due to sub-25ms latency limits and INT8 ONNX forward-only runtime constraints. Forward-hooked
        attribution provides low-latency interpretability.

    How:
        1. Reads optional uploaded observation image bytes asynchronously.
        2. Offloads forward-only CAM generation to an AnyIO worker thread to preserve event loop responsiveness.
        3. Projects terminal spatial activations with the predicted class linear head weights.
        4. If requested via HTMX (`HX-Request: true`), returns server-rendered HTML image fragment.
        5. Otherwise returns JSON `ExplainResponse` containing status, latency, dimensions, and base64 PNG.

    Complexity:
        O(1) async thread dispatch plus O(H * W) bilinear interpolation.

    Args:
        request: FastAPI HTTP request context.
        file: Optional multipart uploaded image file.
        target_class_index: Optional DarwinCore taxon ID or 0-indexed class index to explain.

    Returns:
        Explanation response indicating heatmap generation status, latency, dimensions, and base64 overlay.

    Raises:
        HTTPException: HTTP 413 for oversized payloads, 422 for undecodable uploads or CAM failures.
    """
    t0 = time.perf_counter()
    content: bytes | None = None
    if file is not None:
        content = await file.read()
        settings = get_settings()
        if content is not None and len(content) > settings.service.max_image_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image upload exceeds maximum allowed size of {settings.service.max_image_upload_bytes} bytes.",
            )

    resolved_class = _resolve_target_class_index(target_class_index)
    status_str, shape, overlay_b64 = await anyio.to_thread.run_sync(_run_heatmap_generation, content, resolved_class)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            request=request,
            name="partials/cam_overlay.html",
            context={
                "elapsed_ms": elapsed_ms,
                "overlay_b64": overlay_b64,
            },
        )

    return ExplainResponse(
        heatmap_status=status_str,
        latency_ms=round(elapsed_ms, 2),
        heatmap_shape=shape,
        cam_heatmap_base64=overlay_b64,
    )
