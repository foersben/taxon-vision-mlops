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
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from taxon_vision.config import get_settings
from taxon_vision.inference.cam import (
    SaliencyExtractionError,
    encode_image_base64_png,
    generate_heatmap,
    overlay_cam_on_image,
)
from taxon_vision.service.inference import load_taxa_catalog
from taxon_vision.service.storage import get_image_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

HTMX_REQUEST_HEADER = "HX-Request"
"""str: Header set by HTMX on every request it issues."""


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


def _decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image.

    Args:
        image_bytes: Encoded image payload (JPEG, PNG, WebP, ...).

    Returns:
        The decoded image converted to RGB.

    Raises:
        HTTPException: HTTP 400 if the payload is not a decodable image.
    """
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot decode upload as valid image for CAM: {exc}",
        ) from exc


def _resolve_image_payload(upload_bytes: bytes | None, image_hash: str | None) -> bytes | None:
    """Select the image to explain, honouring the documented source priority.

    Why:
        API clients attach the file directly, whereas the HTMX dashboard cannot re-submit a consumed
        upload and instead references the server-side ephemeral cache by SHA-256 hash.

    How:
        1. A non-empty direct upload always wins.
        2. Otherwise a supplied ``image_hash`` is resolved against the ephemeral cache.
        3. With neither, None is returned and the caller falls back to the headless probe image.

    Args:
        upload_bytes: Bytes of a directly uploaded file, if any.
        image_hash: SHA-256 digest of a previously cached upload, if any.

    Returns:
        The image bytes to explain, or None when no payload was supplied.

    Raises:
        HTTPException: HTTP 422 if ``image_hash`` is given but unknown or expired.
    """
    if upload_bytes:
        return upload_bytes
    if image_hash is None:
        return None
    cached = get_image_cache().get(image_hash)
    if cached is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Observation image expired or unknown. Please upload the observation again.",
        )
    return cached


def _run_heatmap_generation(
    image_bytes: bytes | None,
    target_class: int | None = None,
) -> tuple[str, list[int], str]:
    """Execute synchronous Class Activation Map generation in a worker thread.

    Why:
        Forward-only CAM executes spatial projection and bilinear resizing synchronously.
        Offloading execution prevents blocking the asynchronous HTTP event loop thread.

    How:
        Decodes raw bytes into a PIL image (or synthesises a blank canvas for headless smoke tests),
        runs the model exactly once via `generate_heatmap`, blends the resulting map over the image and
        encodes it, translating `SaliencyExtractionError` into HTTP 422.

    Complexity:
        O(1) forward-pass feature extraction plus O(H * W) bilinear interpolation and blending.

    Args:
        image_bytes: Optional raw image buffer; None selects the probe image.
        target_class: Optional resolved target class index.

    Returns:
        Tuple of (status_string, [height, width] of the heatmap, base64 overlay PNG).

    Raises:
        HTTPException: If the image cannot be decoded or saliency extraction fails.
    """
    if image_bytes:
        image = _decode_image(image_bytes)
    else:
        # Default probe image for headless health and smoke tests
        size = get_settings().model.image_size
        image = Image.new("RGB", (size, size), color="green")

    try:
        heatmap = generate_heatmap(image, target_class=target_class)
    except SaliencyExtractionError as err:
        logger.warning("CAM generation failed: %s", err)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(err)) from err
    overlay = overlay_cam_on_image(image, heatmap)
    return "generated", list(heatmap.shape), encode_image_base64_png(overlay)


async def _read_upload(file: UploadFile | None) -> bytes | None:
    """Read an optional upload, enforcing the configured size limit.

    Args:
        file: Optional multipart upload.

    Returns:
        The uploaded bytes, or None if no file was supplied.

    Raises:
        HTTPException: HTTP 413 if the payload exceeds ``service.max_image_upload_bytes``.
    """
    if file is None:
        return None
    content = await file.read()
    limit = get_settings().service.max_image_upload_bytes
    if len(content) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image upload exceeds maximum allowed size of {limit} bytes.",
        )
    return content


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    image_hash: Annotated[str | None, Query()] = None,
    target_class_index: Annotated[int | None, Query()] = None,
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
        1. Resolves the image from a direct upload, or from the ephemeral cache via ``image_hash``.
        2. Offloads forward-only CAM generation to an AnyIO worker thread to preserve event loop responsiveness.
        3. Projects ViT patch tokens (stripped of prefix tokens, reshaped to the patch grid) with the
           target class linear-head weights.
        4. If requested via HTMX (``HX-Request: true``), returns a server-rendered HTML fragment. HTMX does
           not swap 4xx bodies, so failures are rendered as an inline alert fragment (HTTP 200) instead.
        5. Otherwise returns JSON `ExplainResponse` and propagates failures as proper HTTP errors.

    Complexity:
        O(1) async thread dispatch plus O(H * W) bilinear interpolation.

    Args:
        request: FastAPI HTTP request context.
        file: Optional multipart uploaded image file (API clients).
        image_hash: Optional SHA-256 of an image cached by the dashboard upload (HTMX clients).
        target_class_index: Optional DarwinCore taxon ID or 0-indexed class index to explain.

    Returns:
        Explanation response indicating heatmap generation status, latency, dimensions, and base64 overlay.

    Raises:
        HTTPException: HTTP 413 for oversized payloads, 400 for undecodable uploads, 422 for expired
            hashes or CAM failures (JSON clients only; HTMX clients receive an alert fragment).
    """
    t0 = time.perf_counter()
    is_htmx = request.headers.get(HTMX_REQUEST_HEADER) == "true"

    try:
        content = _resolve_image_payload(await _read_upload(file), image_hash)
        resolved_class = _resolve_target_class_index(target_class_index)
        status_str, shape, overlay_b64 = await anyio.to_thread.run_sync(
            _run_heatmap_generation, content, resolved_class
        )
    except HTTPException as err:
        if not is_htmx:
            raise
        return templates.TemplateResponse(
            request=request,
            name="partials/cam_error.html",
            context={"message": err.detail},
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if is_htmx:
        return templates.TemplateResponse(
            request=request,
            name="partials/cam_overlay.html",
            context={
                "elapsed_ms": elapsed_ms,
                "overlay_b64": overlay_b64,
                "heatmap_height": shape[0],
                "heatmap_width": shape[1],
            },
        )

    return ExplainResponse(
        heatmap_status=status_str,
        latency_ms=round(elapsed_ms, 2),
        heatmap_shape=shape,
        cam_heatmap_base64=overlay_b64,
    )
