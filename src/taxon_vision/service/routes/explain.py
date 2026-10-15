# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Visual explanation endpoint."""

from __future__ import annotations

import io
import logging
import time
from typing import Annotated

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from taxon_vision.config import get_settings
from taxon_vision.inference.cam import generate_heatmap

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


class ExplainResponse(BaseModel):
    """Response containing CAM explanation generation status and latency.

    Attributes:
        heatmap_status: Operational generation status indicator (e.g. 'generated', 'cached').
        latency_ms: Overhead latency incurred during attribution calculation.
        heatmap_shape: Spatial dimensions [height, width] of the generated attribution map.
    """

    heatmap_status: str
    latency_ms: float
    heatmap_shape: list[int] | None = None


def _run_heatmap_generation(image_bytes: bytes | None) -> tuple[str, list[int]]:
    """Execute synchronous Class Activation Map generation in worker thread."""
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

    heatmap = generate_heatmap(image)
    return "generated", list(heatmap.shape)


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    file: Annotated[UploadFile | None, File()] = None,
) -> ExplainResponse:
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
        4. Returns attribution status, execution latency, and spatial heatmap dimensions.

    Args:
        file: Optional multipart uploaded image file.

    Returns:
        Explanation response indicating heatmap generation status, latency, and dimensions.
    """
    t0 = time.perf_counter()
    content: bytes | None = None
    if file is not None:
        content = await file.read()
        settings = get_settings()
        if len(content) > settings.service.max_image_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image upload exceeds maximum allowed size of {settings.service.max_image_upload_bytes} bytes.",
            )

    status_str, shape = await anyio.to_thread.run_sync(_run_heatmap_generation, content)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ExplainResponse(
        heatmap_status=status_str,
        latency_ms=round(elapsed_ms, 2),
        heatmap_shape=shape,
    )
