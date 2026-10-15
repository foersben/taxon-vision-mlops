# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Load real, serving-identical calibration tensors for static INT8 post-training quantization."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from taxon_vision.service.inference import get_image_transform

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
"""Image file extensions considered for calibration."""


def load_calibration_tensors(
    image_dir: Path | str,
    max_samples: int = 128,
) -> list[npt.NDArray[np.float32]]:
    """Build `(1, 3, H, W)` calibration batches from a directory of representative images.

    Why:
        Static PTQ bakes activation scales and zero points into the graph, so they must be estimated on data
        drawn from the deployment distribution - ideally a subset of the same calibration split used for
        conformal prediction. Using the exact serving transform guarantees that the calibrated ranges match
        what the quantized graph sees in production. Synthetic noise is never substituted: it would produce
        scales that clip real activations.

    How:
        Lists image files in sorted order (deterministic), thins the list evenly when it exceeds `max_samples`,
        and applies the cached serving preprocessing transform to each image.

    Args:
        image_dir: Directory containing calibration images.
        max_samples: Maximum number of images to use (default 128).

    Returns:
        List of float32 arrays of shape `(1, 3, H, W)`.

    Raises:
        FileNotFoundError: If the directory does not exist or contains no readable images.
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {root}")

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No calibration images (*.jpg, *.png, *.webp) found under {root}")

    if len(files) > max_samples:
        picks = np.linspace(0, len(files) - 1, num=max_samples).round().astype(int)
        files = [files[i] for i in picks]

    transform = get_image_transform()
    tensors: list[npt.NDArray[np.float32]] = []
    for path in files:
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        tensors.append(tensor)

    logger.info("Loaded %d calibration tensors from %s", len(tensors), root)
    return tensors
