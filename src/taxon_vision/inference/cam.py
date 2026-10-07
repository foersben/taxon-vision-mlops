# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Low-latency CAM visual explanation generator."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from PIL import Image


def generate_heatmap(image: Image.Image) -> npt.NDArray[np.float32]:
    """Generates a mock/fast CAM saliency heatmap (< 20ms).

    Args:
        image: The image to generate a heatmap for.

    Returns:
        A numpy array representing the heatmap.
    """
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    heatmap = np.exp(-r / (w / 3.0))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return typing.cast(npt.NDArray[np.float32], heatmap.astype(np.float32))
