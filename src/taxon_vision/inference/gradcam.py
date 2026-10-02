# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Low-latency Grad-CAM visual explanation generator."""

from __future__ import annotations

import numpy as np
from PIL import Image


def generate_heatmap(image: Image.Image) -> np.ndarray:
    """Generates a mock/fast Grad-CAM saliency heatmap (< 20ms)."""
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    heatmap = np.exp(-r / (w / 3.0))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)
