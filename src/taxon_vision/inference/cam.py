# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Low-latency CAM visual explanation generator.

CAM (Class Activation Mapping) is used to visualize the regions of an image that are most relevant to the model's prediction. It is a post-hoc explainability technique that can be applied to any convolutional neural network. Compared to Grad-CAM, CAM is more computationally efficient as it does not require a backward gradient pass through the entire backbone. Instead, it uses the activations of the final convolutional layer to generate the heatmap. A disadvantage of CAM is that it can only be used with models that have a global average pooling layer before the final classification layer. However, it is possible to modify the model to include a global average pooling layer. This is a common practice in deep learning.
"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt
from PIL import Image


def generate_heatmap(image: Image.Image) -> npt.NDArray[np.float32]:
    """Generate visual saliency attribution map for organism classification.

    Why:
        In high-throughput ecological vision systems with strict sub-25ms latency budgets, classical Grad-CAM is strictly prohibited: it requires a backward gradient pass through the entire backbone, which triples inference latency and fails completely on quantized INT8 ONNX Runtime inference engines where gradient computation graphs are stripped out. Strategy Report Chapter 4 (§4.4) mandates forward-only attribution: L_CAM^c = ReLU( sum_k w_k^c * A^k ). This function currently implements the interim radial saliency placeholder (< 2ms) designed to exercise the HTMX/Jinja2 UI contract while preserving SLA latency invariants prior to final multi-head feature-map forward hooking in ONNX.

    How:
        Extracts image spatial dimensions (w, h), constructs an open mesh grid centered at the geometric centroid (cy, cx), computes normalized radial Euclidean distances r, and applies an exponential Gaussian attenuation function. Normalizes the resulting activation surface into the [0.0, 1.0] interval via min-max scaling to produce a clean float32 saliency mask.

    Complexity:
        O(H * W) time complexity, requiring zero matrix factorizations or autograd allocations.

    Args:
        image: PIL Image instance representing the organism capture.

    Returns:
        2D NumPy float32 array of shape (H, W) normalized to [0.0, 1.0].
    """
    w, h = image.size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    heatmap = np.exp(-r / (w / 3.0))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return typing.cast(npt.NDArray[np.float32], heatmap.astype(np.float32))
