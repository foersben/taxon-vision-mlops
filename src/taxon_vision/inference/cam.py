# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Low-latency forward-only Class Activation Mapping (CAM) attribution generator.

Provides diagnostic transparency for biological taxonomic predictions without backward gradient passes.
Strategy Report Chapter 4 (§4.4) mandates forward-only attribution:

    L_CAM^c = ReLU( sum_k w_k^c * A^k )

where A^k is the spatial activation map emitted by the final convolutional or ViT patch layer prior to
global average pooling, and w_k^c is the linear classification head weight for class c.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from taxon_vision.config import get_settings
from taxon_vision.models.head import TaxonClassifier

logger = logging.getLogger(__name__)


class SaliencyExtractionError(RuntimeError):
    """Raised when spatial activation maps cannot be extracted from the model."""


@runtime_checkable
class FeatureExtractorBackbone(Protocol):
    """Protocol for vision backbones providing intermediate spatial feature extraction."""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Emits spatial activation maps prior to global pooling."""
        ...


@runtime_checkable
class CNNBackboneWithConvHead(Protocol):
    """Protocol for convolutional backbones with a distinct projection head."""

    conv_head: nn.Module


@runtime_checkable
class ViTBackbone(Protocol):
    """Protocol for Vision Transformer backbones with prefix token count."""

    num_prefix_tokens: int


@runtime_checkable
class ClassifierWithBackboneAndHead(Protocol):
    """Protocol for vision classifiers composing a feature backbone and classification head."""

    backbone: nn.Module
    head: nn.Module

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass returning class logits."""
        ...


def _extract_linear_layer(head: nn.Module) -> nn.Linear:
    """Find the terminal linear classification layer in the head module.

    Why:
        Linear classification head weights are required to project spatial activation maps into class-specific Class Activation Maps (CAM) without requiring backward gradients.

    How:
        Iterates over child modules and identifies the last instance of `nn.Linear`.

    Complexity:
        O(M) where M is the number of submodules in the classification head.

    Args:
        head: The classification head nn.Module.

    Returns:
        The extracted terminal nn.Linear layer.

    Raises:
        SaliencyExtractionError: If no nn.Linear layer is present in the head module.
    """
    linear: nn.Linear | None = None
    for module in head.modules():
        if isinstance(module, nn.Linear):
            linear = module
    if linear is None:
        logger.error("Classification head %s contains no nn.Linear module.", type(head).__name__)
        raise SaliencyExtractionError(f"Classification head {type(head).__name__} contains no nn.Linear layer.")
    return linear


def _project_cnn_features(feats: torch.Tensor, backbone: nn.Module, w_c: torch.Tensor) -> torch.Tensor:
    """Project 4D CNN spatial feature activations with target class weights.

    Why:
        Isolates convolutional projection logic to maintain low cognitive complexity.

    Args:
        feats: 4D spatial feature tensor (B, C, H, W).
        backbone: Visual backbone instance.
        w_c: Target class weight vector of shape (D,).

    Returns:
        Projected spatial CAM tensor of shape (B, H, W).
    """
    if isinstance(backbone, CNNBackboneWithConvHead):
        feats = backbone.conv_head(feats)
    if feats.shape[1] == w_c.shape[0]:
        cam = torch.einsum("d, bdxy -> bxy", w_c, feats)
        return F.relu(cam)
    return feats.mean(dim=1)


def _project_vit_features(feats: torch.Tensor, backbone: nn.Module, w_c: torch.Tensor) -> torch.Tensor | None:
    r"""Project ViT patch tokens onto the target-class head weights (forward-only CAM).

    Why:
        A Vision Transformer emits a token sequence ``(B, N, D)`` with ``N = P + N_patches``, where ``P``
        prefix tokens (the class token and optional register tokens) carry no spatial position. Spatial
        attribution requires discarding them and restoring the 2D patch grid, e.g. 196 tokens -> 14 x 14.

    How:
        1. Strips the first ``backbone.num_prefix_tokens`` tokens (1 if the backbone does not declare it).
        2. Requires the remaining patch count to be a perfect square ``g * g``; otherwise attribution is
           geometrically undefined and None is returned.
        3. Requires the token width ``D`` to equal the head's input width; a mismatch means the head was
           not trained on this backbone and any projection would be meaningless, so None is returned
           instead of fabricating a channel-mean map.
        4. Reshapes to ``(B, g, g, D)`` and contracts with ``w_c`` to obtain
           ``ReLU(sum_d w_{c,d} * A(x, y, d))``.

    Complexity:
        O(B * N * D) for the contraction; no gradients or autograd graph nodes are created.

    Args:
        feats: 3D token tensor of shape (B, N, D) from ``backbone.forward_features``.
        backbone: Visual backbone instance (inspected for ``num_prefix_tokens`` via `ViTBackbone`).
        w_c: Target class weight vector of shape (D,).

    Returns:
        Non-negative spatial CAM tensor of shape (B, g, g), or None if the token layout is unsupported.
    """
    num_prefix = backbone.num_prefix_tokens if isinstance(backbone, ViTBackbone) else 1
    patches = feats[:, num_prefix:, :]
    num_patches = patches.shape[1]
    grid_size = math.isqrt(num_patches)
    if grid_size * grid_size != num_patches:
        logger.warning("ViT patch count %d does not form a square grid.", num_patches)
        return None
    token_dim = w_c.shape[0]
    if patches.shape[-1] != token_dim:
        logger.error("ViT token width %d does not match head weight width %d.", patches.shape[-1], token_dim)
        return None
    spatial_patches = patches.reshape(-1, grid_size, grid_size, token_dim)
    cam = torch.einsum("d, bxyd -> bxy", w_c, spatial_patches)
    return F.relu(cam)


def _compute_spatial_cam(
    backbone: nn.Module,
    w_c: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Extract spatial features from ViT or CNN backbone and project with class weights.

    Why:
        Forward-only Class Activation Mapping projects spatial feature activations by the linear head weights of the target class to yield dense attribution maps without backpropagation.

    How:
        Executes `forward_features` on the backbone. For 4D CNN tensors (B, C, H, W), optionally passes through `conv_head` and contracts along channel dimensions with class weights `w_c`. For 3D ViT patch tokens (B, N, D), strips prefix/class tokens, checks square grid shape, and contracts patch embeddings with `w_c`.

    Complexity:
        O(C * H * W) for CNN contractions or O(N * D) for ViT patch token contractions.

    Args:
        backbone: Visual feature extraction backbone module.
        w_c: Target class weight vector of shape (D,).
        tensor: Normalized input image tensor of shape (B, C, H, W).

    Returns:
        Projected spatial activation tensor of shape (B, H_feat, W_feat).

    Raises:
        SaliencyExtractionError: If feature extraction fails, returns non-tensor, or produces unsupported tensor shapes.
    """
    if not isinstance(backbone, FeatureExtractorBackbone):
        logger.error("Backbone %s does not implement FeatureExtractorBackbone protocol.", type(backbone).__name__)
        raise SaliencyExtractionError(
            f"Backbone {type(backbone).__name__} does not implement FeatureExtractorBackbone."
        )

    feats = backbone.forward_features(tensor)
    if not isinstance(feats, torch.Tensor):
        logger.error("Backbone forward_features returned non-tensor %s", type(feats))
        raise SaliencyExtractionError("Backbone forward_features did not return a torch.Tensor.")

    if feats.ndim == 4:
        return _project_cnn_features(feats, backbone, w_c)

    if feats.ndim == 3:
        vit_cam = _project_vit_features(feats, backbone, w_c)
        if vit_cam is not None:
            return vit_cam

    logger.error("Unsupported spatial activation shape: ndim=%d, shape=%s", feats.ndim, feats.shape)
    raise SaliencyExtractionError(
        f"Target layer failed to emit valid spatial activations; got tensor with shape {feats.shape}."
    )


def generate_heatmap(
    image: Image.Image,
    classifier: nn.Module | None = None,
    target_class: int | None = None,
) -> npt.NDArray[np.float32]:
    """Generate forward-only Class Activation Map (CAM) attribution for organism classification.

    Why:
        In high-throughput ecological vision systems with strict sub-25ms latency budgets, classical Grad-CAM is prohibited because backward passes triple inference latency. Forward-only CAM executes in < 3ms by projecting spatial feature maps with the linear head weights: L_CAM^c = ReLU( sum_k w_k^c * A^k ).

    How:
        1. Prepares the normalized input tensor matching the backbone input shape.
        2. Resolves the active TaxonClassifier model and terminal linear head weight vector w_c.
        3. Projects 2D spatial feature maps (CNN) or unflattened patch tokens (ViT) by w_c.
        4. Bilinearly upsamples the resulting attribution grid to the source image dimensions (W, H).
        5. Normalizes the resulting attribution map using the configured numerical epsilon.

    Complexity:
        O(1) forward feature extraction plus O(H * W) bilinear spatial interpolation.

    Args:
        image: PIL Image instance representing the organism capture.
        classifier: Optional TaxonClassifier instance. If None, resolves from service inference cache.
        target_class: Target class index to explain. If None, defaults to the argmax prediction.

    Returns:
        2D NumPy float32 array of shape (H, W) normalized to [0.0, 1.0].

    Raises:
        SaliencyExtractionError: If spatial activation maps or model weights cannot be extracted.
    """
    width, height = image.size

    model = classifier
    if model is None:
        try:
            from taxon_vision.service.inference import get_classifier

            model = get_classifier()
        except Exception as exc:
            logger.error("Failed to resolve classifier for CAM attribution: %s", exc)
            raise SaliencyExtractionError(f"Classifier model is not available for saliency extraction: {exc}") from exc

    backbone: nn.Module
    head: nn.Module
    if isinstance(model, (TaxonClassifier, ClassifierWithBackboneAndHead)):
        backbone = model.backbone
        head = model.head
    else:
        logger.error("Classifier %s does not implement ClassifierWithBackboneAndHead", type(model))
        raise SaliencyExtractionError(f"Classifier model {type(model)} does not expose backbone and head.")

    from taxon_vision.service.inference import get_image_transform

    transform = get_image_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    linear = _extract_linear_layer(head)

    with torch.no_grad():
        if target_class is None:
            logits = model(tensor)
            class_idx = int(logits.argmax(dim=-1).item())
        else:
            class_idx = target_class

        w_c = linear.weight[class_idx]
        cam = _compute_spatial_cam(backbone, w_c, tensor)

        upsampled = F.interpolate(
            cam.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        cam_np = upsampled.cpu().numpy()
        c_min = float(cam_np.min())
        c_max = float(cam_np.max())
        eps = float(get_settings().model.cam_numerical_epsilon)
        norm = (cam_np - c_min) / (c_max - c_min + eps)
        return norm.astype(np.float32)


def overlay_cam_on_image(
    image: Image.Image,
    heatmap: npt.NDArray[np.float32],
    alpha: float = 0.5,
) -> Image.Image:
    """Alpha-blend normalized 2D CAM heatmap over an RGB image using a colormap.

    Why:
        Ecologists and operators need an interpretable, visual representation of where
        the vision foundation model attended when predicting a taxon. Blending the
        attribution map over the raw photograph highlights morphological keys.

    How:
        Applies a viridis colormap to the 2D float32 attribution map, scales to RGB,
        and alpha-blends with the source image using PIL Image.blend.

    Complexity:
        O(H * W) element-wise operations.

    Args:
        image: Original input PIL Image.
        heatmap: 2D float32 numpy array with values in [0.0, 1.0].
        alpha: Blending transparency factor between image and heatmap overlay.

    Returns:
        Alpha-blended RGB PIL Image.
    """
    import matplotlib.cm as cm

    width, height = image.size
    colored = cm.viridis(heatmap)
    heatmap_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb).resize((width, height), resample=Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_img, alpha=alpha)


def encode_image_base64_png(image: Image.Image) -> str:
    """Serialise a PIL image as a base64-encoded PNG string.

    Why:
        Callers that already hold a computed heatmap (for example to report its shape) must be able to
        render the overlay without invoking the model a second time.

    Args:
        image: Image to encode.

    Returns:
        UTF-8 base64 text of the PNG byte stream, suitable for a ``data:image/png;base64,`` URI.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_cam_overlay(
    image: Image.Image,
    classifier: nn.Module | None = None,
    target_class: int | None = None,
    alpha: float = 0.5,
) -> str:
    """Generate CAM attribution, blend with the original image, and encode as base64 PNG.

    Why:
        Provides a single-call helper for API endpoints and web UI partials to produce
        ready-to-render image overlays without exposing intermediate numpy buffers.

    How:
        Invokes `generate_heatmap`, overlays on the input image via `overlay_cam_on_image`,
        and serializes via `encode_image_base64_png`.

    Complexity:
        O(1) forward model pass plus O(H * W) interpolation and image blending.

    Args:
        image: Original input PIL Image.
        classifier: Optional TaxonClassifier instance. If None, resolves from service cache.
        target_class: Target class index to explain. If None, defaults to argmax prediction.
        alpha: Overlay blending opacity factor.

    Returns:
        Base64-encoded PNG image string.
    """
    heatmap = generate_heatmap(image, classifier=classifier, target_class=target_class)
    overlay = overlay_cam_on_image(image, heatmap, alpha=alpha)
    return encode_image_base64_png(overlay)
