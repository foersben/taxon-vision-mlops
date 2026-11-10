# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Vision Foundation Model (VFM) backbone factory.

This module provides factory functions to instantiate pre-trained domain foundation models (specifically BioCLIP-2, DINOv3, DINOv2, MobileNetV4, and EfficientNet) via the `timm` library. The feature extractor backbones are instantiated with their classification heads removed (`num_classes=0`) and all parameter gradients frozen (`requires_grad=False`), decoupling downstream taxonomic head training from full backbone backpropagation.

Model architecture mappings, fallback identifiers, and image dimensions are resolved dynamically from the centralized type-safe application configuration (`taxon_vision.config`), eliminating magic numbers and hardcoded globals.

Typical usage example:
    backbone = create_feature_extractor("bioclip-2", pretrained=True)
    dim = get_feature_dimension(backbone)
    embeddings = extract_features(backbone, batch_images)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple

import timm
import torch
import torch.nn as nn

from taxon_vision.config import ModelConfig, get_settings

BACKBONE_REGISTRY: dict[str, str] = get_settings().model.backbone_registry
"""dict[str, str]: Mapping of canonical model identifiers to timm/HuggingFace model identifiers."""


class CachedFeaturesResult(NamedTuple):
    """Cached feature embedding matrix paired with ground-truth integer labels."""

    features: torch.Tensor
    labels: torch.Tensor


def create_feature_extractor(
    model_name: str | None = None,
    pretrained: bool = False,
    config: ModelConfig | None = None,
    allow_fallback: bool = False,
) -> nn.Module:
    """Instantiate a frozen pre-trained feature extractor backbone.

    Why:
        Fine-tuning Vision Foundation Models end-to-end across long-tailed taxonomic classes risks catastrophic forgetting and exceeds the 11.2 GB VRAM training ceiling. Freezing the backbone (`requires_grad=False`) and removing its classification head (`num_classes=0`) decouples representation extraction from linear probing. A failed load raises by default so un-pretrained random weights are never silently substituted unless `allow_fallback=True` is explicitly passed for offline CI tests.

    How:
        Resolves `model_name` through `config.backbone_registry`, instantiates the `timm` backbone with `num_classes=0`, locks all parameters (`requires_grad=False`), and activates evaluation mode (`eval()`).

    Complexity:
        O(P) time and space where P is the number of backbone parameters.

    Args:
        model_name: Identifier for the desired model backbone. Can be a key from `config.backbone_registry` or any valid timm model name. If None, uses `config.default_extractor`.
        pretrained: Whether to download and load pre-trained weights from the remote registry or Hugging Face Hub.
        config: Model subsystem configuration. If None, loaded from global `get_settings().model`.
        allow_fallback: Whether to fall back to the un-pretrained `config.fallback_extractor` if the requested model cannot be created.

    Returns:
        The instantiated PyTorch backbone with frozen parameters and evaluation mode active.

    Raises:
        RuntimeError: If the requested model cannot be created and `allow_fallback` is False, or if both the requested model and the fallback architecture fail.
    """
    model_cfg = config or get_settings().model
    target_name = model_name or model_cfg.default_extractor
    resolved_name = model_cfg.backbone_registry.get(target_name, target_name)

    try:
        model: nn.Module = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(f"Could not create backbone '{resolved_name}' (pretrained={pretrained}): {exc}") from exc
        # Explicit opt-in: lightweight un-pretrained model for offline CI and unit tests
        fallback_name = model_cfg.fallback_extractor
        model = timm.create_model(fallback_name, pretrained=False, num_classes=0)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


def get_feature_dimension(
    model: nn.Module,
    image_size: int | None = None,
    channels: int | None = None,
    config: ModelConfig | None = None,
) -> int:
    """Retrieve the output feature embedding dimension for a given backbone.

    Why:
        Heterogeneous vision backbones (DINOv3, BioCLIP-2, MobileNetV4) emit varying penultimate feature dimensions (e.g. 512, 768, 1280) and expose inconsistent attribute names across `timm` families. Probing with a single dummy tensor deterministically resolves the true output dimensionality without reflection (`getattr`) or hardcoded architecture tables.

    How:
        Constructs a `(1, C, H, W)` zero tensor on the backbone's device, runs a single forward pass inside `torch.inference_mode()`, and reads `features.shape[-1]`.

    Complexity:
        O(1) single-sample forward pass through the frozen backbone.

    Args:
        model: The feature extractor backbone to inspect.
        image_size: Input spatial image height/width. If None, resolved from `config.image_size`.
        channels: Input image color channels. If None, resolved from `config.channels`.
        config: Model configuration. If None, loaded from `get_settings().model`.

    Returns:
        The dimensional size of the output feature vector (e.g., 512, 768, 1280).
    """
    model_cfg = config or get_settings().model
    size = image_size if image_size is not None else model_cfg.image_size
    num_channels = channels if channels is not None else model_cfg.channels

    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, num_channels, size, size, device=device)
    with torch.inference_mode():
        features = model(dummy_input)
    return int(features.shape[-1])


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Extract dense feature embeddings from a batch of images using a frozen backbone.

    Why:
        Executing forward passes inside `torch.inference_mode()` disables autograd view tracking and version counters, minimizing VRAM footprint and latency during batch embedding extraction.

    How:
        Aligns `model` and `images` onto `target_device`, switches `model` to `eval()`, and evaluates `model(images)` under `torch.inference_mode()`.

    Complexity:
        O(B) backbone forward evaluation where B is the batch size.

    Args:
        model: The frozen feature extractor backbone.
        images: A 4D tensor representing the batch of normalized images of shape `(batch_size, channels, height, width)`.
        device: Target device for execution. If None, the device of the model parameters is used.

    Returns:
        Dense feature embeddings of shape `(batch_size, feature_dim)`.
    """
    if device is not None:
        target_device = torch.device(device)
        model = model.to(target_device)
        images = images.to(target_device)
    else:
        target_device = next(model.parameters()).device
        images = images.to(target_device)

    model.eval()
    with torch.inference_mode():
        embeddings: torch.Tensor = model(images)
    return embeddings


def extract_and_cache_features(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor] | list[Any]],
    device: torch.device | str | None = None,
) -> CachedFeaturesResult:
    """Iterate over an entire dataset to pre-compute and cache feature embeddings.

    Decouples feature extraction from linear head training. By passing the dataset through the frozen vision foundation model once and caching embeddings in memory or storage, subsequent classification head training iterations run in seconds rather than hours.

    Args:
        model: The frozen feature extractor backbone.
        dataloader: An iterable or PyTorch DataLoader yielding batches of `(images, labels)`.
        device: Target execution device. If None, uses the model parameter device.

    Returns:
        CachedFeaturesResult `(features, labels)` where:
            - `features` is a 2D float tensor of shape `(total_samples, feature_dim)`.
            - `labels` is a 1D integer tensor of shape `(total_samples,)`.
    """
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in dataloader:
        images, labels = batch[0], batch[1]
        embeddings = extract_features(model, images, device=device)
        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_embeddings:
        return CachedFeaturesResult(
            features=torch.empty((0, 0), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )

    return CachedFeaturesResult(
        features=torch.cat(all_embeddings, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
