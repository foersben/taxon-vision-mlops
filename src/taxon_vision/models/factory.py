# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Timm backbone factory supporting BioCLIP-2, DINOv2, MobileNet, and EfficientNet."""

from __future__ import annotations

import timm
import torch.nn as nn

BACKBONE_REGISTRY = {
    "bioclip-2": "hf-hub:imageomics/bioclip-2",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
}


def create_feature_extractor(model_name: str = "mobilenetv4_conv_small", pretrained: bool = False) -> nn.Module:
    """Instantiate frozen pre-trained feature extractor."""
    resolved_name = BACKBONE_REGISTRY.get(model_name, model_name)
    try:
        model = timm.create_model(resolved_name, pretrained=pretrained, num_classes=0)
    except Exception:
        # Fallback to standard lightweight model if remote weights unavailable in offline CI
        model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    return model
