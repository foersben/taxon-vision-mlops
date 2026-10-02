"""Test models components.py.

This module provides functionality related to test_models_components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from taxon_vision.inference.gradcam import generate_heatmap
from taxon_vision.models.head import TaxonClassifier
from taxon_vision.models.trainer import train_head_epoch


def test_feature_extractor_and_head() -> None:
    """Test feature extractor and head."""
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 64))
    classifier = TaxonClassifier(backbone=backbone, feature_dim=64, num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = classifier(dummy_input)
    assert logits.shape == (2, 5)


def test_train_head_epoch() -> None:
    """Test train head epoch."""
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16))
    classifier = TaxonClassifier(backbone=backbone, feature_dim=16, num_classes=3)
    optimizer = optim.SGD(classifier.head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataloader = [(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 1]))]
    avg_loss = train_head_epoch(classifier, dataloader, optimizer, criterion)
    assert avg_loss >= 0.0


def test_gradcam_heatmap() -> None:
    """Test gradcam heatmap."""
    img = Image.new("RGB", (100, 100))
    heatmap = generate_heatmap(img)
    assert heatmap.shape == (100, 100)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
