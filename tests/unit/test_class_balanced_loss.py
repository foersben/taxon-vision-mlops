import torch

from taxon_vision.models.loss import ClassBalancedLoss


def test_class_balanced_loss_computation() -> None:
    samples_per_class = [1000, 100, 10]
    loss_fn = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.999)
    assert len(loss_fn.weights) == 3
    # Rare class should have substantially higher weight than frequent class
    assert loss_fn.weights[2] > loss_fn.weights[0]

    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0


def test_class_balanced_loss_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="samples_per_class cannot be empty"):
        ClassBalancedLoss(samples_per_class=[], beta=0.99)

    with pytest.raises(ValueError, match="beta must be in"):
        ClassBalancedLoss(samples_per_class=[10, 20], beta=1.5)
