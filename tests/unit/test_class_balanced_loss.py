import torch

from taxon_vision.models.loss import ClassBalancedLoss


def test_class_balanced_loss_computation():
    samples_per_class = [1000, 100, 10]
    loss_fn = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.999)
    assert len(loss_fn.weights) == 3
    # Rare class should have substantially higher weight than frequent class
    assert loss_fn.weights[2] > loss_fn.weights[0]

    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
