# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Data transfer objects and configuration containers for training and tuning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EmbeddingSplit:
    """Pre-computed feature representations and labels for training and validation.

    Groups training and validation tensors together to eliminate parameter clumps and guarantee consistent feature dimensions across splits.

    Attributes:
        train_embeddings: Pre-computed representations of shape `(N_train, feature_dim)`.
        train_labels: Ground-truth class labels of shape `(N_train,)`.
        val_embeddings: Validation embeddings of shape `(N_val, feature_dim)`.
        val_labels: Validation class labels of shape `(N_val,)`.
        samples_per_class: Optional pre-computed class sample counts for loss weighting.
    """

    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    val_embeddings: torch.Tensor
    val_labels: torch.Tensor
    samples_per_class: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Validate tensor shapes and consistency across training and validation splits.

        Raises:
            ValueError: If feature dimensions differ between train and validation tensors, or if label counts do not match sample counts.
        """
        if self.train_embeddings.ndim != 2 or self.val_embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D tensors, got train={self.train_embeddings.shape} "
                f"and val={self.val_embeddings.shape}."
            )
        if self.train_embeddings.shape[0] != self.train_labels.shape[0]:
            raise ValueError(
                f"Train embeddings count ({self.train_embeddings.shape[0]}) does not match "
                f"train labels count ({self.train_labels.shape[0]})."
            )
        if self.val_embeddings.shape[0] != self.val_labels.shape[0]:
            raise ValueError(
                f"Val embeddings count ({self.val_embeddings.shape[0]}) does not match "
                f"val labels count ({self.val_labels.shape[0]})."
            )
        if self.train_embeddings.shape[1] != self.val_embeddings.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: train is {self.train_embeddings.shape[1]}, "
                f"val is {self.val_embeddings.shape[1]}."
            )

    @property
    def feature_dim(self) -> int:
        """Dimensionality of feature vectors extracted by the backbone.

        Why:
            The feature dimension defines the dimensionality of the embedding space produced by the neural network's backbone. This value is critical for designing downstream components such as the classification head, nearest-neighbor search indexes, and dimensionality reduction projections. Ensuring a consistent feature dimension across all embeddings guarantees compatibility between these components and simplifies the architecture design.

        How:
            The feature dimension is determined by the output dimensionality of the final layer in the backbone network (e.g., the output of a ResNet's global average pooling layer). It is accessed directly from the shape of the pre-computed embedding tensors.

        Complexity:
            - time complexity: O(1), as the dimensionality is retrieved directly from the tensor metadata.
            - space complexity: O(1), as no additional memory is allocated.

        Returns:
            The dimensionality of feature vectors extracted by the backbone.
        """
        return int(self.train_embeddings.shape[1])

    def get_samples_per_class(self, num_classes: int) -> list[int]:
        """Resolve sample frequencies per class, falling back to counting training labels.

        Args:
            num_classes: Total number of target classification categories.

        Returns:
            List containing the count of training observations per class.
        """
        if self.samples_per_class is not None:
            return list(self.samples_per_class)
        return [max(1, int((self.train_labels == c).sum().item())) for c in range(num_classes)]


@dataclass(frozen=True)
class HeadTrainingConfig:
    """Hyperparameters and callbacks controlling classification head training.

    Attributes:
        epochs: Number of complete training epochs to execute.
        batch_size: Mini-batch size for gradient optimization steps.
        pruner_callback: Optional callable taking `(epoch, val_pr_auc)` that returns
            True if training should terminate early.
        early_stopping_patience: Number of epochs without validation PR-AUC improvement
            before terminating training early. None disables early stopping.
        reduce_lr_patience: Number of epochs without validation loss improvement before
            reducing the optimizer learning rate. None disables LR scheduler.
        restore_best_weights: Whether to restore the head weights that achieved peak
            validation PR-AUC upon training completion.
        noise_std: Standard deviation of Gaussian noise injected into input embeddings
            during training for manifold regularization. 0.0 disables noise.
    """

    epochs: int = 10
    batch_size: int = 64
    pruner_callback: Callable[[int, float], bool] | None = None
    early_stopping_patience: int | None = 4
    reduce_lr_patience: int | None = 2
    restore_best_weights: bool = True
    noise_std: float = 0.0


@dataclass(frozen=True)
class TuningConfig:
    """Configuration options governing Bayesian hyperparameter optimization.

    Attributes:
        n_trials: Total number of hyperparameter evaluation trials to run.
        epochs_per_trial: Maximum training epochs allocated to each trial.
        batch_size: Mini-batch size for gradient steps during trial training.
        seed: Deterministic random seed for Optuna TPE sampler.
        tracking_uri: Target MLflow tracking URI (e.g. local directory or remote DagsHub).
        experiment_name: MLflow experiment name under which runs will be organized.
    """

    n_trials: int = 15
    epochs_per_trial: int = 10
    batch_size: int = 64
    seed: int = 42
    tracking_uri: str | None = None
    experiment_name: str = "taxon-vision-hyperparameter-tuning"
