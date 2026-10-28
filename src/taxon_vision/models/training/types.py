# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Data transfer objects and configuration containers for training and tuning."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import torch
from torch import nn, optim

from taxon_vision.models.loss import ClassBalancedLoss


class TrainingComponentsTuple(NamedTuple):
    """NamedTuple representation of wired training components."""

    head: nn.Module
    criterion: ClassBalancedLoss
    optimizer: optim.Optimizer
    config: HeadTrainingConfig
    params: dict[str, Any]


class ResolvedTrainingParametersTuple(NamedTuple):
    """NamedTuple representation of resolved baseline training parameters."""

    extractor: str
    epochs: int
    batch_size: int
    learning_rate: float
    checkpoint_path: Path | str


class PromotionAlias(StrEnum):
    """Enumeration of model promotion aliases and gating outcomes."""

    PRODUCTION = "Production"
    CHALLENGER = "Challenger"
    ARCHIVED = "Archived"
    SKIPPED_BELOW_INCUMBENT = "skipped_below_incumbent"


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Resolved hyperparameter tuple and provenance tracking."""

    learning_rate: float
    weight_decay: float
    dropout: float
    cb_beta: float
    batch_size: int
    noise_std: float
    hyperparameter_source: str


@dataclass(frozen=True)
class EmbeddingSplit:
    """Pre-computed feature representations and labels for training, validation, calibration, and test.

    Groups training, validation, calibration, and holdout test tensors together to eliminate parameter
    clumps and guarantee consistent feature dimensions across splits.

    Attributes:
        train_embeddings: Pre-computed representations of shape `(N_train, feature_dim)`.
        train_labels: Ground-truth class labels of shape `(N_train,)`.
        val_embeddings: Validation embeddings of shape `(N_val, feature_dim)`.
        val_labels: Validation class labels of shape `(N_val,)`.
        cal_embeddings: Optional calibration embeddings of shape `(N_cal, feature_dim)`.
        cal_labels: Optional calibration class labels of shape `(N_cal,)`.
        test_embeddings: Optional holdout test embeddings of shape `(N_test, feature_dim)`.
        test_labels: Optional holdout test class labels of shape `(N_test,)`.
        samples_per_class: Optional pre-computed class sample counts for loss weighting.
    """

    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    val_embeddings: torch.Tensor
    val_labels: torch.Tensor
    cal_embeddings: torch.Tensor | None = None
    cal_labels: torch.Tensor | None = None
    test_embeddings: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    samples_per_class: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Validate tensor shapes and consistency across training, validation, calibration, and test splits.

        Raises:
            ValueError: If feature dimensions differ across splits, or if label counts do not match sample counts.
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
        dim = self.train_embeddings.shape[1]
        self._validate_optional_split("cal", self.cal_embeddings, self.cal_labels, dim)
        self._validate_optional_split("test", self.test_embeddings, self.test_labels, dim)

    @staticmethod
    def _validate_optional_split(
        name: str,
        embeddings: torch.Tensor | None,
        labels: torch.Tensor | None,
        expected_dim: int,
    ) -> None:
        """Validate an optional auxiliary split tensor pair."""
        if embeddings is None and labels is None:
            return
        if embeddings is None or labels is None:
            raise ValueError(f"Both {name}_embeddings and {name}_labels must be provided together.")
        if embeddings.ndim != 2:
            raise ValueError(f"{name.capitalize()} embeddings must be a 2D tensor, got {embeddings.shape}.")
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{name.capitalize()} embeddings count ({embeddings.shape[0]}) does not match "
                f"{name} labels count ({labels.shape[0]})."
            )
        if embeddings.shape[1] != expected_dim:
            raise ValueError(f"Feature dimension mismatch: train is {expected_dim}, {name} is {embeddings.shape[1]}.")

    @classmethod
    def _load_from_directory(cls, dir_path: Path) -> EmbeddingSplit:
        """Load and assemble EmbeddingSplit from a directory containing partition files."""
        composite_pt = dir_path / "cached_features.pt"
        if composite_pt.exists():
            return cls.from_cache(composite_pt)
        train_pt = dir_path / "train_embeddings.pt"
        val_pt = dir_path / "val_embeddings.pt"
        if not (train_pt.exists() and val_pt.exists()):
            raise ValueError(f"Directory {dir_path} lacks valid embedding cache files")
        tr_data = torch.load(train_pt, map_location="cpu", weights_only=False)
        va_data = torch.load(val_pt, map_location="cpu", weights_only=False)
        cal_pt = dir_path / "cal_embeddings.pt"
        test_pt = dir_path / "test_embeddings.pt"
        ca_data = torch.load(cal_pt, map_location="cpu", weights_only=False) if cal_pt.exists() else {}
        te_data = torch.load(test_pt, map_location="cpu", weights_only=False) if test_pt.exists() else {}
        return cls(
            train_embeddings=tr_data["embeddings"],
            train_labels=tr_data["labels"],
            val_embeddings=va_data["embeddings"],
            val_labels=va_data["labels"],
            cal_embeddings=ca_data.get("embeddings"),
            cal_labels=ca_data.get("labels"),
            test_embeddings=te_data.get("embeddings"),
            test_labels=te_data.get("labels"),
            samples_per_class=tr_data.get("samples_per_class"),
        )

    @classmethod
    def _load_from_payload(cls, payload: Any, path: Path) -> EmbeddingSplit:
        """Parse in-memory deserialized object or dictionary into an EmbeddingSplit."""
        if isinstance(payload, EmbeddingSplit):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("split"), EmbeddingSplit):
            return payload["split"]
        if isinstance(payload, dict) and "train_embeddings" in payload and "val_embeddings" in payload:
            return cls(
                train_embeddings=payload["train_embeddings"],
                train_labels=payload["train_labels"],
                val_embeddings=payload["val_embeddings"],
                val_labels=payload["val_labels"],
                cal_embeddings=payload.get("cal_embeddings"),
                cal_labels=payload.get("cal_labels"),
                test_embeddings=payload.get("test_embeddings"),
                test_labels=payload.get("test_labels"),
                samples_per_class=payload.get("samples_per_class"),
            )
        raise ValueError(f"Unsupported payload format in embedding cache: {path}")

    @classmethod
    def from_cache(cls, candidate: Path | str) -> EmbeddingSplit:
        """Load and deserialize EmbeddingSplit from serialized cache file or directory.

        Args:
            candidate: Filesystem path to cache .pt file or directory.

        Returns:
            Validated EmbeddingSplit instance.

        Raises:
            FileNotFoundError: If cache file does not exist.
            ValueError: If file payload cannot be parsed into an EmbeddingSplit.
        """
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Embedding cache not found: {candidate_path}")
        if candidate_path.is_dir():
            return cls._load_from_directory(candidate_path)
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        return cls._load_from_payload(payload, candidate_path)

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

    epochs: int = 30
    batch_size: int = 32
    pruner_callback: Callable[[int, float], bool] | None = None
    early_stopping_patience: int | None = 5
    reduce_lr_patience: int | None = 2
    restore_best_weights: bool = True
    noise_std: float = 0.01


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
        extractor_name: Vision backbone extractor identifier.
        storage_url: Optional database storage URI for study persistence.
        study_name: Optional study identifier string.
        reset_study: Whether to purge prior trial records before starting.
        min_resource: Minimum resource allocation for successive halving pruning.
        reduction_factor: Reduction factor for successive halving pruning.
    """

    n_trials: int = 20
    epochs_per_trial: int = 20
    batch_size: int = 32
    seed: int = 42
    tracking_uri: str | None = None
    experiment_name: str = "taxon-vision-head-tuning"
    extractor_name: str = "dinov3"
    storage_url: str | None = None
    study_name: str | None = None
    reset_study: bool = False
    min_resource: int = 3
    reduction_factor: int = 2


@dataclass(frozen=True)
class TrainingComponents:
    """Encapsulates wired PyTorch model head, loss criterion, optimizer, and training configs.

    Provides attribute access (.head, .criterion, ...) and sequence unpacking support
    (head, criterion, optimizer, config, params = components) to maintain 100% backward
    compatibility with legacy 5-tuple consumers.
    """

    head: nn.Module
    criterion: ClassBalancedLoss
    optimizer: optim.Optimizer
    config: HeadTrainingConfig
    params: dict[str, Any]

    def to_tuple(self) -> TrainingComponentsTuple:
        """Export as typed 5-tuple for sequence-unpacking boundaries."""
        return TrainingComponentsTuple(
            head=self.head,
            criterion=self.criterion,
            optimizer=self.optimizer,
            config=self.config,
            params=self.params,
        )

    def __iter__(self) -> Iterator[Any]:
        """Enable tuple-style unpacking: head, criterion, optimizer, config, params = components."""
        yield self.head
        yield self.criterion
        yield self.optimizer
        yield self.config
        yield self.params


@dataclass(frozen=True)
class ResolvedTrainingParameters:
    """Resolved baseline parameters and paths for training pipeline execution."""

    extractor: str
    epochs: int
    batch_size: int
    learning_rate: float
    checkpoint_path: Path | str

    def to_tuple(self) -> ResolvedTrainingParametersTuple:
        """Export as typed 5-tuple for sequence-unpacking boundaries."""
        return ResolvedTrainingParametersTuple(
            extractor=self.extractor,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            checkpoint_path=self.checkpoint_path,
        )

    def __iter__(self) -> Iterator[Any]:
        """Enable tuple-style unpacking for backwards compatibility."""
        yield self.extractor
        yield self.epochs
        yield self.batch_size
        yield self.learning_rate
        yield self.checkpoint_path


class TrainingPipelineResult(TypedDict):
    """Structured result dictionary returned by run_training_pipeline.

    Guarantees strict static type safety while remaining transparently compatible
    with dict-based access patterns across REST API endpoints, CLI, and test assertions.
    """

    status: str
    extractor: str
    epochs_trained: int
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    checkpoint_path: str
    duration_seconds: float
    promotion_status: str | None
    hyperparameter_source: str
    mlflow_run_id: str | None
    history: dict[str, list[float]]
