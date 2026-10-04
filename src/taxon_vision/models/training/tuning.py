# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Automated hyperparameter optimization with Optuna and MLflow."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import optuna
import torch.nn as nn
import torch.optim as optim

from taxon_vision.config import get_settings
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig

logger = logging.getLogger(__name__)


class HyperparameterObjective:
    """Optuna objective function for tuning the classification head.

    Evaluates candidate learning rates, weight decays, dropout rates, and
    Class-Balanced Loss beta values on pre-computed feature embeddings.
    """

    def __init__(
        self,
        data: EmbeddingSplit,
        num_classes: int,
        config: TuningConfig,
        mlflow_available: bool = False,
    ) -> None:
        """Initialize the hyperparameter optimization objective.

        Args:
            data: Pre-computed feature representations and ground-truth labels for both
                training and validation splits.
            num_classes: Total number of target classification categories.
            config: Tuning hyperparameters specifying epoch budget and batch size.
            mlflow_available: Flag indicating whether an active MLflow server is available
                for metric and parameter logging.
        """
        self.data = data
        self.num_classes = num_classes
        self.config = config
        self.mlflow_available = mlflow_available
        self.samples_per_class = data.get_samples_per_class(num_classes)
        self.feature_dim = data.feature_dim

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute a single hyperparameter evaluation trial.

        Samples hyperparameter candidates from Optuna distributions, constructs the
        head, trains it on cached embeddings, and reports validation accuracy.

        Args:
            trial: Optuna trial instance providing parameter suggestions and pruning APIs.

        Returns:
            Best top-1 validation accuracy achieved during trial execution.

        Raises:
            optuna.TrialPruned: If trial performance falls behind median thresholds.
        """
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        beta = trial.suggest_float("beta", 0.99, 0.9999)

        # Construct head and loss
        head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        criterion = ClassBalancedLoss(samples_per_class=self.samples_per_class, beta=beta)
        optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

        def pruner_callback(epoch: int, val_acc: float) -> bool:
            trial.report(val_acc, step=epoch)
            return trial.should_prune()

        # MLflow run for trial
        active_run = None
        if self.mlflow_available:
            try:
                active_run = mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
                mlflow.log_params(
                    {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "beta": beta,
                        "batch_size": self.config.batch_size,
                    }
                )
            except Exception as err:
                logger.debug("Failed to start MLflow run: %s", err)

        head_config = HeadTrainingConfig(
            epochs=self.config.epochs_per_trial,
            batch_size=self.config.batch_size,
            pruner_callback=pruner_callback,
        )

        try:
            history = train_head_on_cached_embeddings(
                head=head,
                data=self.data,
                optimizer=optimizer,
                criterion=criterion,
                config=head_config,
            )
            final_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0

            if active_run is not None:
                try:
                    for ep, (t_loss, v_loss, v_acc) in enumerate(
                        zip(
                            history["train_loss"],
                            history["val_loss"],
                            history["val_accuracy"],
                            strict=False,
                        )
                    ):
                        mlflow.log_metric("train_loss", t_loss, step=ep)
                        mlflow.log_metric("val_loss", v_loss, step=ep)
                        mlflow.log_metric("val_accuracy", v_acc, step=ep)
                except Exception as err:
                    logger.debug("Failed to log MLflow metrics: %s", err)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return final_acc

        finally:
            if active_run is not None:
                try:
                    mlflow.end_run()
                except Exception:
                    pass


def tune_hyperparameters(
    data: EmbeddingSplit,
    num_classes: int,
    config: TuningConfig | None = None,
) -> tuple[dict[str, Any], optuna.Study]:
    r"""Automate Bayesian hyperparameter tuning using Optuna with ASHA pruning and MLflow tracking.

    Tunes learning rate, weight decay, dropout rate, and Class-Balanced Loss $\beta$
    hyperparameters. Uses the Tree-structured Parzen Estimator (TPE) algorithm to
    sample configurations and an Asynchronous Successive Halving Pruner to terminate
    sub-optimal configurations within their first epochs.

    Each trial logs its parameters and performance curves to MLflow (configured for
    DagsHub or local tracking). The optimal trial parameters and study object are returned.

    Args:
        data: Pre-computed feature representations and labels for train and validation splits.
        num_classes: Total number of target species or taxa categories.
        config: Tuning configuration controlling trials, epochs, batch size, seed, and MLflow options.
            If None, default configuration values are applied.

    Returns:
        A tuple `(best_params, study)` where:
            - `best_params` is a dictionary of the best hyperparameter values found.
            - `study` is the completed Optuna study object.
    """
    cfg = config or TuningConfig()

    # Setup MLflow tracking from centralized configuration
    mlflow_cfg = get_settings().mlflow
    resolved_uri = (
        cfg.tracking_uri if cfg.tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.tracking_uri)
    )
    resolved_experiment = cfg.experiment_name or mlflow_cfg.experiment_name

    mlflow_available = False
    if resolved_uri:
        try:
            mlflow.set_tracking_uri(resolved_uri)
            mlflow.set_experiment(resolved_experiment)
            mlflow_available = True
        except Exception as exc:
            logger.warning("MLflow tracking setup failed or remote server unreachable: %s. Continuing locally.", exc)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = HyperparameterObjective(
        data=data,
        num_classes=num_classes,
        config=cfg,
        mlflow_available=mlflow_available,
    )

    study.optimize(objective, n_trials=cfg.n_trials)

    best_params = study.best_params
    logger.info("Hyperparameter search complete. Best parameters: %s, Best value: %.4f", best_params, study.best_value)

    # Log best overall run
    if mlflow_available:
        try:
            with mlflow.start_run(run_name="optuna_best_model"):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_accuracy", study.best_value)
                mlflow.set_tag("optuna.optimization_algorithm", "TPE+ASHA")
                mlflow.set_tag("pipeline.stage", "Phase1_Step3_Tuning")
        except Exception as err:
            logger.debug("Failed to log best run to MLflow: %s", err)

    return best_params, study
