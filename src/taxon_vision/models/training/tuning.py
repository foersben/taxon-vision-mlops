# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Automated hyperparameter optimization with Optuna and MLflow."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import mlflow
import optuna
from torch import nn, optim

from taxon_vision.config import get_settings
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig

logger = logging.getLogger(__name__)


class HyperparameterObjective:
    """Optuna objective function evaluating candidate hyperparameters on cached embeddings.

    Why:
        Grid and random searches scale exponentially with the number of hyperparameters and waste compute exploring unpromising regions of hyperparameter space. Bayesian optimization with Tree-structured Parzen Estimators (TPE) constructs probabilistic models of the objective function p(x|y), focusing search effort on high-performing configurations. Coupling TPE with Asynchronous Successive Halving (ASHA) pruning allows aborting underperforming trials within 2 epochs, reducing total tuning duration by 3-5x.

    How:
        1. Samples learning rate, weight decay, dropout rate, and Class-Balanced beta from prior distributions.
        2. Instantiates classification head, Class-Balanced Loss, and AdamW optimizer.
        3. Mounts an optional nested MLflow run under the active tuning experiment.
        4. Trains the head on cached embeddings, invoking Optuna's pruning callback at each epoch.
        5. Logs per-epoch trajectories and returns final validation Macro PR-AUC as the optimization target.

    Attributes:
        data: Pre-computed feature representations and ground-truth labels.
        num_classes: Total number of target classification categories.
        config: Tuning hyperparameters specifying epoch budget and batch size.
        mlflow_available: Flag indicating whether an active MLflow server is available.
        samples_per_class: Number of samples per class.
        feature_dim: Dimension of the feature vectors.
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
            data: Pre-computed feature representations and ground-truth labels.
            num_classes: Total number of target classification categories.
            config: Tuning hyperparameters specifying epoch budget and batch size.
            mlflow_available: Flag indicating whether an active MLflow server is available.
        """
        self.data = data
        self.num_classes = num_classes
        self.config = config
        self.mlflow_available = mlflow_available
        self.samples_per_class = data.get_samples_per_class(num_classes)
        self.feature_dim = data.feature_dim

    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample hyperparameters for the current Optuna trial.

        Args:
            trial: Optuna trial instance.

        Returns:
            Dictionary containing sampled hyperparameters.
        """
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "beta": trial.suggest_float("beta", 0.99, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.05),
        }

    def _construct_components(self, params: dict[str, Any]) -> tuple[nn.Module, ClassBalancedLoss, optim.Optimizer]:
        """Construct the classification head, loss criterion, and optimizer.

        Args:
            params: Hyperparameter configuration.

        Returns:
            Tuple containing the classification head, loss criterion, and optimizer.
        """
        head = nn.Sequential(
            nn.Dropout(p=params["dropout"]),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        criterion = ClassBalancedLoss(samples_per_class=self.samples_per_class, beta=params["beta"])
        optimizer = optim.AdamW(head.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        return head, criterion, optimizer

    def _start_mlflow_trial(self, trial: optuna.Trial, params: dict[str, Any]) -> Any:
        """Initialize an MLflow nested run and log trial parameters.

        Args:
            trial: Optuna trial instance.
            params: Hyperparameter configuration.
        """
        if not self.mlflow_available:
            return None
        try:
            run = mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)
            mlflow.log_params(params)
            mlflow.set_tag("optuna.trial_number", trial.number)
            mlflow.set_tag("pipeline.stage", "HyperparameterSearch")
            return run
        except Exception as err:
            logger.debug("Failed to start MLflow run: %s", err)
            return None

    def _log_trial_metrics(self, run: Any, history: dict[str, list[float]]) -> None:
        """Log training metrics for the completed trial to MLflow.

        Args:
            run: MLflow run object.
            history: Dictionary containing training history.
        """
        if run is None:
            return
        try:
            for ep, (t_loss, v_loss, v_acc, v_f1, v_pr) in enumerate(
                zip(
                    history["train_loss"],
                    history["val_loss"],
                    history["val_accuracy"],
                    history["val_f1"],
                    history["val_pr_auc"],
                    strict=False,
                )
            ):
                mlflow.log_metric("train_loss", t_loss, step=ep)
                mlflow.log_metric("val_loss", v_loss, step=ep)
                mlflow.log_metric("val_accuracy", v_acc, step=ep)
                mlflow.log_metric("val_f1", v_f1, step=ep)
                mlflow.log_metric("val_pr_auc", v_pr, step=ep)
        except Exception as err:
            logger.debug("Failed to log MLflow metrics: %s", err)

    def __call__(self, trial: optuna.Trial) -> float:
        """Execute a single hyperparameter evaluation trial.

        Args:
            trial: Optuna trial instance.

        Returns:
            Best validation Macro PR-AUC achieved.

        Raises:
            optuna.TrialPruned: If trial performance is pruned.
        """
        params = self._sample_hyperparameters(trial)
        head, criterion, optimizer = self._construct_components(params)

        def pruner_callback(epoch: int, val_pr_auc: float) -> bool:
            """Pruner callback function.

            Args:
                epoch: Epoch number.
                val_pr_auc: Validation PR-AUC.

            Returns:
                True if training should be pruned, False otherwise.
            """
            trial.report(val_pr_auc, step=epoch)
            return trial.should_prune()

        active_run = self._start_mlflow_trial(trial, params)
        head_config = HeadTrainingConfig(
            epochs=self.config.epochs_per_trial,
            batch_size=int(params["batch_size"]),
            noise_std=float(params["noise_std"]),
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

            self._log_trial_metrics(active_run, history)

            if trial.should_prune():
                raise optuna.TrialPruned()

            best_val = float(max(history["val_pr_auc"])) if history.get("val_pr_auc") else 0.0
            if active_run:
                try:
                    mlflow.log_metric("peak_val_pr_auc", best_val)
                except Exception:
                    pass
            return best_val

        finally:
            if active_run is not None:
                with contextlib.suppress(Exception):
                    mlflow.end_run()


def _setup_mlflow(cfg: TuningConfig) -> bool:
    """Configure MLflow for hyperparameter tuning.

    Args:
        cfg: Tuning configuration.

    Returns:
        True if MLflow is available, False otherwise.
    """
    mlflow_cfg = get_settings().mlflow
    resolved_uri = (
        cfg.tracking_uri if cfg.tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg.tracking_uri)
    )
    resolved_experiment = cfg.experiment_name or mlflow_cfg.experiment_name

    if not resolved_uri:
        return False

    try:
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(resolved_experiment)
        return True
    except Exception as exc:
        logger.warning("MLflow tracking setup failed or remote server unreachable: %s. Continuing locally.", exc)
        return False


def _log_best_run(best_params: dict[str, Any], best_value: float, active: bool) -> None:
    """Log the best overall Optuna configuration to MLflow.

    Args:
        best_params: Best hyperparameter configuration.
        best_value: Best macro PR-AUC achieved.
        active: Whether MLflow is available.
    """
    if not active:
        return
    try:
        from taxon_vision.models.training.tracking import _get_dvc_dataset_hash

        with mlflow.start_run(run_name="optuna_best_model"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_pr_auc", best_value)
            mlflow.set_tag("optuna.optimization_algorithm", "TPE+ASHA")
            mlflow.set_tag("pipeline.stage", "Phase1_Step3_Tuning")

            dvc_hash = _get_dvc_dataset_hash()
            if dvc_hash:
                mlflow.log_param("dvc_dataset_hash", dvc_hash)

            try:
                import subprocess

                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
                mlflow.set_tag("git.commit_sha", git_sha)
                mlflow.set_tag("git.branch", git_branch)
            except Exception:
                pass
    except Exception as err:
        logger.debug("Failed to log best run to MLflow: %s", err)


def tune_hyperparameters(
    data: EmbeddingSplit,
    num_classes: int,
    config: TuningConfig | None = None,
) -> tuple[dict[str, Any], optuna.Study]:
    r"""Automate Bayesian hyperparameter tuning using Optuna with ASHA pruning and MLflow tracking.

    Why:
        Fine-tuning hyperparameters (learning rate, weight decay, dropout, and Class-Balanced Loss $\beta$)
        interactively or via ad-hoc heuristics leads to suboptimal convergence and poor generalization
        on long-tailed biological distributions. Automating search via Bayesian Optimization (TPE)
        guarantees reproducible exploration of the hyperparameter landscape while pruning poor trials
        early to respect compute budgets.

    How:
        1. Sets up MLflow tracking URI and experiment if configured.
        2. Configures `TPESampler` with deterministic PRNG seed and `SuccessiveHalvingPruner`
           with `min_resource=2` and `reduction_factor=2`.
        3. Instantiates `HyperparameterObjective` and executes `study.optimize` across `n_trials`.
        4. Logs the optimal parameter configuration and peak Macro PR-AUC to MLflow as a consolidated run.
        5. Returns the best parameters dictionary and the full Optuna study object.

    Args:
        data: Pre-computed feature representations and labels.
        num_classes: Total number of target species or taxa categories.
        config: Tuning configuration controlling trials, epochs, batch size, seed, and MLflow.

    Returns:
        A tuple `(best_params, study)` of the best configuration and the Optuna study.
    """
    cfg = config or TuningConfig()
    mlflow_available = _setup_mlflow(cfg)

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

    _log_best_run(best_params, study.best_value, mlflow_available)
    return best_params, study
