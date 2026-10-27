# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Automated hyperparameter optimization with Optuna and MLflow."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from optuna.exceptions import ExperimentalWarning
from torch import nn, optim

from taxon_vision.config import get_settings
from taxon_vision.models.loss import ClassBalancedLoss
from taxon_vision.models.training.dataset import get_dvc_dataset_hash
from taxon_vision.models.training.embeddings import train_head_on_cached_embeddings
from taxon_vision.models.training.tracking import _inject_lineage_tags
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig
from taxon_vision.models.training.visualizations import log_optuna_diagnostic_plots

warnings.filterwarnings("ignore", category=ExperimentalWarning)

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
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "cb_beta": trial.suggest_float("cb_beta", 0.99, 0.9999),
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
        lr = float(params.get("learning_rate", params.get("lr", 1e-3)))
        beta = float(params.get("cb_beta", params.get("beta", 0.9999)))
        dropout_p = float(params.get("dropout", 0.2))
        weight_decay = float(params.get("weight_decay", 1e-4))
        head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.feature_dim, self.num_classes),
        )
        criterion = ClassBalancedLoss(samples_per_class=self.samples_per_class, beta=beta)
        optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
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
            run = mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True)
            mlflow.log_params(params)
            mlflow.set_tag("optuna.trial_number", trial.number)
            mlflow.set_tag("pipeline.stage", "HyperparameterSearch")
            return run
        except Exception as err:
            logger.debug("Failed to start MLflow run: %s", err)
            return None

    def _log_trial_metrics(self, run: Any, history: dict[str, list[float]]) -> None:
        """Log training metrics for the completed trial to MLflow using batched API.

        Args:
            run: MLflow run object.
            history: Dictionary containing training history.
        """
        if run is None:
            return
        try:
            run_info = run.info if run is not None else None
            run_id = str(run_info.run_id) if run_info is not None and run_info.run_id is not None else None
            now_ms = int(time.time() * 1000)
            metrics: list[Metric] = []
            for ep, (t_loss, v_loss, v_acc, v_f1, v_pr) in enumerate(
                zip(
                    history.get("train_loss", []),
                    history.get("val_loss", []),
                    history.get("val_accuracy", []),
                    history.get("val_f1", []),
                    history.get("val_pr_auc", []),
                    strict=False,
                )
            ):
                metrics.extend(
                    [
                        Metric("train_loss", float(t_loss), now_ms, ep),
                        Metric("val_loss", float(v_loss), now_ms, ep),
                        Metric("val_accuracy", float(v_acc), now_ms, ep),
                        Metric("val_f1", float(v_f1), now_ms, ep),
                        Metric("val_pr_auc", float(v_pr), now_ms, ep),
                    ]
                )
            if metrics:
                if run_id:
                    client = MlflowClient()
                    client.log_batch(run_id=run_id, metrics=metrics)
                else:
                    for m in metrics:
                        mlflow.log_metric(m.key, m.value, step=m.step)
        except Exception as err:
            logger.debug("Failed to log trial metrics batch: %s", err)

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
    if mlflow.active_run() is not None:
        return True

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


def _apply_git_lineage_tags() -> None:
    """Resolve and attach Git commit and branch tags to active MLflow run."""
    _inject_lineage_tags()


def _log_best_run(
    best_params: dict[str, Any],
    best_value: float,
    active: bool,
    study: optuna.Study | None = None,
) -> None:
    """Log the best overall Optuna configuration and diagnostic plots to MLflow.

    Args:
        best_params: Best hyperparameter configuration.
        best_value: Best macro PR-AUC achieved.
        active: Whether MLflow is available.
        study: Optional Optuna study instance for logging optimization plots.
    """
    if not active:
        return
    try:
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_params(best_params)
            mlflow.log_metric("best_val_pr_auc", best_value)
            mlflow.set_tag("optuna.optimization_algorithm", "TPE+ASHA")
            log_optuna_diagnostic_plots(study)
        else:
            with mlflow.start_run(run_name="optuna_best_model"):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_pr_auc", best_value)
                mlflow.set_tag("optuna.optimization_algorithm", "TPE+ASHA")
                mlflow.set_tag("pipeline.stage", "Phase1_Step3_Tuning")

                dvc_hash = get_dvc_dataset_hash()
                if dvc_hash:
                    mlflow.log_param("dvc_dataset_hash", dvc_hash)

                _apply_git_lineage_tags()
                log_optuna_diagnostic_plots(study)
    except Exception as err:
        logger.debug("Failed to log best run to MLflow: %s", err)


def _resolve_optuna_storage(storage_uri: str | None) -> str | None:
    """Resolve and configure Optuna storage backend with SQLite WAL mode.

    Why:
        Concurrent access between ephemeral Jenkins training pods and the Optuna
        dashboard on SQLite requires Write-Ahead Logging (WAL) to prevent database
        locks and corruption. Pre-initializing the SQLite database with WAL mode and
        normal synchronization enables high-throughput concurrent reads and writes.

    How:
        1. Returns None if storage_uri is None, empty, or "memory".
        2. If starting with "sqlite:///", extracts the filesystem path and creates
           parent directories.
        3. Opens an SQLite connection with 30s timeout and executes WAL PRAGMAs.
        4. Returns the storage URI string ready for Optuna study creation.

    Complexity:
        - time complexity: O(1) file metadata check and database PRAGMA execution.
        - space complexity: O(1) in-memory connection handle.

    Args:
        storage_uri: Target database URI or None.

    Returns:
        Configured database URI or None for in-memory studies.
    """
    if not storage_uri or storage_uri == "memory":
        return None
    if storage_uri.startswith("sqlite:///"):
        db_path = storage_uri[len("sqlite:///") :]
        target_path = Path(db_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        finally:
            conn.close()
        return storage_uri
    return storage_uri


def _normalize_trial_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize hyperparameter keys to canonical learning_rate and cb_beta.

    Args:
        params: Raw parameter mapping from Optuna trial.

    Returns:
        Dictionary with standardized parameter names and float/int types.
    """
    lr_val = float(params.get("learning_rate", params.get("lr", 1e-3)))
    beta_val = float(params.get("cb_beta", params.get("beta", 0.9999)))
    return {
        "learning_rate": lr_val,
        "weight_decay": float(params.get("weight_decay", 1e-4)),
        "dropout": float(params.get("dropout", 0.2)),
        "cb_beta": beta_val,
        "noise_std": float(params.get("noise_std", 0.01)),
        "batch_size": int(params.get("batch_size", 32)),
    }


def _export_trials_history(study: optuna.Study, reports_dir: Path) -> None:
    """Export Optuna trials dataframe to Parquet report.

    Args:
        study: Completed Optuna study instance.
        reports_dir: Target directory for persisted reports.
    """
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        trials_df = study.trials_dataframe()
        parquet_path = reports_dir / "tuning_trials.parquet"
        trials_df.to_parquet(parquet_path, index=False)
        logger.info("Exported Optuna trials history to %s", parquet_path)
    except Exception as err:
        logger.warning("Could not export trials dataframe to parquet: %s", err)


def _persist_best_params(
    study: optuna.Study,
    extractor: str,
    checkpoint_dir: Path,
) -> Path | None:
    """Persist best trial hyperparameters to YAML keyed by extractor.

    Args:
        study: Completed Optuna study instance.
        extractor: Vision backbone extractor identifier.
        checkpoint_dir: Target directory for checkpoint persistence.

    Returns:
        Path to best_params.yaml on success, or None on failure.
    """
    if len(study.trials) == 0:
        return None
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_params_yaml = checkpoint_dir / "best_params.yaml"

        existing_data: dict[str, Any] = {}
        if best_params_yaml.is_file():
            try:
                with best_params_yaml.open("r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        existing_data = loaded
            except Exception as read_err:
                logger.warning("Could not parse existing best_params.yaml: %s", read_err)

        normalized_best = _normalize_trial_params(study.best_params)
        best_val = float(study.best_value) if study.best_trials else 0.0

        existing_data[extractor] = {
            "best_params": normalized_best,
            "best_val_pr_auc": round(best_val, 6),
            "n_trials": len(study.trials),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "dvc_dataset_hash": get_dvc_dataset_hash(),
        }

        with best_params_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(existing_data, f, sort_keys=False)
        logger.info("Saved best hyperparameters for '%s' to %s", extractor, best_params_yaml)
        return best_params_yaml
    except Exception as err:
        logger.warning("Failed to persist best params for '%s': %s", extractor, err)
        return None


def tune_hyperparameters(
    data: EmbeddingSplit,
    num_classes: int,
    config: TuningConfig | None = None,
    checkpoint_dir: Path | str = "models/checkpoints",
    reports_dir: Path | str = "reports",
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
           with `min_resource` and `reduction_factor`.
        3. Instantiates `HyperparameterObjective` and executes `study.optimize` across `n_trials`.
        4. Logs the optimal parameter configuration and peak Macro PR-AUC to MLflow as a consolidated run.
        5. Persists optimal parameters to models/checkpoints/best_params.yaml and trials history to reports/tuning_trials.parquet.
        6. Returns the best parameters dictionary and the full Optuna study object.

    Args:
        data: Pre-computed feature representations and labels.
        num_classes: Total number of target species or taxa categories.
        config: Tuning configuration controlling trials, epochs, batch size, seed, and MLflow.
        checkpoint_dir: Destination directory for best_params.yaml persistence.
        reports_dir: Destination directory for tuning_trials.parquet export.

    Returns:
        A tuple `(best_params, study)` of the best configuration and the Optuna study.
    """
    cfg = config or TuningConfig()
    settings = get_settings()
    mlflow_available = _setup_mlflow(cfg)

    storage_uri = _resolve_optuna_storage(cfg.storage_url or settings.optuna.optuna_storage_uri)
    study_prefix = settings.optuna.optuna_study_prefix
    study_name = cfg.study_name or f"{study_prefix}_{cfg.extractor_name}"

    if cfg.reset_study and storage_uri:
        with contextlib.suppress(Exception):
            optuna.delete_study(study_name=study_name, storage=storage_uri)
            logger.info("Deleted prior Optuna study '%s' to reset Bayesian history.", study_name)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=cfg.min_resource,
        reduction_factor=cfg.reduction_factor,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=not cfg.reset_study,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = HyperparameterObjective(
        data=data,
        num_classes=num_classes,
        config=cfg,
        mlflow_available=mlflow_available,
    )

    study.optimize(objective, n_trials=cfg.n_trials)

    best_params = study.best_params
    if "learning_rate" in best_params and "lr" not in best_params:
        best_params["lr"] = best_params["learning_rate"]
    if "cb_beta" in best_params and "beta" not in best_params:
        best_params["beta"] = best_params["cb_beta"]

    logger.info("Hyperparameter search complete. Best parameters: %s, Best value: %.4f", best_params, study.best_value)

    _persist_best_params(study, cfg.extractor_name, Path(checkpoint_dir))
    _export_trials_history(study, Path(reports_dir))

    _log_best_run(best_params, study.best_value, mlflow_available, study=study)
    return best_params, study
