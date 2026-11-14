# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Automated hyperparameter optimization with Optuna and MLflow."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

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
from taxon_vision.models.training.tracking import (
    _ensure_dagshub_auth,
    _inject_lineage_tags,
    _resolve_reachable_tracking_uri,
)
from taxon_vision.models.training.types import EmbeddingSplit, HeadTrainingConfig, TuningConfig
from taxon_vision.models.training.visualizations import log_optuna_diagnostic_plots

warnings.filterwarnings("ignore", category=ExperimentalWarning)

logger = logging.getLogger(__name__)


class TrialComponents(NamedTuple):
    """Instantiated classification head, loss criterion, and optimizer for an Optuna trial."""

    head: nn.Module
    criterion: ClassBalancedLoss
    optimizer: optim.Optimizer


class TuningResult(NamedTuple):
    """Best hyperparameter mapping paired with the completed Optuna Study object."""

    best_params: dict[str, Any]
    study: optuna.Study


@dataclass(frozen=True)
class _TrialScoreSummary:
    """Calibration-penalised trial objective breakdown for Optuna scoring and MLflow telemetry."""

    objective_score: float
    peak_val_pr_auc: float
    min_val_loss: float


def _compute_calibrated_objective_score(
    history: dict[str, list[float]],
    val_loss_penalty: float,
) -> _TrialScoreSummary:
    """Compute a calibration-aware HPO score penalising high validation loss alongside Macro PR-AUC.

    Why:
        Pure rank-based Macro PR-AUC is scale-invariant to logit magnitude, allowing Bayesian search to eliminate L2 weight decay and inflate linear head norms until softmax saturates on out-of-distribution inputs. Dividing peak PR-AUC by ``1 + penalty * min_val_loss`` penalises overconfident validation misclassifications whilst remaining strictly bounded within ``[0.0, 1.0]``.

    How:
        Extracts the maximum ``val_pr_auc`` and minimum ``val_loss`` across epochs in ``history`` and returns ``_TrialScoreSummary`` carrying the penalised scalar score and raw component metrics.

    Args:
        history: Epoch metric history produced by ``train_head_on_cached_embeddings``.
        val_loss_penalty: Non-negative damping coefficient applied to minimum validation loss.

    Returns:
        Frozen ``_TrialScoreSummary`` containing ``objective_score``, ``peak_val_pr_auc``, and ``min_val_loss``.
    """
    val_prs = history.get("val_pr_auc")
    val_losses = history.get("val_loss")
    peak_pr = float(max(val_prs)) if val_prs else 0.0
    min_loss = float(min(val_losses)) if val_losses else 0.0
    penalty_factor = 1.0 + max(0.0, val_loss_penalty) * max(0.0, min_loss)
    return _TrialScoreSummary(
        objective_score=peak_pr / penalty_factor,
        peak_val_pr_auc=peak_pr,
        min_val_loss=min_loss,
    )


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
        """Sample regularised hyperparameters within configured bounds for the current Optuna trial.

        Why:
            Unbounded learning rates and weak L2 weight decay floors (`1e-6`) cause linear classification head weights to grow excessively (`||W|| >> 10`), producing saturated softmax probabilities on out-of-distribution specimens. Constraining sampling to `self.config`'s regularised domain preserves moderate logit scales.

        How:
            Queries Optuna TPE continuous/categorical samplers using bounds configured on `self.config`.

        Args:
            trial: Optuna trial instance.

        Returns:
            Dictionary containing sampled hyperparameters.
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", self.config.lr_min, self.config.lr_max, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", self.config.weight_decay_min, self.config.weight_decay_max, log=True
            ),
            "dropout": trial.suggest_float("dropout", self.config.dropout_min, self.config.dropout_max),
            "cb_beta": trial.suggest_float("cb_beta", 0.99, 0.9999),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.05),
        }

    def _construct_components(self, params: dict[str, Any]) -> TrialComponents:
        """Construct the classification head, loss criterion, and optimizer.

        Args:
            params: Hyperparameter configuration.

        Returns:
            TrialComponents containing the classification head, loss criterion, and optimizer.
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
        return TrialComponents(head=head, criterion=criterion, optimizer=optimizer)

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
                if active_run:
                    epochs_completed = len(history.get("val_pr_auc", []))
                    mlflow.set_tag("optuna.trial_state", "PRUNED")
                    mlflow.set_tag("optuna.pruned_at_epoch", str(epochs_completed))
                    mlflow.end_run(status="KILLED")
                    active_run = None
                raise optuna.TrialPruned()

            score_summary = _compute_calibrated_objective_score(history, self.config.val_loss_penalty)
            if active_run:
                try:
                    mlflow.set_tag("optuna.trial_state", "COMPLETE")
                    mlflow.log_metric("peak_val_pr_auc", score_summary.peak_val_pr_auc)
                    mlflow.log_metric("min_val_loss", score_summary.min_val_loss)
                except Exception as metric_err:
                    logger.debug("Failed to log peak_val_pr_auc metric: %s", metric_err)
            return score_summary.objective_score

        finally:
            if active_run is not None:
                with contextlib.suppress(Exception):
                    mlflow.end_run(status="FINISHED")


def _setup_mlflow(cfg: TuningConfig) -> bool:
    """Configure MLflow for hyperparameter tuning via DagsHub or an explicit environment override.

    Args:
        cfg: Tuning configuration.

    Returns:
        True if MLflow tracking is reachable and authenticated, False otherwise.
    """
    if mlflow.active_run() is not None:
        return True

    settings = get_settings()
    mlflow_cfg = settings.mlflow
    resolved_experiment = cfg.experiment_name or mlflow_cfg.experiment_name

    if cfg.tracking_uri is not None:
        resolved_uri = cfg.tracking_uri.strip()
        if not resolved_uri:
            return False
        if "dagshub.com" in resolved_uri and not _ensure_dagshub_auth(resolved_uri):
            return False
    else:
        resolved_uri = _resolve_reachable_tracking_uri(settings)

    if not resolved_uri:
        return False

    try:
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(resolved_experiment)
        return True
    except Exception as exc:
        logger.warning("MLflow tracking setup failed or remote server unreachable: %s. Skipping HPO MLflow logs.", exc)
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
            mlflow.log_metric("best_val_pr_auc", best_value)
            mlflow.set_tag("optuna.optimization_algorithm", "TPE+ASHA")
            mlflow.set_tag("optuna.best_trial_number", str(study.best_trial.number) if study else "0")
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


def _run_optuna_study_in_mlflow(
    study: optuna.Study,
    objective: HyperparameterObjective,
    study_name: str,
) -> bool:
    """Execute Optuna optimisation under a parent MLflow run when MLflow is active outside an existing run."""
    started_study_parent = objective.mlflow_available and mlflow.active_run() is None
    if not started_study_parent:
        study.optimize(objective, n_trials=objective.config.n_trials)
        return False
    with mlflow.start_run(run_name=f"optuna_study_{study_name}"):
        study.optimize(objective, n_trials=objective.config.n_trials)
        with contextlib.suppress(Exception):
            mlflow.log_params(study.best_params)
            _apply_git_lineage_tags()
            dvc_hash = get_dvc_dataset_hash()
            if dvc_hash:
                mlflow.log_param("dvc_dataset_hash", dvc_hash)
        _log_best_run(study.best_params, study.best_value, objective.mlflow_available, study=study)
    return True


def _with_legacy_param_aliases(raw_params: dict[str, Any]) -> dict[str, Any]:
    """Copy Optuna best_params with legacy `lr` and `beta` aliases populated alongside canonical keys."""
    best_params = dict(raw_params)
    if "learning_rate" in best_params and "lr" not in best_params:
        best_params["lr"] = best_params["learning_rate"]
    if "cb_beta" in best_params and "beta" not in best_params:
        best_params["beta"] = best_params["cb_beta"]
    return best_params


def tune_hyperparameters(
    data: EmbeddingSplit,
    num_classes: int,
    config: TuningConfig | None = None,
    checkpoint_dir: Path | str = "models/checkpoints",
    reports_dir: Path | str = "reports",
) -> TuningResult:
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
        TuningResult `(best_params, study)` of the best configuration and the Optuna study.
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

    started_study_parent = _run_optuna_study_in_mlflow(study, objective, study_name)

    best_params = _with_legacy_param_aliases(study.best_params)

    logger.info("Hyperparameter search complete. Best parameters: %s, Best value: %.4f", best_params, study.best_value)

    _persist_best_params(study, cfg.extractor_name, Path(checkpoint_dir))
    _export_trials_history(study, Path(reports_dir))

    if not started_study_parent:
        _log_best_run(best_params, study.best_value, mlflow_available, study=study)
    return TuningResult(best_params=best_params, study=study)
