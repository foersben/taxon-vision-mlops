# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Pydantic v2 schemas for API contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ConformalRoutingTier(StrEnum):
    """Four-tier epistemic safety and conformal routing contract.

    Attributes:
        DECISIVE: |C| = 1 and pi_max >= tau_conf (immediate automated resolution).
        MODERATE_AMBIGUITY: 1 < |C| <= 3 (triggers LanceDB metadata-filtered HNSW + RRF retrieval).
        SEVERE_AMBIGUITY: |C| > 3 or (|C| = 1 and pi_max < tau_conf) (routes to expert sorting queue).
        ABSTENTION: |C| = 0 (empty prediction set, rejected out of hand).
    """

    DECISIVE = "DECISIVE"
    MODERATE_AMBIGUITY = "MODERATE_AMBIGUITY"
    SEVERE_AMBIGUITY = "SEVERE_AMBIGUITY"
    ABSTENTION = "ABSTENTION"


class TaxonPrediction(BaseModel):
    """Predicted taxonomic identity with calibrated confidence score.

    Why:
        Establishes an unambiguous representation of individual candidate species, linking machine-learning probability outputs to canonical Darwin Core identifiers and binomial scientific nomenclature.

    How:
        Encapsulates integer taxon identifier, binomial scientific name, vernacular common name, and top-1 probability score rounded to four decimal places.

    Attributes:
        taxon_id: Unique integer identifier of the taxon from the knowledge base.
        scientific_name: Binomial scientific name of the taxon.
        common_name: Vernacular common name of the taxon.
        confidence: Probability score of the prediction, rounded to four decimal places.
    """

    taxon_id: int
    scientific_name: str
    common_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """Structured inference response containing top predictions, conformal sets, and triage flags.

    Why:
        Production ecological inference must not return uncalibrated point estimates in isolation. Clients require both the primary identification, ranked alternatives, distribution-free conformal guarantee sets, and actionable triage recommendations (human review required, OOD flags) to safely automate field workflows.

    How:
        Bundles the top-1 `TaxonPrediction`, secondary candidates admitted into the conformal set, conformal set scientific names list, 4-tier conformal routing contract state, ambiguity boolean flags, epistemic referral triggers, and end-to-end execution latency in milliseconds.

    Attributes:
        top_prediction: The top predicted taxon with the highest confidence score.
        top_candidates: A list of top predicted taxa, including the top prediction and secondary candidates.
        conformal_prediction_set: A list of scientific names of the taxa in the conformal set.
    """

    top_prediction: TaxonPrediction
    top_candidates: list[TaxonPrediction]
    conformal_prediction_set: list[str]
    routing_tier: ConformalRoutingTier = ConformalRoutingTier.DECISIVE
    is_conformal_ambiguous: bool
    is_ood_flagged: bool
    requires_human_review: bool
    latency_ms: float
    cam_heatmap_base64: str | None = None


class TrainingRequest(BaseModel):
    """Parameters governing model head training execution.

    Why:
        Allows callers to trigger and configure linear classification head optimization over cached foundation model representations via REST API without altering underlying server configuration files.

    How:
        Defines optional backbone extractor name, epoch budget, batch size, optimizer learning rate, and a background dispatch flag (`background=True`) to offload long jobs. Unspecified hyperparameters resolve dynamically from centralized application settings.

    Attributes:
        extractor: Name of the backbone extractor to use for training.
        epochs: Number of epochs to train for.
        batch_size: Batch size to use for training.
        learning_rate: Learning rate to use for training.
        background: Whether to train in the background.
    """

    extractor: str | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    background: bool = False


class TrainingResponse(BaseModel):
    """Execution status and evaluation metrics from a model head training run.

    Why:
        Communicates the convergence outcomes, weight artifact destinations, and operational status of local or remote training pipelines to human operators and MLOps controllers.

    How:
        Serializes operational status ('started' or 'completed'), target backbone extractor, completed epochs count, final train and validation losses, top-1 accuracy, persisted checkpoint path, total elapsed duration, and summary status message.

    Attributes:
        status: Status of the training run.
        extractor: Backbone extractor used for training.
        epochs_trained: Number of epochs trained.
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
        final_val_accuracy: Final validation accuracy.
        checkpoint_path: Path to the trained model checkpoint.
        duration_seconds: Total duration of the training run in seconds.
        message: Summary status message.
    """

    status: str
    extractor: str
    epochs_trained: int
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    final_val_accuracy: float | None = None
    checkpoint_path: str
    duration_seconds: float
    message: str


class DriftEvaluationResponse(BaseModel):
    """Outcome of an on-demand statistical drift evaluation cycle.

    Why:
        Supports proactive model maintenance by providing automated, data-driven evidence of representational drift between reference and serving data.

    How:
        Runs statistical divergence tests (W1 Earth Mover's Distance, optionally MMD²) on sliding windows of production embeddings against a fixed reference set. Returns raw metrics, configurable thresholds, binary detection flags, and comprehensive artifacts for logging and reporting.

    Attributes:
        drift_detected: Whether drift was detected.
        status: Status of the drift evaluation.
        w1_distance: W1 distance between reference and window distributions.
        w1_threshold: Decision threshold applied to W1 distance.
    """

    drift_detected: bool
    status: str
    w1_distance: float
    w1_threshold: float
    mmd2: float | None = None
    mmd2_p_value: float | None = None
    window_size: int
    n_reference: int
    mlflow_run_id: str | None = None
    report_path: str | None = None


class DriftRunSummary(BaseModel):
    """Summary of the most recently recorded drift evaluation run.

    Why:
        Provides a high-level, time-ordered view of drift evaluation results without the full computational context (data paths, full stats, MLflow runs).

    How:
        Serializes key metrics (W1 distance, MMD², p-value), flags (drift detected, reference size), and structural metadata (created time, window size).

    Attributes:
        id: Unique database identifier of the drift run.
        created_at: UTC timestamp when the drift evaluation was executed.
        window_size: Number of recent production samples used for the sliding window.
        w1_distance: Earth Mover's Distance (W1) between reference and window distributions.
        w1_threshold: Decision threshold applied to W1 distance.
        mmd2: Squared Maximum Mean Discrepancy (MMD²) between distributions.
        mmd2_p_value: Monte Carlo p-value for the MMD² statistic.
        drift_detected: Boolean flag indicating if drift was detected (True if W1 > threshold).
        mlflow_run_id: Optional MLflow run identifier for reproducibility and logging.
    """

    id: int | None = None
    created_at: str | None = None
    window_size: int
    w1_distance: float
    w1_threshold: float
    mmd2: float | None = None
    mmd2_p_value: float | None = None
    drift_detected: bool
    mlflow_run_id: str | None = None
