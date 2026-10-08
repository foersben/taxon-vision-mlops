# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Pydantic v2 schemas for API contracts."""

from __future__ import annotations

from pydantic import BaseModel


class TaxonPrediction(BaseModel):
    """Predicted taxonomic identity with calibrated confidence score.

    Why:
        Establishes an unambiguous representation of individual candidate species,
        linking machine-learning probability outputs to canonical Darwin Core identifiers
        and binomial scientific nomenclature.

    How:
        Encapsulates integer taxon identifier, binomial scientific name, vernacular
        common name, and top-1 probability score rounded to four decimal places.
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
        Bundles the top-1 `TaxonPrediction`, secondary candidates admitted into the conformal set, conformal set scientific names list, ambiguity boolean flags, epistemic referral triggers, and end-to-end execution latency in milliseconds.
    """

    top_prediction: TaxonPrediction
    top_candidates: list[TaxonPrediction]
    conformal_prediction_set: list[str]
    is_conformal_ambiguous: bool
    is_ood_flagged: bool
    requires_human_review: bool
    latency_ms: float


class TrainingRequest(BaseModel):
    """Parameters governing model head training execution.

    Why:
        Allows callers to trigger and configure linear classification head optimization
        over cached foundation model representations via REST API without altering
        underlying server configuration files.

    How:
        Defines default backbone extractor name, epoch budget, batch size, optimizer
        learning rate, and a background dispatch flag (`background=True`) to offload long jobs.
    """

    extractor: str = "mobilenetv4_conv_small"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    background: bool = False


class TrainingResponse(BaseModel):
    """Execution status and evaluation metrics from a model head training run.

    Why:
        Communicates the convergence outcomes, weight artifact destinations, and operational
        status of local or remote training pipelines to human operators and MLOps controllers.

    How:
        Serializes operational status ('started' or 'completed'), target backbone extractor,
        completed epochs count, final train and validation losses, top-1 accuracy, persisted
        checkpoint path, total elapsed duration, and summary status message.
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
