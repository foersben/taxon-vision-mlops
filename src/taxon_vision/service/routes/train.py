# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI endpoint for model head training."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status

from taxon_vision.models.training.runner import run_training_pipeline
from taxon_vision.service.schemas import TrainingRequest, TrainingResponse

router = APIRouter(tags=["Training"])


@router.post(
    "/training",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger model head training",
)
@router.post(
    "/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger model head training (alias)",
)
@router.post(
    "/api/v1/training",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger model head training (API v1)",
)
@router.post(
    "/api/v1/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger model head training (API v1 alias)",
)
async def trigger_training(
    background_tasks: BackgroundTasks,
    payload: TrainingRequest | None = None,
) -> TrainingResponse:
    """Trigger classification head training with Class-Balanced Loss.

    Args:
        payload: Optional training configuration parameters.
        background_tasks: FastAPI background task manager.

    Returns:
        Structured response detailing execution status and evaluation metrics.
    """
    req = payload or TrainingRequest()

    if req.background:
        background_tasks.add_task(
            run_training_pipeline,
            extractor=req.extractor,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
        )
        return TrainingResponse(
            status="started",
            extractor=req.extractor,
            epochs_trained=0,
            checkpoint_path="models/checkpoints/head.pt",
            duration_seconds=0.0,
            message=f"Training job queued in background for {req.epochs} epochs.",
        )

    result = run_training_pipeline(
        extractor=req.extractor,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
    )

    return TrainingResponse(
        status=result["status"],
        extractor=result["extractor"],
        epochs_trained=result["epochs_trained"],
        final_train_loss=result["final_train_loss"],
        final_val_loss=result["final_val_loss"],
        final_val_accuracy=result["final_val_accuracy"],
        checkpoint_path=result["checkpoint_path"],
        duration_seconds=result["duration_seconds"],
        message=f"Training finished: {result['epochs_trained']} epochs on {result['extractor']} head.",
    )
