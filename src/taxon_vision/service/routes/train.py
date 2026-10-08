# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""FastAPI endpoint for model head training."""

from __future__ import annotations

from functools import partial

import anyio
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

    Why:
        Training a neural classification head on cached embeddings involves intensive tensor linear algebra, matrix multiplications, and loss evaluations. In an asyncio FastAPI application, executing long-running CPU/GPU-bound tasks directly on the main event loop thread causes event loop starvation, freezing all concurrent requests (e.g. `/health`, `/metrics`, and `/predict`). Offloading execution either to FastAPI background tasks (when `req.background=True`) or to an asynchronous worker threadpool via AnyIO preserves server responsiveness and prevents HTTP connection timeouts.

    How:
        Parses optional TrainingRequest configuration, selecting between two execution modes:
        1. Asynchronous Background Task (`req.background=True`): Queues `run_training_pipeline` into FastAPI's BackgroundTasks runner and immediately returns an HTTP 200 `started` status with zero execution blocking.
        2. Synchronous Non-Blocking Threadpool (`req.background=False`): Dispatches `run_training_pipeline` to AnyIO's threadpool worker via `anyio.to_thread.run_sync`, awaiting completion without blocking the asyncio event loop, then returns full validation metrics and updated checkpoint paths.

    Args:
        background_tasks: FastAPI background task manager.
        payload: Optional training configuration parameters (epochs, batch size, learning rate).

    Returns:
        Structured TrainingResponse detailing training execution status and evaluation metrics.
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

    train_func = partial(
        run_training_pipeline,
        extractor=req.extractor,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
    )
    result = await anyio.to_thread.run_sync(train_func)

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
