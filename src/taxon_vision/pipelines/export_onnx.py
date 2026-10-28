# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 5: ONNX Model Export & Quantization Pipeline.

First-class CLI entrypoint that traces trained PyTorch backbone representations and classification head into a production ONNX graph with optional INT8 quantization.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from taxon_vision.config import get_settings
from taxon_vision.inference.onnx_exporter import (
    _build_and_restore_classifier,
    _finalize_export_artifacts,
    _quantize_exported_graph,
    _resolve_export_output,
    export_to_onnx,
)
from taxon_vision.models.head import TaxonClassifier

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for ONNX model export and quantization."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Export TaxonClassifier PyTorch checkpoint to ONNX format with optional INT8 quantization."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained head checkpoint .pt file. Defaults to settings.model.head_checkpoint_path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination path for base FP32 ONNX graph. Defaults to models/checkpoints/model.onnx.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default=None,
        help="Feature extractor backbone architecture name. Defaults to settings.model.default_extractor.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image resolution in pixels. Defaults to settings.model.image_size.",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        default=settings.model.allow_untrained_fallback,
        help="Allow exporting with uninitialized/offline weights (for testing only).",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Apply INT8 quantization after ONNX export (mode selected via --quantize-mode).",
    )
    parser.add_argument(
        "--quantize-mode",
        choices=["auto", "static", "dynamic"],
        default=None,
        help=(
            "static: QDQ PTQ with calibration images (required for Conv/ViT backbone graphs). "
            "dynamic: MatMul/Gemm only, no data (linear-only graphs, CI smoke tests). "
            "auto: static for the full backbone graph. Defaults to settings.model.quantization_mode."
        ),
    )
    parser.add_argument(
        "--quantized-output",
        type=str,
        default=None,
        help="Destination of the INT8 graph. Defaults to settings.model.onnx_model_path.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory of representative images for static PTQ. Defaults to settings.model.calibration_dir.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Maximum calibration images for static PTQ. Defaults to settings.model.calibration_samples.",
    )
    return parser


def _export_dedicated_head(
    classifier: TaxonClassifier,
    target_artifact: Path,
    quantize_int8: bool,
) -> None:
    """Export the standalone classification head as an ONNX graph for embedding projections.

    Why:
        Executing linear classification on pre-computed feature representations via ONNX Runtime
        avoids parsing raw tensor initializers or loading PyTorch models in inference runtimes.

    Args:
        classifier: Trained TaxonClassifier containing the linear projection head.
        target_artifact: Path to the exported model graph (used to determine checkpoint directory).
        quantize_int8: Whether dynamic INT8 quantization should also be applied to the head.
    """
    head_dir = target_artifact.parent
    head_fp32 = head_dir / "head.onnx"
    try:
        logger.info("Exporting dedicated classification head ONNX graph to %s...", head_fp32)
        export_to_onnx(classifier.head, head_fp32, input_shape=(1, classifier.feature_dim))
        if quantize_int8:
            from taxon_vision.inference.quantizer import quantize_onnx_dynamic

            head_int8 = head_dir / "head_int8.onnx"
            quantize_onnx_dynamic(head_fp32, head_int8)
            logger.info("Quantized dedicated classification head ONNX graph saved to %s", head_int8)
            if head_fp32.exists():
                head_fp32.unlink()
    except Exception as exc:
        logger.warning("Could not export dedicated classification head ONNX graph: %s", exc)


def export_onnx_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 5: ONNX Model Export & Quantization.

    Args:
        argv: Optional list of command-line argument strings.

    Returns:
        Exit code: 0 for success, 1 on failure.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    output_file, temp_fp32_path = _resolve_export_output(args.output, args.quantize_int8)

    try:
        settings = get_settings()
        image_dim = args.image_size or settings.model.image_size

        classifier = _build_and_restore_classifier(args, settings)
        if classifier is None:
            return 1

        logger.info(
            "Exporting computational graph to ONNX (%s) with input shape (1, 3, %d, %d)...",
            output_file,
            image_dim,
            image_dim,
        )
        export_to_onnx(classifier, output_file, input_shape=(1, 3, image_dim, image_dim))

        target_artifact: Path = output_file
        if args.quantize_int8:
            target_artifact = _quantize_exported_graph(output_file, args, settings)

        _export_dedicated_head(classifier, target_artifact, args.quantize_int8)

        import mlflow

        if mlflow.active_run():
            from taxon_vision.models.training.tracking import MLflowTracker

            MLflowTracker.log_onnx_model(
                onnx_model=target_artifact,
                artifact_path="model",
                num_classes=classifier.num_classes,
                image_size=image_dim,
                registered_model_name="taxon_vision_onnx",
            )

        from taxon_vision.models.training.tracking import log_stage_metrics_to_parent_run

        file_size_mb = target_artifact.stat().st_size / (1024 * 1024) if target_artifact.exists() else 0.0
        log_stage_metrics_to_parent_run(
            metrics={
                "onnx_model_size_mb": round(file_size_mb, 2),
                "quantized_int8": 1.0 if args.quantize_int8 else 0.0,
            },
            prefix="export.",
        )

        return _finalize_export_artifacts(target_artifact, args.output, args.quantize_int8)

    except Exception as err:
        logger.exception("Fatal error during ONNX export: %s", err)
        return 1
    finally:
        if temp_fp32_path is not None and temp_fp32_path.exists():
            temp_fp32_path.unlink()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 5: ONNX Model Export & Quantization."""
    return export_onnx_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
