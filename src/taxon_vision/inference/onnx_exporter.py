# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Export PyTorch model to ONNX format."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.jit._trace import TracerWarning

from taxon_vision.config import Settings, get_settings
from taxon_vision.inference.calibration import load_calibration_tensors
from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
from taxon_vision.models.head import TaxonClassifier, load_head_checkpoint
from taxon_vision.service.inference import load_taxa_catalog

logger = logging.getLogger(__name__)


class _ONNXDualOutputModule(nn.Module):
    """Wrapper module ensuring exported ONNX graphs emit both logits and dense embeddings.

    Why:
        Standard PyTorch classifiers return only the final projection logits. Exporting an ONNX graph that provides both class logits and dense feature embeddings allows downstream inference systems (OOD detection, drift monitoring, active learning query strategies) to consume embeddings directly without running an additional feature extraction pass.

    How:
        Delegates forward execution to `classifier.forward_with_embeddings(x)`.

    Complexity:
        O(1) forward delegation overhead.
    """

    def __init__(self, classifier: TaxonClassifier) -> None:
        """Initialize the dual-output wrapper.

        Args:
            classifier: TaxonClassifier instance to wrap.
        """
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute forward pass returning (logits, embeddings).

        Args:
            x: Input 4D image tensor of shape (batch_size, channels, height, width).

        Returns:
            Tuple of (logits, embeddings).
        """
        return self.classifier.forward_with_embeddings(x)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...] | None = None,
) -> Path:
    """Export model to ONNX graph with dynamic batch sizing and dual outputs (logits, embeddings).

    Why:
        In production workflows, the ML model must be decoupled from the specific deep learning framework used during training to maximize deployment flexibility. Emitting dual outputs (logits, embeddings) allows inference engines to support both classification and downstream representation retrieval without redundant forward passes.

    How:
        Exports the trained PyTorch model to a self-contained ONNX graph using torch.onnx.export. Traces the graph using a dummy CPU input tensor, enables constant folding, specifies dynamic axes for variable batch sizes, and targets the configured opset version.

    Args:
        model: PyTorch model module to export.
        output_path: Target filesystem path for serialized ONNX graph.
        input_shape: Expected input tensor shape tuple. Defaults to (1, settings.model.channels, settings.model.image_size, settings.model.image_size).

    Returns:
        Resolved Path to the exported ONNX model.
    """
    settings = get_settings()
    model.eval()
    model.cpu()
    if input_shape is None:
        cfg = settings.model
        input_shape = (1, cfg.channels, cfg.image_size, cfg.image_size)
    dummy_input = torch.randn(*input_shape, device="cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_dual = isinstance(model, TaxonClassifier) or isinstance(model, _ONNXDualOutputModule)
    export_module: nn.Module = _ONNXDualOutputModule(model) if isinstance(model, TaxonClassifier) else model

    output_names = ["logits", "embeddings"] if is_dual else ["logits"]
    dynamic_axes: dict[str, dict[int, str]] = {
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }
    if is_dual:
        dynamic_axes["embeddings"] = {0: "batch_size"}

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        torch.onnx.export(
            export_module,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=settings.model.onnx_opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    return output_path


def _load_head_checkpoint(classifier: TaxonClassifier, checkpoint_file: Path) -> None:
    """Safely load trained classification head weights onto CPU.

    Why:
        Checkpoints may be saved from GPU devices (e.g. cuda:0) or wrapped inside dictionary structures. Deserializing strictly to CPU and normalizing keys ensures robust loading across heterogeneous training and deployment environments.

    How:
        Delegates to `load_head_checkpoint`, which loads with weights_only=True and map_location='cpu', unwraps nested checkpoint dictionaries, strips any leading 'head.' prefixes, and enforces strict state_dict restoration into classifier.head.

    Args:
        classifier: TaxonClassifier instance containing the target classification head.
        checkpoint_file: Path to the serialized PyTorch head checkpoint.
    """
    load_head_checkpoint(classifier.head, checkpoint_file)


def _resolve_backbone(
    extractor_name: str,
    fallback_extractor: str,
    allow_untrained: bool,
) -> nn.Module | None:
    """Instantiate the backbone feature extractor, respecting the untrained flag.

    Args:
        extractor_name: Name of the backbone feature extractor.
        fallback_extractor: Name of the fallback feature extractor.
        allow_untrained: Whether to allow untrained backbone.

    Returns:
        Backbone feature extractor.
    """
    use_pretrained = not allow_untrained
    logger.info("Initializing backbone feature extractor: %s (pretrained=%s)", extractor_name, use_pretrained)
    try:
        return create_feature_extractor(
            extractor_name,
            pretrained=use_pretrained,
            allow_fallback=allow_untrained,
        )
    except Exception as exc:
        logger.error(
            "Failed to create backbone '%s' (fallback '%s' not usable): %s", extractor_name, fallback_extractor, exc
        )
        return None


def _restore_head(
    classifier: TaxonClassifier,
    checkpoint_file: Path,
    allow_untrained: bool,
    extractor_name: str,
    feat_dim: int,
    num_classes: int,
) -> bool:
    """Restore checkpoint weights or validate fallback behavior.

    Args:
        classifier: Classifier module to load weights into.
        checkpoint_file: Path to checkpoint file.
        allow_untrained: Whether to allow untrained backbone.
        extractor_name: Name of the feature extractor.
        feat_dim: Feature dimension.
        num_classes: Number of classes.

    Returns:
        True if head was restored or fallback is allowed, False otherwise.
    """
    if not checkpoint_file.exists():
        if not allow_untrained:
            logger.error(
                "Trained checkpoint not found at %s. Run training first or pass --allow-untrained for tests.",
                checkpoint_file,
            )
            return False
        logger.warning("No checkpoint at %s; exporting initialized weights (--allow-untrained)", checkpoint_file)
        return True

    try:
        _load_head_checkpoint(classifier, checkpoint_file)
        logger.info("Restored trained head weights from %s", checkpoint_file)
        return True
    except Exception as load_err:
        if not allow_untrained:
            logger.error(
                "Checkpoint %s is incompatible with backbone '%s' (feat_dim=%d, classes=%d): %s",
                checkpoint_file,
                extractor_name,
                feat_dim,
                num_classes,
                load_err,
            )
            return False
        logger.warning("Checkpoint shape mismatch (%s); exporting untrained weights (--allow-untrained)", load_err)
        return True


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the ONNX exporter CLI.

    Returns:
        ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="TaxonVision ONNX Graph Exporter CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch head checkpoint (.pt). Defaults to settings.model.head_checkpoint_path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path for serialized FP32 ONNX model graph.",
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
    settings = get_settings()
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        default=settings.model.allow_untrained_fallback,
        help="Allow exporting with uninitialized/offline weights (for unit testing only).",
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


def _quantize_exported_graph(fp32_path: Path, args: argparse.Namespace, settings: Settings) -> Path:
    """Quantize the exported FP32 backbone+head graph to INT8 using the operator-appropriate mode.

    Why:
        The exported graph contains the full backbone (convolutions or ViT blocks), so static QDQ PTQ with real calibration images is the correct default. Dynamic quantization remains available explicitly for data-free smoke tests, but is never selected silently for a backbone graph.

    Args:
        fp32_path: Path to the exported FP32 ONNX graph.
        args: Parsed CLI arguments.
        settings: Application settings providing defaults.

    Returns:
        Path to the quantized INT8 graph.

    Raises:
        FileNotFoundError: If static calibration is selected and no calibration images are available.
    """
    from taxon_vision.inference.quantizer import quantize_onnx_model

    model_cfg = settings.model
    requested = args.quantize_mode or model_cfg.quantization_mode
    cal_dir = Path(args.calibration_dir or model_cfg.calibration_dir)
    calibration = None
    mode: Literal["auto", "static", "dynamic"]
    if requested == "static":
        calibration = load_calibration_tensors(
            cal_dir,
            args.calibration_samples or model_cfg.calibration_samples,
        )
        mode = "static"
    elif requested == "auto":
        if cal_dir.exists() and any(cal_dir.glob("*.*")):
            calibration = load_calibration_tensors(
                cal_dir,
                args.calibration_samples or model_cfg.calibration_samples,
            )
            mode = "static"
        else:
            mode = "dynamic"
    elif requested == "dynamic":
        mode = "dynamic"
    else:
        mode = "auto"

    default_quantized = (
        fp32_path.with_name(f"{fp32_path.stem}.int8.onnx")
        if args.output and args.output != "models/checkpoints/model.onnx"
        else Path(model_cfg.onnx_model_path)
    )
    quantized_path = Path(args.quantized_output or default_quantized)

    quantize_onnx_model(fp32_path, quantized_path, calibration_tensors=calibration, mode=mode)
    logger.info("Quantized INT8 ONNX graph (%s) saved to %s", mode, quantized_path)
    return quantized_path


def _resolve_export_output(output_arg: str | None, quantize_int8: bool) -> tuple[Path, Path | None]:
    """Resolve target output path and optional temporary FP32 path.

    Args:
        output_arg: CLI output argument.
        quantize_int8: Whether INT8 quantization is requested.

    Returns:
        Tuple of (output_path, temporary_fp32_path_or_none).
    """
    if output_arg is not None:
        return Path(output_arg), None
    if not quantize_int8:
        return Path("models/checkpoints/model.onnx"), None

    import tempfile

    temp_fp32 = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    temp_path = Path(temp_fp32.name)
    temp_fp32.close()
    return temp_path, temp_path


def _build_and_restore_classifier(args: argparse.Namespace, settings: Settings) -> TaxonClassifier | None:
    """Instantiate backbone, configure classifier, and restore head checkpoint.

    Args:
        args: Parsed command-line arguments.
        settings: Global application settings.

    Returns:
        TaxonClassifier instance or None on error.
    """
    extractor_name = args.extractor or settings.model.default_extractor
    checkpoint_file = Path(args.checkpoint or settings.model.head_checkpoint_path)

    backbone = _resolve_backbone(extractor_name, settings.model.fallback_extractor, args.allow_untrained)
    if backbone is None:
        return None

    feat_dim = get_feature_dimension(backbone)
    taxa = load_taxa_catalog()
    if not taxa:
        logger.error("Authoritative taxa catalog loaded from %s is empty", settings.model.taxa_catalog_path)
        return None

    num_classes = len(taxa)
    classifier = TaxonClassifier(backbone=backbone, feature_dim=feat_dim, num_classes=num_classes)

    if not _restore_head(
        classifier,
        checkpoint_file,
        args.allow_untrained,
        extractor_name,
        feat_dim,
        num_classes,
    ):
        return None
    return classifier


def _finalize_export_artifacts(target_artifact: Path, output_arg: str | None, quantize_int8: bool) -> int:
    """Validate exported model artifact existence and purge intermediate unquantized models.

    Args:
        target_artifact: Path to finalized output model graph.
        output_arg: Raw user-specified output path string.
        quantize_int8: Whether quantization was applied.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    if not target_artifact.exists() or target_artifact.stat().st_size == 0:
        logger.error("ONNX export failed: Output file %s was not created", target_artifact)
        return 1

    logger.info("ONNX export succeeded: %s (%d bytes)", target_artifact, target_artifact.stat().st_size)
    if output_arg is None and quantize_int8:
        stale_fp32 = Path("models/checkpoints/model.onnx")
        if stale_fp32.exists():
            stale_fp32.unlink()
    return 0


def export_classifier_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for exporting the trained TaxonClassifier to ONNX format.

    Why:
        Automation pipelines (e.g., Jenkins CI/CD, DVC) require a deterministic CLI tool to trace and serialize PyTorch checkpoints into high-performance ONNX computational graphs with optional INT8 quantization.

    How:
        1. Parses command-line flags (--checkpoint, --output, --extractor, --image-size, --allow-untrained, --quantize-int8, --quantize-mode, --quantized-output, --calibration-dir, --calibration-samples).
        2. Resolves configuration defaults from application settings.
        3. Instantiates the TaxonClassifier using the pretrained backbone (unless --allow-untrained is set).
        4. Restores trained weights from checkpoint using _load_head_checkpoint (failing if missing unless --allow-untrained).
        5. Traces and exports the model to ONNX via export_to_onnx.
        6. Optionally applies dynamic INT8 quantization via quantize_onnx_model.

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

        target_artifact = output_file
        if args.quantize_int8:
            target_artifact = _quantize_exported_graph(output_file, args, settings)

        return _finalize_export_artifacts(target_artifact, args.output, args.quantize_int8)

    except Exception as err:
        logger.exception("Fatal error during ONNX export: %s", err)
        return 1
    finally:
        if temp_fp32_path is not None and temp_fp32_path.exists():
            temp_fp32_path.unlink()


if __name__ == "__main__":
    sys.exit(export_classifier_cli())
