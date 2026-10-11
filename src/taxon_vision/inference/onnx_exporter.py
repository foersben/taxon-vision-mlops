# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Export PyTorch model to ONNX format."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(model: nn.Module, output_path: Path, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> Path:
    """Export model to ONNX graph with dynamic batch sizing.

    Why:
        In production workflows, the ML model must be decoupled from the specific deep learning framework used during training to maximize deployment flexibility. ONNX (Open Neural Network Exchange) serves as the de facto open-standard interoperability layer, enabling the model to be deployed using high-performance inference engines (e.g., ONNX Runtime, TensorRT) across diverse hardware targets (CPUs, GPUs, edge accelerators) without requiring the original training framework. This separation ensures the inference infrastructure can be optimized independently of the research and development environment.

    How:
        Exports the trained PyTorch model to a self-contained ONNX graph using torch.onnx.export. The export process traces the computational graph using a dummy input tensor, generating a serialized graph representation. Key optimization arguments include enabling constant folding to fuse operations (e.g., scaling, bias addition) into single nodes, specifying dynamic axes to permit variable batch sizes during inference, and selecting an opset version (e.g., 17) that balances feature availability with broad runtime compatibility.

    Complexity:
        - time complexity: O(M), where M is the number of layers in the model. The export process traces the computational graph once, performing constant folding and other optimizations that scale linearly with model size.
        - space complexity: O(M), due to the storage requirements for the ONNX graph representation and intermediate symbolic tensors during tracing.

    Args:
        model: The model to export.
        output_path: The path to export the model to.
        input_shape: The shape of the input to the model.

    Returns:
        The path to the exported model.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return output_path


def export_classifier_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for exporting the trained TaxonClassifier to ONNX format.

    Why:
        Automation pipelines (e.g., Jenkins CI/CD) and developers need a command-line interface
        to trace and serialize PyTorch checkpoints into high-performance ONNX computational graphs
        without writing bespoke Python scripts.

    How:
        1. Parses command-line flags (--checkpoint, --output, --extractor, --image-size).
        2. Resolves configuration defaults from application settings.
        3. Instantiates the TaxonClassifier using the specified backbone and restores weights from checkpoint.
        4. Invokes export_to_onnx with the requested input dimensions.
        5. Validates that the artifact was generated and logs confirmation.

    Args:
        argv: Optional list of command-line argument strings.

    Returns:
        Exit code: 0 for success, 1 on failure.
    """
    import argparse
    import logging
    import sys

    from taxon_vision.config import get_settings
    from taxon_vision.models.factory import create_feature_extractor, get_feature_dimension
    from taxon_vision.models.head import TaxonClassifier
    from taxon_vision.service.inference import load_taxa_catalog

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
        default="models/checkpoints/model.onnx",
        help="Destination path for serialized ONNX model graph.",
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

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    try:
        settings = get_settings()
        extractor_name = args.extractor or settings.model.default_extractor
        image_dim = args.image_size or settings.model.image_size
        output_file = Path(args.output)
        checkpoint_file = Path(args.checkpoint or settings.model.head_checkpoint_path)

        logger.info("Initializing backbone feature extractor: %s", extractor_name)
        try:
            backbone = create_feature_extractor(extractor_name, pretrained=False)
        except Exception as exc:
            logger.warning(
                "Could not create backbone %s (%s); attempting fallback %s",
                extractor_name,
                exc,
                settings.model.fallback_extractor,
            )
            backbone = create_feature_extractor(settings.model.fallback_extractor, pretrained=False)

        feat_dim = get_feature_dimension(backbone)
        taxa = load_taxa_catalog()
        num_classes = len(taxa) if taxa else 10

        classifier = TaxonClassifier(backbone=backbone, feature_dim=feat_dim, num_classes=num_classes)

        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, weights_only=True)
            if "1.weight" in state_dict and state_dict["1.weight"].shape == classifier.head[1].weight.shape:
                classifier.head.load_state_dict(state_dict)
                logger.info("Restored checkpoint weights from %s", checkpoint_file)
            else:
                logger.warning("Checkpoint %s shape mismatch; exporting with default weights", checkpoint_file)
        else:
            logger.info("No checkpoint found at %s; exporting model with initialized weights", checkpoint_file)

        logger.info(
            "Exporting computational graph to ONNX (%s) with input shape (1, 3, %d, %d)...",
            output_file,
            image_dim,
            image_dim,
        )
        export_to_onnx(classifier, output_file, input_shape=(1, 3, image_dim, image_dim))

        if output_file.exists():
            logger.info("ONNX export succeeded: %s (%d bytes)", output_file, output_file.stat().st_size)
            return 0

        logger.error("ONNX export failed: Output file %s was not created", output_file)
        return 1

    except Exception as err:
        logger.exception("Fatal error during ONNX export: %s", err)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(export_classifier_cli())
