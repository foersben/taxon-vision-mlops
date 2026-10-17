# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Stage 5: ONNX Model Export & Quantization Pipeline.

Traces trained PyTorch backbone representations and classification head into a production
ONNX graph with optional dynamic INT8 quantization.
"""

from __future__ import annotations

import sys

from taxon_vision.inference.onnx_exporter import export_classifier_cli


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 5: ONNX Model Export & Quantization."""
    return export_classifier_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
