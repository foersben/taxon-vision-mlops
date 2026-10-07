# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""CLI entrypoint for running species inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from taxon_vision.service.inference import run_prediction

logger = logging.getLogger(__name__)


def _build_terminal_report(result: dict[str, Any], img_path: Path) -> str:
    """Format inference output as structured terminal report.

    The confidence describes the probability of the prediction being correct. The conformal set is a set of possible taxa that could be the correct one with a confidence of 95%.

    Args:
        result: Raw prediction dictionary.
        img_path: Source observation image path.

    Returns:
        Formatted multi-line report string.
    """
    conf_pct = float(str(result.get("confidence", 0.0))) * 100.0
    latency = float(str(result.get("latency_ms", 0.0)))
    c_set = result.get("conformal_set", [])
    review_needed = result.get("requires_human_review", False)

    lines = [
        "------------------------------------------------------------",
        "           TaxonVision Species Prediction Report",
        "------------------------------------------------------------",
        f"Image Path:           {img_path}",
        f"Scientific Name:      {result.get('scientific_name')}",
        f"Common Name:          {result.get('common_name')}",
        f"Confidence:           {conf_pct:.2f}%",
        f"Conformal Set (95%):  {c_set}",
        f"Human Triage Needed:  {review_needed}",
        f"Latency:              {latency:.2f} ms",
        "------------------------------------------------------------",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI prediction entrypoint.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(description="TaxonVision Species Prediction CLI")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image file (defaults to data/sample/406185885_taxon_47120.jpg)",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    raw_path = args.image or "data/sample/406185885_taxon_47120.jpg"
    img_path = Path(raw_path)

    if not img_path.exists():
        logger.error("Image file not found at: %s", img_path)
        return 1

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    result = run_prediction(image_bytes)

    if args.json:
        formatted_json = {
            "scientific_name": str(result["scientific_name"]),
            "common_name": str(result["common_name"]),
            "confidence": float(str(result["confidence"])),
            "conformal_set": list(result["conformal_set"]),
            "requires_human_review": bool(result["requires_human_review"]),
            "latency_ms": float(str(result["latency_ms"])),
        }
        print(json.dumps(formatted_json, indent=2))
        return 0

    print(_build_terminal_report(result, img_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
