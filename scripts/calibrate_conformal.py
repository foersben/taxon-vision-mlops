#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""CLI shim for Stage 6: Conformal Prediction Calibration.

Delegates directly to canonical implementation in taxon_vision.pipelines.calibrate.
"""

from __future__ import annotations

import sys

from taxon_vision.pipelines.calibrate import (
    calibrate_cli,
    load_embeddings_and_labels,
    main,
)

__all__ = ["calibrate_cli", "load_embeddings_and_labels", "main"]

if __name__ == "__main__":
    sys.exit(calibrate_cli())
