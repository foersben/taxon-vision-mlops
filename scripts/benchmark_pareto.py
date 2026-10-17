#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""CLI shim for Stage 7: Pareto Benchmark & Holdout Statistical Evaluation.

Delegates directly to canonical implementation in taxon_vision.pipelines.evaluate.
"""

from __future__ import annotations

import sys

from taxon_vision.pipelines.evaluate import (
    ParetoResult,
    evaluate_cli,
    get_canonical_pareto_profiles,
    main,
)

__all__ = ["ParetoResult", "evaluate_cli", "get_canonical_pareto_profiles", "main"]

if __name__ == "__main__":
    sys.exit(evaluate_cli())
