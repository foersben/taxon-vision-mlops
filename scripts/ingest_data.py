# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""CLI shim for Stage 1: Ingestion.

Delegates to canonical implementation in taxon_vision.pipelines.ingest.
"""

from __future__ import annotations

import sys

from taxon_vision.pipelines.ingest import ingest_cli

if __name__ == "__main__":
    sys.exit(ingest_cli())
