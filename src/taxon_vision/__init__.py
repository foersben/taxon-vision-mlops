# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""TaxonVision-MLOps: Automated Species Identification Service."""

from __future__ import annotations

import importlib.metadata

try:
    __version__: str = importlib.metadata.version("taxon-vision-mlops")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"
