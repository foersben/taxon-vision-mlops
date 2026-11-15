# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""TaxonVision-MLOps: Automated Species Identification Service."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("taxon-vision-mlops")
except PackageNotFoundError:
    __version__ = "0.2.0"
