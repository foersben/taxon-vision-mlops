# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""DVC data versioning and manifest tracking."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_dataset_hash(directory: Path) -> str:
    """Compute deterministic SHA-256 hash across dataset directory.

    Args:
        directory: The directory to compute the hash for.

    Returns:
        A string representing the SHA-256 hash of the directory.
    """
    sha = hashlib.sha256()
    if not directory.exists():
        return sha.hexdigest()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            sha.update(p.read_bytes())
    return sha.hexdigest()
