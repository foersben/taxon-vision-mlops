# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Taxonomic revision and ground truth mutation handler."""

from __future__ import annotations


def reconcile_taxonomic_mutation(old_taxon_id: int, new_taxon_id: int, mapping: dict[int, int]) -> int:
    """Resolve species splits, lumps, and reclassifications."""
    return mapping.get(old_taxon_id, new_taxon_id)
