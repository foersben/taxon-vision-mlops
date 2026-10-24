# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""View-model builders translating raw inference results into template-ready structures.

Keeping this logic out of the route handlers separates presentation concerns (ranking, rounding,
conformal-membership flags) from HTTP concerns (request parsing, error mapping) and makes the
transformation unit-testable without a running ASGI application.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import numpy as np

PERCENT_SCALE = 100.0
"""float: Conversion factor from probability to percentage points."""

PERCENT_DECIMALS = 1
"""int: Decimal places retained for displayed percentages."""


class CandidateView(TypedDict):
    """One ranked taxon entry displayed in the probability drawer.

    Attributes:
        taxon_id: Canonical DarwinCore taxon identifier (0 if the catalogue omits it).
        scientific_name: Latin binomial.
        common_name: Vernacular name.
        confidence: Softmax probability in [0, 1].
        confidence_pct: Probability expressed in percent, rounded for display.
        in_conformal_set: Whether the taxon is a member of the conformal prediction set C(x).
    """

    taxon_id: int
    scientific_name: str
    common_name: str
    confidence: float
    confidence_pct: float
    in_conformal_set: bool


def build_top_candidates(
    probabilities: Sequence[float],
    catalog: Sequence[Mapping[str, Any]],
    conformal_set: Sequence[str],
    top_k: int,
) -> list[CandidateView]:
    """Rank taxa by probability and annotate each with conformal-set membership.

    Why:
        Operators judge a prediction not only by its top-1 label but by how probability mass is spread
        across plausible siblings. Marking members of the conformal set C(x) shows which alternatives the
        statistical guarantee still considers admissible.

    How:
        Sorts class indices by descending probability (stable, so ties keep catalogue order), keeps the
        first ``top_k`` and joins each index with its catalogue metadata.

    Complexity:
        O(K log K) for the sort over K taxa.

    Args:
        probabilities: Softmax probabilities, one per catalogue entry, in catalogue order.
        catalog: Taxa catalogue entries providing ``scientific_name`` and ``common_name`` and
            optionally ``id``.
        conformal_set: Scientific names contained in the conformal prediction set.
        top_k: Maximum number of candidates to return.

    Returns:
        Up to ``top_k`` candidate views ordered by descending probability.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    ranked = np.argsort(-probs, kind="stable")[:top_k]
    members = set(conformal_set)
    candidates: list[CandidateView] = []
    for index in ranked:
        entry = catalog[int(index)]
        confidence = float(probs[index])
        candidates.append(
            CandidateView(
                taxon_id=int(entry.get("id", 0)),
                scientific_name=str(entry["scientific_name"]),
                common_name=str(entry["common_name"]),
                confidence=confidence,
                confidence_pct=round(confidence * PERCENT_SCALE, PERCENT_DECIMALS),
                in_conformal_set=str(entry["scientific_name"]) in members,
            )
        )
    return candidates
