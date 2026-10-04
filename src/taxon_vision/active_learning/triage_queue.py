# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Human-in-the-Loop review queue prioritizing ambiguous observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriageItem:
    """Observation item prioritized for expert taxonomic review."""

    observation_id: str
    uncertainty_score: float
    conformal_set_size: int
    image_url: str


class TriageQueue:
    """Priority review queue sorting observations by uncertainty score."""

    def __init__(self) -> None:
        """Initialize an empty priority triage queue."""
        self.queue: list[TriageItem] = []

    def push(self, item: TriageItem) -> None:
        """Insert an observation item and re-sort by uncertainty in descending order.

        Args:
            item: Observation review item to enqueue.
        """
        self.queue.append(item)
        self.queue.sort(key=lambda x: x.uncertainty_score, reverse=True)

    def pop(self) -> TriageItem | None:
        """Retrieve and remove the most uncertain observation from the queue.

        Returns:
            The highest-uncertainty observation item, or None if the queue is empty.
        """
        return self.queue.pop(0) if self.queue else None
