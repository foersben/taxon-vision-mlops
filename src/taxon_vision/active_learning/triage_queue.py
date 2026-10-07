# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Human-in-the-Loop review queue prioritizing ambiguous observations."""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class TriageItem:
    """Observation item prioritized for expert taxonomic review.

    Attributes:
        observation_id: Unique string identifier of the raw observation.
        uncertainty_score: Scalar epistemic uncertainty measure (e.g. 1 - margin).
        conformal_set_size: Cardinality of the admissible conformal prediction set |C(x)|.
        image_url: Remote or local storage locator for the observation image.
    """

    observation_id: str
    uncertainty_score: float
    conformal_set_size: int
    image_url: str


class TriageQueue:
    """Priority review queue sorting observations by uncertainty score.

    Why:
        In continuous active learning workflows, human taxonomic verification is a finite,
        expensive resource. Routing all ambiguous observations naively or using an un-indexed
        list sort on every insertion incurs O(N log N) overhead per push and O(N) per pop.
        Employing a binary max-heap (implemented via min-heap with negated priorities) guarantees
        strict O(log N) worst-case time complexity for enqueuing and dequeuing, enabling
        unbounded scaling as field observations stream into the staging tier.

    How:
        Stores tuples of `(-uncertainty_score, sequence_counter, item)` inside a binary heap.
        The negated score converts Python's default min-heap into a max-heap prioritizing highest
        uncertainty. The monotonic integer counter breaks ties deterministically between observations
        with identical uncertainty scores without requiring comparative operators on `TriageItem`.

    Complexity:
        - push: O(log N) time complexity.
        - pop: O(log N) time complexity.
        - space: O(N) space complexity.
    """

    def __init__(self) -> None:
        """Initialize an empty priority triage heap and counter."""
        self._heap: list[tuple[float, int, TriageItem]] = []
        self._counter: int = 0

    @property
    def queue(self) -> list[TriageItem]:
        """Expose items ordered by uncertainty score descending for inspection.

        Returns:
            List of TriageItem instances sorted from highest to lowest uncertainty.
        """
        return [item for _, _, item in sorted(self._heap, key=lambda x: x[0])]

    def __len__(self) -> int:
        """Return the number of pending items in the triage queue."""
        return len(self._heap)

    def push(self, item: TriageItem) -> None:
        """Insert an observation item into the priority heap in O(log N) time.

        Args:
            item: Observation review item to enqueue.
        """
        heapq.heappush(self._heap, (-item.uncertainty_score, self._counter, item))
        self._counter += 1

    def pop(self) -> TriageItem | None:
        """Retrieve and remove the most uncertain observation from the queue in O(log N) time.

        Returns:
            The highest-uncertainty observation item, or None if the queue is empty.
        """
        if not self._heap:
            return None
        _, _, item = heapq.heappop(self._heap)
        return item
