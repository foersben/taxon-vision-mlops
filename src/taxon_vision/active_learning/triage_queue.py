# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Human-in-the-Loop review queue prioritizing ambiguous observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriageItem:
    """Triage item.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    observation_id: str
    uncertainty_score: float
    conformal_set_size: int
    image_url: str


class TriageQueue:
    """Priority review queue.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    def __init__(self) -> None:
        """Init  ."""
        self.queue: list[TriageItem] = []

    def push(self, item: TriageItem) -> None:
        """Push.

        Args:
            item: The item parameter.
        """
        self.queue.append(item)
        self.queue.sort(key=lambda x: x.uncertainty_score, reverse=True)

    def pop(self) -> TriageItem | None:
        """Pop.

        Returns:
            The resulting value from the operation.
        """
        return self.queue.pop(0) if self.queue else None
