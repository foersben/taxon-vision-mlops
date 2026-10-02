# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Human-in-the-Loop review queue prioritizing ambiguous observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriageItem:
    observation_id: str
    uncertainty_score: float
    conformal_set_size: int
    image_url: str


class TriageQueue:
    """Priority review queue."""

    def __init__(self) -> None:
        self.queue: list[TriageItem] = []

    def push(self, item: TriageItem) -> None:
        self.queue.append(item)
        self.queue.sort(key=lambda x: x.uncertainty_score, reverse=True)

    def pop(self) -> TriageItem | None:
        return self.queue.pop(0) if self.queue else None
