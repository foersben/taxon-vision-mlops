# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Human-in-the-Loop review queue prioritizing ambiguous observations."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path


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
        In continuous active learning workflows, human taxonomic verification is a finite, expensive resource. Routing all ambiguous observations naively or using an un-indexed list sort on every insertion incurs O(N log N) overhead per push and O(N) per pop. Employing a binary max-heap (implemented via min-heap with negated priorities) guarantees strict O(log N) worst-case time complexity for enqueuing and dequeuing, enabling unbounded scaling as field observations stream into the staging tier.

    How:
        Stores tuples of `(-uncertainty_score, sequence_counter, item)` inside a binary heap. The negated score converts Python's default min-heap into a max-heap prioritizing highest uncertainty. The monotonic integer counter breaks ties deterministically between observations with identical uncertainty scores without requiring comparative operators on `TriageItem`.

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


@dataclass(frozen=True)
class _PendingTriageItem:
    """Unsynchronised local capture candidate ranked by margin uncertainty.

    Why:
        Pairs the priority scalar with both filesystem paths so items popped from `TriageQueue` resolve directly to their `(meta_path, crop_path)` files without directory re-scans.

    How:
        Constructed by `_collect_pending_triage_items` for every verified unsynchronised capture pair found on the PVC.

    Attributes:
        score: Margin uncertainty scalar (`1.0 - (p1 - p2)`).
        meta_path: Filesystem path to `{obs_id}_meta.json`.
        crop_path: Filesystem path to `{obs_id}_crop.jpg`.
    """

    score: float
    meta_path: Path
    crop_path: Path


def _collect_pending_triage_items(target_dir: Path) -> list[_PendingTriageItem]:
    """Scan local flagged captures and order unsynchronised open-licensed crop pairs via TriageQueue.

    Why:
        When backlogged observations accumulate offline on `/var/lib/taxon-vision/flagged_observations`, uploading them in strict priority order guarantees that the highest-uncertainty specimens enter the human review queue first under finite expert annotation bandwidth.

    How:
        Globs `*_meta.json`, skips malformed or already synchronised records, verifies crop existence, pushes `TriageItem` entries into a max-heap `TriageQueue` ordered by uncertainty score, and drains items in descending priority order.

    Args:
        target_dir: Local directory holding `{obs_id}_meta.json` and `{obs_id}_crop.jpg` pairs.

    Returns:
        List of `_PendingTriageItem` entries sorted from highest to lowest margin uncertainty.
    """
    queue = TriageQueue()
    by_id: dict[str, _PendingTriageItem] = {}
    for meta_path in target_dir.glob("*_meta.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("synced_to_label_studio"):
            continue
        obs_id = str(data.get("observation_id") or meta_path.stem)
        crop_path = target_dir / str(data.get("image_path") or f"{obs_id}_crop.jpg")
        if crop_path.is_file():
            score = float(data.get("uncertainty_score", 0.0))
            set_size = len(data.get("conformal_set", []))
            by_id[obs_id] = _PendingTriageItem(score, meta_path, crop_path)
            queue.push(TriageItem(obs_id, score, set_size, str(crop_path)))

    ordered: list[_PendingTriageItem] = []
    while (item := queue.pop()) is not None:
        if item.observation_id in by_id:
            ordered.append(by_id[item.observation_id])
    return ordered
