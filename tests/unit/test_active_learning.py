# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Unit tests for active learning query strategies and triage review queue."""

from __future__ import annotations

import numpy as np

from taxon_vision.active_learning.query import margin_sampling
from taxon_vision.active_learning.triage_queue import TriageItem, TriageQueue


def test_margin_sampling() -> None:
    """Verify margin sampling correctly ranks ambiguous probability distributions higher."""
    probs = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    scores = margin_sampling(probs)
    # The second row is more ambiguous, so uncertainty score should be higher
    assert scores[1] > scores[0]


def test_triage_queue() -> None:
    """Verify priority queue orders items by uncertainty descending and handles popping."""
    queue = TriageQueue()
    queue.push(TriageItem(observation_id="obs_1", uncertainty_score=0.4, conformal_set_size=1, image_url="url1"))
    queue.push(TriageItem(observation_id="obs_2", uncertainty_score=0.9, conformal_set_size=3, image_url="url2"))
    assert len(queue) == 2
    assert len(queue.queue) == 2

    # Highest uncertainty first
    item = queue.pop()
    assert item is not None
    assert item.observation_id == "obs_2"

    item2 = queue.pop()
    assert item2 is not None
    assert item2.observation_id == "obs_1"

    assert queue.pop() is None
    assert len(queue) == 0
