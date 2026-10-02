"""Test active learning.py.

This module provides functionality related to test_active_learning.
"""

import numpy as np

from taxon_vision.active_learning.query import margin_sampling
from taxon_vision.active_learning.triage_queue import TriageItem, TriageQueue


def test_margin_sampling() -> None:
    """Test margin sampling."""
    probs = np.array([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]])
    scores = margin_sampling(probs)
    # The second row is more ambiguous, so uncertainty score should be higher
    assert scores[1] > scores[0]


def test_triage_queue() -> None:
    """Test triage queue."""
    queue = TriageQueue()
    queue.push(TriageItem(observation_id="obs_1", uncertainty_score=0.4, conformal_set_size=1, image_url="url1"))
    queue.push(TriageItem(observation_id="obs_2", uncertainty_score=0.9, conformal_set_size=3, image_url="url2"))
    # Highest uncertainty first
    item = queue.pop()
    assert item is not None
    assert item.observation_id == "obs_2"
    item2 = queue.pop()
    assert item2.observation_id == "obs_1"
    assert queue.pop() is None
