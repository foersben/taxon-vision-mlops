import numpy as np

from taxon_vision.feedback.collector import FeedbackSubmission
from taxon_vision.monitoring.drift import compute_wasserstein_drift


def test_compute_wasserstein_drift():
    ref = np.random.randn(100, 16)
    curr = np.random.randn(100, 16) + 2.0
    drift = compute_wasserstein_drift(ref, curr)
    assert drift > 0.0


def test_feedback_submission():
    sub = FeedbackSubmission(observation_id="123", validated_taxon_id=2, reviewer_id="user_1")
    assert sub.validated_taxon_id == 2
