"""Test feedback loop.py.

This module provides functionality related to test_feedback_loop.
"""

from taxon_vision.feedback.reconciliation import reconcile_taxonomic_mutation


def test_taxonomic_reconciliation() -> None:
    """Test taxonomic reconciliation."""
    # Species 1 was split into species 5
    mapping = {1: 5}
    assert reconcile_taxonomic_mutation(1, 1, mapping) == 5
    assert reconcile_taxonomic_mutation(2, 2, mapping) == 2
