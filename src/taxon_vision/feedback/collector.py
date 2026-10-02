# SPDX-FileCopyrightText: 2026 Benjamin Förster
# SPDX-License-Identifier: MIT
"""Feedback ingestion ledger."""

from __future__ import annotations

from pydantic import BaseModel


class FeedbackSubmission(BaseModel):
    """Feedback submission.

    Attributes:
        Various internal state and configuration variables used by the class.
    """

    observation_id: str
    validated_taxon_id: int
    reviewer_id: str
    notes: str = ""
