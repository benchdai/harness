"""Scoring module for the Bench'd harness."""

from .deterministic import ScoreResult, score_response
from .interpretation import (
    compute_interpretation,
    compute_score_status,
    build_structured_score,
    get_purpose_alignment,
    PURPOSE_ALIGNMENT,
    PURPOSE_ALIGNMENT_VERSION,
)

__all__ = [
    "ScoreResult",
    "score_response",
    "compute_interpretation",
    "compute_score_status",
    "build_structured_score",
    "get_purpose_alignment",
    "PURPOSE_ALIGNMENT",
    "PURPOSE_ALIGNMENT_VERSION",
]
