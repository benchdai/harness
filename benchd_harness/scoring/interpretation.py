"""
Interpretation computation for Bench'd structured scores.

This module computes interpretation labels at scoring time so they are
persisted in the signed manifest — not computed on-the-fly in the UI.

Mirrors the logic from benchd/lib/scoring-schema.ts computeInterpretation(),
but runs in the harness pipeline so the result is immutable and verifiable.
"""

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────
# Types (mirror TypeScript scoring-schema.ts)
# ─────────────────────────────────────────────────────────

INTERPRETATION_LABELS = (
    "excellent",           # >= 90%
    "strong",             # >= 70%
    "average",            # >= 40%
    "weak",              # > 0% but < 40%, on a claimed capability
    "capability_limited", # low score driven by unsupported sub-dimensions
    "not_applicable",    # metric doesn't apply
)

SCORE_STATUSES = (
    "measured",
    "not_claimed",
    "not_applicable",
    "not_supported",
    "pending",
    "adapter_missing",
    "isolation_failed",
    "runtime_failed",
)

PURPOSE_ALIGNMENT_VERSION = "1.0"

# Purpose alignment table — which metrics matter for which tracks
PURPOSE_ALIGNMENT: dict[str, dict[str, str]] = {
    "conversational": {
        "Knowledge Retrieval": "adjacent",
        "Knowledge Scale": "orthogonal",
        "LongMemEval": "core",
        "LoCoMo": "core",
        "Truth Arbitration": "core",
        "Memory Poisoning": "core",
        "Budget Curves": "adjacent",
        "Reliability": "core",
    },
    "knowledge-brain": {
        "Knowledge Retrieval": "core",
        "Knowledge Scale": "core",
        "LongMemEval": "orthogonal",
        "LoCoMo": "orthogonal",
        "Truth Arbitration": "core",
        "Memory Poisoning": "adjacent",
        "Budget Curves": "core",
        "Reliability": "adjacent",
    },
    "graph": {
        "Knowledge Retrieval": "core",
        "Knowledge Scale": "core",
        "LongMemEval": "orthogonal",
        "LoCoMo": "orthogonal",
        "Truth Arbitration": "core",
        "Memory Poisoning": "adjacent",
        "Budget Curves": "core",
        "Reliability": "adjacent",
    },
    "agent-memory": {
        "Knowledge Retrieval": "core",
        "Knowledge Scale": "adjacent",
        "LongMemEval": "adjacent",
        "LoCoMo": "orthogonal",
        "Truth Arbitration": "core",
        "Memory Poisoning": "core",
        "Budget Curves": "core",
        "Reliability": "core",
    },
    "baseline": {
        "Knowledge Retrieval": "core",
        "Knowledge Scale": "core",
        "LongMemEval": "core",
        "LoCoMo": "core",
        "Truth Arbitration": "core",
        "Memory Poisoning": "core",
        "Budget Curves": "core",
        "Reliability": "core",
    },
}


def compute_interpretation(
    raw_value: Optional[float],
    capability_claim: str,
    sub_dimensions: list[dict],
) -> dict:
    """
    Compute the interpretation label and summary for a score.

    This is the canonical implementation — called at scoring time and
    persisted in the manifest. The UI reads this field; it does not recompute.

    Args:
        raw_value: Score as percentage (0-100), or None if not yet tested.
        capability_claim: One of "claimed", "claimed_partial", "not_claimed", "unknown".
        sub_dimensions: List of sub-dimension dicts with keys:
            id, label, raw_value, max_value, status, explanation (optional).

    Returns:
        Dict with "label" and "summary" keys.
    """
    if raw_value is None:
        return {"label": "not_applicable", "summary": "Not yet tested."}

    # Check if low score is driven by unsupported sub-dimensions
    unsupported_dims = [
        d for d in sub_dimensions
        if d.get("status") in ("not_supported", "not_claimed")
    ]
    measured_dims = [d for d in sub_dimensions if d.get("status") == "measured"]

    if measured_dims:
        measured_avg = sum(
            (d.get("raw_value") or 0) for d in measured_dims
        ) / len(measured_dims)
    else:
        measured_avg = 0.0

    if raw_value < 40 and unsupported_dims and measured_avg > 40:
        unsupported_names = ", ".join(d.get("label", d.get("id", "?")) for d in unsupported_dims)
        return {
            "label": "capability_limited",
            "summary": (
                f"Low overall score driven by unsupported capabilities "
                f"({unsupported_names}). Measured dimensions average {round(measured_avg)}%."
            ),
        }

    if raw_value >= 90:
        return {"label": "excellent", "summary": "Exceptional performance across all tested dimensions."}
    if raw_value >= 70:
        return {"label": "strong", "summary": "Strong performance on most dimensions."}
    if raw_value >= 40:
        return {"label": "average", "summary": "Moderate performance with room for improvement."}
    if raw_value > 0:
        return {"label": "weak", "summary": "Below average. Review failure traces for specific issues."}
    return {"label": "weak", "summary": "System did not pass any test items in this benchmark."}


def compute_score_status(
    scored_questions: int,
    total_questions: int,
    adapter_error: bool = False,
    isolation_passed: bool = True,
) -> str:
    """Determine the ScoreStatus for a benchmark run."""
    if adapter_error:
        return "runtime_failed"
    if not isolation_passed:
        return "isolation_failed"
    if scored_questions == 0:
        return "pending"
    return "measured"


def get_purpose_alignment(track_id: str, metric_name: str) -> str:
    """Look up purpose alignment for a track/metric pair."""
    track = PURPOSE_ALIGNMENT.get(track_id, {})
    return track.get(metric_name, "not_applicable")


def build_structured_score(
    *,
    metric_id: str,
    metric_version: str,
    track_id: str,
    raw_value: Optional[float],
    status: str,
    capability_claim: str,
    sub_dimensions: list[dict],
    run_id: str,
    harness_version: str,
    judge_model: str,
    judge_temperature: float,
    adapter_version: str,
    runtime_class: str,
    scored_at: str,
    sample_size: int,
    questions_total: int,
    methodology_url: str,
    adapter_author: str = "benchd",
    container_image_hash: Optional[str] = None,
) -> dict:
    """
    Build a complete StructuredScore dict suitable for manifest embedding.

    This is the Python-side equivalent of the TypeScript StructuredScore interface.
    All fields are computed at scoring time and become part of the signed manifest.
    """
    interpretation = compute_interpretation(raw_value, capability_claim, sub_dimensions)
    purpose_alignment = get_purpose_alignment(track_id, metric_id)
    completion_rate = sample_size / questions_total if questions_total > 0 else 0.0

    structured = {
        # Identity
        "metric_id": metric_id,
        "metric_version": metric_version,
        "track_id": track_id,

        # Values
        "raw_value": raw_value,
        "status": status,
        "capability_claim": capability_claim,

        # Context
        "expectation_profile": {
            "track_id": track_id,
            "purpose_alignment": purpose_alignment,
            "track_mean": None,   # populated by baseline computation (Item 2)
            "track_p25": None,
            "track_p75": None,
            "sample_size": 0,
        },
        "sub_dimensions": sub_dimensions,
        "interpretation": interpretation,

        # Purpose alignment versioning
        "purpose_alignment_version": PURPOSE_ALIGNMENT_VERSION,

        # Run context
        "run_context": {
            "run_id": run_id,
            "harness_version": harness_version,
            "judge_model": judge_model,
            "judge_temperature": judge_temperature,
            "adapter_version": adapter_version,
            "runtime_class": runtime_class,
            "container_image_hash": container_image_hash,
            "scored_at": scored_at,
        },

        # Freshness (current at scoring time, may become stale later)
        "freshness": "current",

        # Confidence
        "confidence_metadata": {
            "sample_size": sample_size,
            "questions_total": questions_total,
            "completion_rate": round(completion_rate, 4),
        },

        # Adapter provenance
        "adapter_author": adapter_author,

        # References
        "methodology_url": methodology_url,
    }

    return structured
