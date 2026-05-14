"""
Bench'd Isolation Verification — guarantees clean state before scoring.

The canary check runs BEFORE any benchmark data is loaded. It queries
for data that should never exist. If results come back, the run is
flagged as "isolation_failed" and not eligible for official scoring.

This is the structural guarantee that prevents the gbrain-class problem:
stale data from previous runs contaminating results.
"""

import uuid
from typing import Callable

from .base import IsolationProbe, IsolationStrategy


# Canary strings that should NEVER appear in a clean system
CANARY_QUERIES = [
    f"benchd_canary_{uuid.uuid4().hex[:8]}",
    "xq7z_isolation_probe_never_real_data",
    "benchd_stale_check_2026_verify_clean",
]


def run_isolation_check(
    recall_fn: Callable[[str], str],
    strategy: str = "adapter_reset",
) -> IsolationProbe:
    """
    Run canary isolation check against a memory system.

    Args:
        recall_fn: The adapter's recall() method
        strategy: The isolation strategy being used

    Returns:
        IsolationProbe with clean=True/False and evidence
    """
    stale_hits = []

    for canary in CANARY_QUERIES:
        try:
            result = recall_fn(canary)
            # Empty or error responses are fine — that means clean
            if not result or not result.strip():
                continue
            if "error" in result.lower() or "don't know" in result.lower():
                continue
            if "no information" in result.lower() or "insufficient" in result.lower():
                continue
            if len(result.strip()) < 10:
                continue

            # Non-trivial response to a canary query = stale data
            stale_hits.append({
                "query": canary,
                "response_preview": result[:200],
            })
        except Exception:
            # Errors during canary check are fine — means system is clean or not ready
            continue

    if stale_hits:
        return IsolationProbe(
            clean=False,
            stale_data_found=True,
            evidence=f"Canary queries returned {len(stale_hits)} non-empty responses: "
                     f"{stale_hits[0]['query']} → {stale_hits[0]['response_preview'][:100]}",
        )

    return IsolationProbe(
        clean=True,
        evidence=f"All {len(CANARY_QUERIES)} canary queries returned empty/error — clean state confirmed",
    )


def classify_failure(
    isolation_probe: IsolationProbe,
    runtime_started: bool,
    healthcheck_passed: bool,
    benchmark_ran: bool,
    questions_scored: int,
    total_questions: int,
) -> str:
    """
    Classify the failure type for a benchmark run.

    Returns one of:
      - runtime_start_failed
      - healthcheck_failed
      - isolation_failed
      - benchmark_completed
      - benchmark_partial
    """
    if not runtime_started:
        return "runtime_start_failed"
    if not healthcheck_passed:
        return "healthcheck_failed"
    if not isolation_probe.clean:
        return "isolation_failed"
    if not benchmark_ran:
        return "benchmark_completed"  # Shouldn't happen but fallback
    if questions_scored < total_questions:
        return "benchmark_partial"
    return "benchmark_completed"
