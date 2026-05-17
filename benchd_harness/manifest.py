"""Manifest schema builder for Bench'd harness run results."""

import json
import statistics
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from benchd_harness.scoring.interpretation import (
    build_structured_score,
    compute_interpretation,
    compute_score_status,
)


@dataclass
class TraceRecord:
    """Per-question trace for failure analysis."""

    trace_id: str
    question_id: str
    dimension: str
    ingest_history: List[Dict[str, Any]]
    query: str
    raw_recall: str  # what the memory system returned directly
    generated_answer: str  # what the answerer LLM produced (empty if no judge)
    response: str  # the final response used for scoring
    expected_answer: str
    scored_correct: bool
    scoring_method: str
    score: float
    max_score: float
    judge_reasoning: Optional[str]
    status: str  # "scored" | "pending_llm_judge"
    latency_ms: float = 0.0  # milliseconds for ingest + recall
    recall_tokens: int = 0  # approximate tokens in raw_recall response
    ingest_tokens: int = 0  # approximate tokens in ingest history
    # Retrieval quality metrics
    retrieval_hit: bool = False        # R@K — expected answer found in raw recall
    partial_hit: bool = False          # At least 50% of answer words found
    word_overlap: float = 0.0          # Jaccard similarity
    answer_density: float = 0.0        # Fraction of expected words found in recall
    compression_ratio: float = 0.0     # expected_len / recall_len
    # Faithfulness metrics
    grounded: bool = False             # Is answer supported by retrieved evidence?
    hallucination_risk: float = 0.0    # 0.0 = fully grounded, 1.0 = fully hallucinated
    abstained: bool = False            # Did system say "I don't know"?


def estimate_tokens(text: str) -> int:
    """Approximate token count: ~1.3 tokens per word."""
    return len(text.split()) * 4 // 3


def _compute_efficiency(
    traces: List[TraceRecord],
    passed: int,
) -> Dict[str, Any]:
    """Compute aggregate efficiency metrics from trace records."""
    latencies = [t.latency_ms for t in traces]
    recall_tokens_list = [t.recall_tokens for t in traces]
    ingest_tokens_list = [t.ingest_tokens for t in traces]

    total_recall_tokens = sum(recall_tokens_list)

    if not latencies:
        return {
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "mean_recall_tokens": 0.0,
            "total_recall_tokens": 0,
            "mean_ingest_tokens": 0.0,
            "tokens_per_correct_answer": None,
        }

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted_latencies[p95_idx], 2),
        "mean_recall_tokens": round(statistics.mean(recall_tokens_list), 2),
        "total_recall_tokens": total_recall_tokens,
        "mean_ingest_tokens": round(statistics.mean(ingest_tokens_list), 2),
        "tokens_per_correct_answer": (
            round(total_recall_tokens / passed, 2) if passed > 0 else None
        ),
    }


def build_manifest(
    run_id: str,
    system_name: str,
    adapter_name: str,
    adapter_version: Optional[str],
    benchmark_slug: str,
    benchmark_name: str,
    benchmark_version: str,
    harness_version: str,
    started_at: datetime,
    completed_at: datetime,
    traces: List[TraceRecord],
    judge_metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Build the canonical manifest dict from run results.

    Computes:
    - Per-dimension verified scores (only exact/regex scored items)
    - Per-dimension nuance scores (only llm items -- null if all pending)
    - Overall verified score (average of dimensions that have scored items)
    - Overall nuance score (null if no llm items scored)
    - Total question count, pass count, fail count
    """
    # Group traces by dimension
    dimensions = ["recall", "temporal", "reasoning"]
    traces_by_dim: Dict[str, List[TraceRecord]] = {d: [] for d in dimensions}
    for t in traces:
        if t.dimension in traces_by_dim:
            traces_by_dim[t.dimension].append(t)

    # Compute verified scores (exact/regex only, status="scored")
    verified: Dict[str, Optional[float]] = {}
    for dim in dimensions:
        dim_traces = traces_by_dim[dim]
        scored = [
            t for t in dim_traces
            if t.scoring_method in ("exact", "regex") and t.status == "scored"
        ]
        if not scored:
            verified[dim] = None
        else:
            total_max = sum(t.max_score for t in scored)
            total_earned = sum(t.score for t in scored)
            verified[dim] = (total_earned / total_max * 100.0) if total_max > 0 else 0.0

    # Overall verified: average of dimensions that have scored items
    scored_dims = [v for v in verified.values() if v is not None]
    verified_overall = (
        sum(scored_dims) / len(scored_dims) if scored_dims else None
    )

    # Compute nuance scores (llm items only)
    nuance: Dict[str, Optional[float]] = {}
    for dim in dimensions:
        dim_traces = traces_by_dim[dim]
        llm_traces = [t for t in dim_traces if t.scoring_method == "llm"]
        llm_scored = [t for t in llm_traces if t.status == "scored"]
        if not llm_scored:
            nuance[dim] = None
        else:
            total_max = sum(t.max_score for t in llm_scored)
            total_earned = sum(t.score for t in llm_scored)
            nuance[dim] = (total_earned / total_max * 100.0) if total_max > 0 else 0.0

    nuance_scored_dims = [v for v in nuance.values() if v is not None]
    nuance_overall = (
        sum(nuance_scored_dims) / len(nuance_scored_dims)
        if nuance_scored_dims
        else None
    )

    # Summary counts
    total_questions = len(traces)
    scored_questions = sum(1 for t in traces if t.status == "scored")
    pending_questions = total_questions - scored_questions
    passed = sum(1 for t in traces if t.scored_correct)
    failed = scored_questions - passed

    # Efficiency metrics
    efficiency = _compute_efficiency(traces, passed)

    # Retrieval & quality metrics
    retrieval_hits = sum(1 for t in traces if t.retrieval_hit)
    partial_hits = sum(1 for t in traces if t.partial_hit)
    retrieval_total = len(traces)
    retrieval_recall_at_k = (
        (retrieval_hits / retrieval_total * 100.0) if retrieval_total > 0 else 0.0
    )
    partial_recall = (
        (partial_hits / retrieval_total * 100.0) if retrieval_total > 0 else 0.0
    )
    mean_word_overlap = (
        sum(t.word_overlap for t in traces) / retrieval_total if retrieval_total > 0 else 0.0
    )
    mean_answer_density = (
        sum(t.answer_density for t in traces) / retrieval_total if retrieval_total > 0 else 0.0
    )
    mean_compression = (
        sum(t.compression_ratio for t in traces) / retrieval_total if retrieval_total > 0 else 0.0
    )

    # Faithfulness metrics
    grounded_count = sum(1 for t in traces if t.grounded)
    abstained_count = sum(1 for t in traces if t.abstained)
    mean_hallucination_risk = (
        sum(t.hallucination_risk for t in traces) / retrieval_total if retrieval_total > 0 else 0.0
    )

    # Compute interpretation at scoring time (persisted, not re-computed in UI)
    overall_score = verified_overall if verified_overall is not None else nuance_overall
    score_status = compute_score_status(
        scored_questions=scored_questions,
        total_questions=total_questions,
    )
    interpretation = compute_interpretation(
        raw_value=overall_score,
        capability_claim="claimed",  # if we're scoring it, capability is claimed
        sub_dimensions=[],  # top-level; sub-dimensions handled per-metric
    )

    # Build structured score metadata for the manifest
    structured_score = build_structured_score(
        metric_id=benchmark_slug,
        metric_version=benchmark_version,
        track_id="baseline",  # default; overridden by caller if track is known
        raw_value=overall_score,
        status=score_status,
        capability_claim="claimed",
        sub_dimensions=[],
        run_id=run_id,
        harness_version=harness_version,
        judge_model=judge_metadata.get("judge_model", "deterministic") if judge_metadata else "deterministic",
        judge_temperature=judge_metadata.get("temperature", 0.0) if judge_metadata else 0.0,
        adapter_version=adapter_version or "unknown",
        runtime_class="S1",  # default tier
        scored_at=completed_at.isoformat(),
        sample_size=scored_questions,
        questions_total=total_questions,
        methodology_url=f"https://benchd.ai/methodology/metrics/{benchmark_slug}",
    )

    return {
        "version": "1.1.0",
        "run_id": run_id,
        "system": {
            "name": system_name,
            "adapter": adapter_name,
            "version": adapter_version,
        },
        "benchmark": {
            "slug": benchmark_slug,
            "name": benchmark_name,
            "version": benchmark_version,
        },
        "harness": {
            "version": harness_version,
        },
        "judge": judge_metadata,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "scores": {
            "verified": {
                "recall": verified.get("recall"),
                "temporal": verified.get("temporal"),
                "reasoning": verified.get("reasoning"),
                "overall": verified_overall,
            },
            "nuance": {
                "recall": nuance.get("recall"),
                "temporal": nuance.get("temporal"),
                "reasoning": nuance.get("reasoning"),
                "overall": nuance_overall,
            },
        },
        "summary": {
            "total_questions": total_questions,
            "scored_questions": scored_questions,
            "pending_questions": pending_questions,
            "passed": passed,
            "failed": failed,
        },
        "efficiency": efficiency,
        "retrieval": {
            "recall_at_k": round(retrieval_recall_at_k, 2),
            "partial_recall": round(partial_recall, 2),
            "mean_word_overlap": round(mean_word_overlap, 4),
            "mean_answer_density": round(mean_answer_density, 4),
            "mean_compression_ratio": round(mean_compression, 4),
            "k": 5,
            "total_hits": retrieval_hits,
            "partial_hits": partial_hits,
            "total_questions": retrieval_total,
        },
        "faithfulness": {
            "grounded_pct": round(grounded_count / retrieval_total * 100, 2) if retrieval_total > 0 else 0.0,
            "abstention_pct": round(abstained_count / retrieval_total * 100, 2) if retrieval_total > 0 else 0.0,
            "mean_hallucination_risk": round(mean_hallucination_risk, 4),
            "grounded_count": grounded_count,
            "abstained_count": abstained_count,
        },
        "traces": [asdict(t) for t in traces],
        # Structured score metadata (persisted at scoring time)
        "structured_score": structured_score,
        "interpretation": interpretation,
    }


def generate_run_id() -> str:
    """Generate a unique run ID: run_ + 12 char hex."""
    return f"run_{uuid.uuid4().hex[:12]}"
