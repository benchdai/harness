"""Deterministic scoring functions for benchmark responses."""

from dataclasses import dataclass
from typing import Optional, List
import re


@dataclass
class ScoreResult:
    """Result of scoring a single response."""

    scored_correct: bool
    score: float  # 0.0 or 1.0 for now
    max_score: float
    scoring_method: str  # "exact" | "regex" | "llm"
    judge_reasoning: Optional[str] = None
    status: str = "scored"  # "scored" | "pending_llm_judge"


def _normalize(s) -> str:
    """Strip whitespace and lowercase a string. Coerces non-strings."""
    return str(s).strip().lower()


def score_response(
    response: str,
    expected_answer: str,
    scoring_method: str,
    max_score: float = 1.0,
) -> ScoreResult:
    """
    Score a memory system's response against the expected answer.

    - exact: normalized lowercase/trimmed comparison. Also checks if expected
      is contained in response.
    - regex: compile expected_answer as regex, test against response
    - llm: return pending status (no LLM judge yet)
    """
    response = str(response)
    expected_answer = str(expected_answer)

    if scoring_method == "exact":
        return _score_exact(response, expected_answer, max_score)
    elif scoring_method == "regex":
        return _score_regex(response, expected_answer, max_score)
    elif scoring_method == "reliability_trap":
        return _score_reliability_trap(response, expected_answer, max_score)
    elif scoring_method == "llm":
        return _score_llm_pending(max_score)
    else:
        raise ValueError(f"Unknown scoring_method: {scoring_method!r}")


def _score_exact(response: str, expected_answer: str, max_score: float) -> ScoreResult:
    """Exact match scoring with containment fallback."""
    norm_response = _normalize(response)
    norm_expected = _normalize(expected_answer)

    # Exact equality after normalization
    is_correct = norm_response == norm_expected

    # Containment: expected answer appears within the response
    if not is_correct:
        is_correct = norm_expected in norm_response

    return ScoreResult(
        scored_correct=is_correct,
        score=max_score if is_correct else 0.0,
        max_score=max_score,
        scoring_method="exact",
        judge_reasoning=None,
        status="scored",
    )


def _score_regex(response: str, expected_answer: str, max_score: float) -> ScoreResult:
    """Regex match scoring (case-insensitive search)."""
    pattern = re.compile(expected_answer, re.IGNORECASE)
    match = pattern.search(response)
    is_correct = match is not None

    return ScoreResult(
        scored_correct=is_correct,
        score=max_score if is_correct else 0.0,
        max_score=max_score,
        scoring_method="regex",
        judge_reasoning=None,
        status="scored",
    )


def _score_reliability_trap(response: str, expected_answer: str, max_score: float) -> ScoreResult:
    """
    Score a reliability trap question.

    expected_answer contains pipe-separated keywords that indicate correct behavior.
    The response PASSES if it contains any expected keyword (system behaved correctly).
    """
    norm_response = _normalize(response)
    expected_keywords = [_normalize(kw) for kw in expected_answer.split("|") if kw.strip()]

    # Check if any expected keyword is present (system did the right thing)
    is_correct = any(kw in norm_response for kw in expected_keywords)

    # Generate reasoning
    if is_correct:
        matched = [kw for kw in expected_keywords if kw in norm_response]
        reasoning = f"PASS: Response contains expected indicator: '{matched[0]}'"
    else:
        reasoning = f"FAIL: Response did not contain any expected keywords: {expected_keywords}"

    return ScoreResult(
        scored_correct=is_correct,
        score=max_score if is_correct else 0.0,
        max_score=max_score,
        scoring_method="reliability_trap",
        judge_reasoning=reasoning,
        status="scored",
    )


def _score_llm_pending(max_score: float) -> ScoreResult:
    """Return a pending result for items that need LLM judging."""
    return ScoreResult(
        scored_correct=False,
        score=0.0,
        max_score=max_score,
        scoring_method="llm",
        judge_reasoning=None,
        status="pending_llm_judge",
    )


def score_retrieval_recall(
    raw_recall: str,
    expected_answer: str,
    k: int = 5,
) -> bool:
    """
    Check if the expected answer appears anywhere in the raw retrieved results.
    This is R@K — retrieval recall at K.

    Simple containment check: does the normalized expected answer appear
    in the normalized raw recall text?

    This is separate from QA scoring — it measures retrieval quality only.
    """
    norm_recall = str(raw_recall).strip().lower()
    norm_expected = str(expected_answer).strip().lower()

    if not norm_expected or not norm_recall:
        return False

    return norm_expected in norm_recall


@dataclass
class RetrievalMetrics:
    """Full suite of retrieval quality metrics for a single question."""
    retrieval_hit: bool = False        # R@K — expected answer found in raw recall
    partial_hit: bool = False          # At least 50% of expected answer words found
    word_overlap: float = 0.0          # Jaccard similarity between expected and recalled words
    answer_density: float = 0.0        # Fraction of recall tokens that are answer-relevant
    recall_length: int = 0             # Length of raw recall in characters
    expected_length: int = 0           # Length of expected answer in characters
    compression_ratio: float = 0.0     # expected_length / recall_length (higher = more efficient)


def compute_retrieval_metrics(
    raw_recall: str,
    expected_answer: str,
) -> RetrievalMetrics:
    """
    Compute a full suite of retrieval quality metrics.

    These metrics measure how well the memory system retrieved relevant
    evidence, independent of whether the LLM answerer got the final answer right.
    """
    norm_recall = str(raw_recall).strip().lower()
    norm_expected = str(expected_answer).strip().lower()

    if not norm_expected or not norm_recall:
        return RetrievalMetrics(
            recall_length=len(norm_recall),
            expected_length=len(norm_expected),
        )

    # R@K — exact containment
    retrieval_hit = norm_expected in norm_recall

    # Word-level analysis
    recall_words = set(norm_recall.split())
    expected_words = set(norm_expected.split())

    if not expected_words:
        return RetrievalMetrics(
            retrieval_hit=retrieval_hit,
            recall_length=len(norm_recall),
            expected_length=len(norm_expected),
        )

    # Overlap metrics
    intersection = recall_words & expected_words
    union = recall_words | expected_words

    word_overlap = len(intersection) / len(union) if union else 0.0
    partial_hit = len(intersection) / len(expected_words) >= 0.5

    # Density — what fraction of expected answer words appear in recall
    answer_density = len(intersection) / len(expected_words) if expected_words else 0.0

    # Compression ratio — higher means the system returned a concise result
    compression_ratio = (
        len(norm_expected) / len(norm_recall) if len(norm_recall) > 0 else 0.0
    )

    return RetrievalMetrics(
        retrieval_hit=retrieval_hit,
        partial_hit=partial_hit,
        word_overlap=round(word_overlap, 4),
        answer_density=round(answer_density, 4),
        recall_length=len(norm_recall),
        expected_length=len(norm_expected),
        compression_ratio=round(compression_ratio, 4),
    )


def compute_faithfulness(
    raw_recall: str,
    generated_answer: str,
    expected_answer: str,
) -> dict:
    """
    Compute faithfulness metrics — did the answer come from memory or hallucination?

    Returns:
        grounded: bool — is the generated answer supported by the raw recall?
        hallucination_risk: float — 0.0 (fully grounded) to 1.0 (fully hallucinated)
        abstained: bool — did the system say "I don't know" or equivalent?
    """
    norm_recall = str(raw_recall).strip().lower()
    norm_answer = str(generated_answer).strip().lower()
    norm_expected = str(expected_answer).strip().lower()

    # Check if the system abstained
    abstention_phrases = [
        "insufficient information",
        "not enough information",
        "i don't know",
        "i don't have",
        "no information",
        "cannot determine",
        "not found",
        "information not found",
        "not available",
    ]
    abstained = any(phrase in norm_answer for phrase in abstention_phrases)

    if not norm_answer or abstained:
        return {
            "grounded": False,
            "hallucination_risk": 0.0,  # abstention isn't hallucination
            "abstained": True,
        }

    # Check if answer content is supported by recall
    answer_words = set(norm_answer.split()) - {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "to", "for", "of", "and", "or"}
    recall_words = set(norm_recall.split())

    if not answer_words:
        return {"grounded": True, "hallucination_risk": 0.0, "abstained": False}

    supported = answer_words & recall_words
    grounded_ratio = len(supported) / len(answer_words)

    return {
        "grounded": grounded_ratio >= 0.5,
        "hallucination_risk": round(1.0 - grounded_ratio, 4),
        "abstained": False,
    }


def aggregate_scores(
    results: List[ScoreResult], dimension: Optional[str] = None
) -> dict:
    """
    Aggregate a list of ScoreResults into summary stats.

    Only items with status="scored" are counted toward the score calculation.
    Pending LLM-judge items are tracked separately.

    Args:
        results: List of ScoreResult objects to aggregate.
        dimension: Optional label (unused in calculation, available for callers).

    Returns:
        Dictionary with keys: total, scored, correct, pending, score, max_score.
    """
    total = len(results)
    scored_results = [r for r in results if r.status == "scored"]
    pending = total - len(scored_results)

    correct = sum(1 for r in scored_results if r.scored_correct)
    earned = sum(r.score for r in scored_results)
    max_possible = sum(r.max_score for r in scored_results)

    # Score as a percentage (0-100); avoid division by zero
    pct = (earned / max_possible * 100.0) if max_possible > 0 else 0.0

    return {
        "total": total,
        "scored": len(scored_results),
        "correct": correct,
        "pending": pending,
        "score": pct,
        "max_score": max_possible,
    }
