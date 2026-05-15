"""
Bench'd Failure Taxonomy v1.0

Standardized reason codes for benchmark failures. Every failed trace
gets a reason_code from this taxonomy, making failures aggregable,
citable, and machine-consumable.

Usage:
  from benchd_harness.failure_taxonomy import classify_failure, FailureCode

  code = classify_failure(
      expected="Denver",
      actual="Austin",
      raw_recall="User moved to Austin in Jan...",
      query="Where does the user live?",
  )
  # Returns FailureCode.STALE_MEMORY
"""

from enum import Enum
from typing import Optional


class FailureCode(str, Enum):
    """Standardized failure reason codes."""

    # Retrieval failures
    RETRIEVAL_MISS = "retrieval_miss"            # Expected info was in memory but not retrieved
    RETRIEVAL_IRRELEVANT = "retrieval_irrelevant"  # Retrieved content unrelated to query
    OVER_RETRIEVAL = "over_retrieval"             # Too much context, answer buried in noise

    # Temporal failures
    STALE_MEMORY = "stale_memory"                # Used outdated/superseded information
    TEMPORAL_CONFUSION = "temporal_confusion"      # Wrong time ordering or date interpretation
    TEMPORAL_MISSING = "temporal_missing"          # No temporal awareness at all

    # Entity/fact failures
    WRONG_ENTITY = "wrong_entity"                # Mixed up similar entities (Sarah vs Sara)
    WRONG_FACT = "wrong_fact"                    # Retrieved correct entity, wrong attribute
    PARTIAL_ANSWER = "partial_answer"            # Correct but incomplete

    # Hallucination failures
    HALLUCINATION = "hallucination"              # Fabricated information not in memory
    UNSUPPORTED_CLAIM = "unsupported_claim"      # Answer not grounded in retrieved evidence

    # Memory management failures
    DELETION_FAILURE = "deletion_failure"        # Failed to forget when explicitly asked
    MISSING_PROVENANCE = "missing_provenance"    # No source/timestamp/ID on memory item
    CROSS_CONTAMINATION = "cross_contamination"  # Data leaked from another session/agent/run

    # Multi-agent failures
    CROSS_AGENT_LEAK = "cross_agent_leak"        # Agent B saw Agent A's private data
    HANDOFF_LOSS = "handoff_loss"                # Memory lost during agent handoff
    CONFLICT_UNRESOLVED = "conflict_unresolved"  # Contradictory facts, no resolution

    # System failures
    EMPTY_RECALL = "empty_recall"                # System returned nothing
    RECALL_ERROR = "recall_error"                # System threw an error
    TIMEOUT = "timeout"                          # Query exceeded time limit
    FORMAT_MISMATCH = "format_mismatch"          # Answer format doesn't match expected

    # Judge failures
    JUDGE_DISAGREEMENT = "judge_disagreement"    # LLM judge and deterministic scorer disagree
    ABSTENTION_WRONG = "abstention_wrong"        # Said "I don't know" when answer was available


def classify_failure(
    expected: str,
    actual: str,
    raw_recall: str = "",
    query: str = "",
    dimension: str = "",
    scoring_method: str = "",
) -> FailureCode:
    """
    Classify a benchmark failure into a standardized reason code.

    Uses heuristics based on the expected answer, actual response,
    raw recall content, and query context.
    """
    expected_lower = expected.lower().strip()
    actual_lower = actual.lower().strip()
    recall_lower = raw_recall.lower().strip()

    # Empty or error responses
    if not actual_lower or actual_lower == "":
        return FailureCode.EMPTY_RECALL
    if "[recall error" in actual_lower or "[error" in actual_lower:
        return FailureCode.RECALL_ERROR
    if "timed out" in actual_lower:
        return FailureCode.TIMEOUT

    # Abstention when answer exists
    abstention_phrases = ["don't know", "no information", "insufficient", "not mentioned", "cannot determine"]
    if any(p in actual_lower for p in abstention_phrases):
        if expected_lower in recall_lower:
            return FailureCode.RETRIEVAL_MISS  # Was in recall but answerer abstained
        return FailureCode.ABSTENTION_WRONG if expected_lower else FailureCode.EMPTY_RECALL

    # Check if expected answer was in the raw recall
    if recall_lower and expected_lower not in recall_lower:
        # Expected answer not in retrieved content
        if len(recall_lower) > 500:
            return FailureCode.RETRIEVAL_IRRELEVANT
        return FailureCode.RETRIEVAL_MISS

    # Temporal dimension failures
    if dimension in ("temporal", "stale_memory", "knowledge_update"):
        return FailureCode.STALE_MEMORY

    # Entity confusion
    if dimension == "entity_confusion":
        return FailureCode.WRONG_ENTITY

    # Deletion
    if dimension == "deletion":
        return FailureCode.DELETION_FAILURE

    # Hallucination
    if dimension == "hallucination":
        return FailureCode.HALLUCINATION

    # Generic: answer was wrong but recall had the info
    if expected_lower in recall_lower:
        return FailureCode.WRONG_FACT

    return FailureCode.RETRIEVAL_MISS


def get_taxonomy_spec() -> dict:
    """Return the full taxonomy as a structured spec for publishing."""
    return {
        "version": "1.0.0",
        "name": "Bench'd Failure Taxonomy",
        "codes": {
            code.value: {
                "name": code.name,
                "category": code.value.split("_")[0] if "_" in code.value else "other",
                "description": code.__doc__ or "",
            }
            for code in FailureCode
        },
    }
