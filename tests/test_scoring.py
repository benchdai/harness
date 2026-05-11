"""Tests for benchd_harness.scoring.deterministic."""

from benchd_harness.scoring.deterministic import score_response, aggregate_scores, ScoreResult


class TestExactMatch:
    def test_exact_match_identical(self):
        result = score_response("Marcus Chen", "Marcus Chen", "exact")
        assert result.scored_correct is True
        assert result.score == 1.0
        assert result.status == "scored"

    def test_exact_match_case_insensitive(self):
        result = score_response("marcus chen", "Marcus Chen", "exact")
        assert result.scored_correct is True
        assert result.score == 1.0

    def test_exact_match_contained(self):
        result = score_response(
            "The user's name is Marcus Chen", "Marcus Chen", "exact"
        )
        assert result.scored_correct is True
        assert result.score == 1.0

    def test_exact_match_fail(self):
        result = score_response("John Smith", "Marcus Chen", "exact")
        assert result.scored_correct is False
        assert result.score == 0.0
        assert result.status == "scored"


class TestRegexMatch:
    def test_regex_match(self):
        result = score_response("I now prefer Rust", r"[Rr]ust", "regex")
        assert result.scored_correct is True
        assert result.score == 1.0

    def test_regex_no_match(self):
        result = score_response("I prefer Python", r"[Rr]ust", "regex")
        assert result.scored_correct is False
        assert result.score == 0.0


class TestLLMScoring:
    def test_llm_returns_pending(self):
        result = score_response(
            "Some response about anything", "expected answer", "llm"
        )
        assert result.status == "pending_llm_judge"
        assert result.scored_correct is False
        assert result.score == 0.0
        assert result.scoring_method == "llm"


class TestAggregateScores:
    def test_aggregate_scores(self):
        results = [
            ScoreResult(scored_correct=True, score=1.0, max_score=1.0, scoring_method="exact", status="scored"),
            ScoreResult(scored_correct=False, score=0.0, max_score=1.0, scoring_method="exact", status="scored"),
            ScoreResult(scored_correct=True, score=1.0, max_score=1.0, scoring_method="regex", status="scored"),
            ScoreResult(scored_correct=False, score=0.0, max_score=1.0, scoring_method="llm", status="pending_llm_judge"),
        ]
        agg = aggregate_scores(results)

        assert agg["total"] == 4
        assert agg["scored"] == 3
        assert agg["pending"] == 1
        assert agg["correct"] == 2
        # 2 correct out of 3 scored, each worth 1.0 -> 2/3 * 100
        assert abs(agg["score"] - (2.0 / 3.0 * 100.0)) < 0.01
        assert agg["max_score"] == 3.0
