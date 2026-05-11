"""Tests for benchd_harness.manifest."""

import re
from datetime import datetime, timezone

from benchd_harness.manifest import build_manifest, generate_run_id, TraceRecord


def _make_trace(
    question_id: str,
    dimension: str,
    scoring_method: str,
    scored_correct: bool,
    status: str = "scored",
) -> TraceRecord:
    return TraceRecord(
        trace_id=f"trace_{question_id}",
        question_id=question_id,
        dimension=dimension,
        ingest_history=[{"role": "user", "content": "test"}],
        query="test query",
        raw_recall="test recall",
        generated_answer="",
        response="test response",
        expected_answer="test answer",
        scored_correct=scored_correct,
        scoring_method=scoring_method,
        score=1.0 if scored_correct else 0.0,
        max_score=1.0,
        judge_reasoning=None,
        status=status,
    )


class TestBuildManifest:
    def test_build_manifest_structure(self):
        now = datetime.now(timezone.utc)
        traces = [
            _make_trace("q1", "recall", "exact", True),
            _make_trace("q2", "temporal", "regex", False),
        ]
        manifest = build_manifest(
            run_id="run_abc123",
            system_name="echo",
            adapter_name="echo",
            adapter_version="0.1.0",
            benchmark_slug="smoke-memory-v0",
            benchmark_name="Smoke Memory v0",
            benchmark_version="0.1.0",
            harness_version="0.1.0",
            started_at=now,
            completed_at=now,
            traces=traces,
        )

        assert "version" in manifest
        assert "run_id" in manifest
        assert "system" in manifest
        assert "benchmark" in manifest
        assert "harness" in manifest
        assert "scores" in manifest
        assert "summary" in manifest
        assert "traces" in manifest
        assert manifest["run_id"] == "run_abc123"
        assert manifest["system"]["adapter"] == "echo"
        assert manifest["benchmark"]["slug"] == "smoke-memory-v0"

    def test_build_manifest_scores(self):
        now = datetime.now(timezone.utc)
        traces = [
            _make_trace("q1", "recall", "exact", True),
            _make_trace("q2", "recall", "exact", True),
            _make_trace("q3", "recall", "exact", False),
            _make_trace("q4", "temporal", "regex", True),
            _make_trace("q5", "reasoning", "llm", False, status="pending_llm_judge"),
        ]
        manifest = build_manifest(
            run_id="run_test",
            system_name="echo",
            adapter_name="echo",
            adapter_version="0.1.0",
            benchmark_slug="smoke-memory-v0",
            benchmark_name="Smoke Memory v0",
            benchmark_version="0.1.0",
            harness_version="0.1.0",
            started_at=now,
            completed_at=now,
            traces=traces,
        )

        scores = manifest["scores"]
        # recall: 2 correct out of 3 exact -> ~66.67%
        assert abs(scores["verified"]["recall"] - (2.0 / 3.0 * 100)) < 0.01
        # temporal: 1 correct out of 1 regex -> 100%
        assert scores["verified"]["temporal"] == 100.0
        # reasoning has no exact/regex items -> None
        assert scores["verified"]["reasoning"] is None
        # nuance reasoning: llm pending -> None
        assert scores["nuance"]["reasoning"] is None

        summary = manifest["summary"]
        assert summary["total_questions"] == 5
        assert summary["scored_questions"] == 4
        assert summary["pending_questions"] == 1
        assert summary["passed"] == 3
        assert summary["failed"] == 1


class TestGenerateRunId:
    def test_generate_run_id_format(self):
        run_id = generate_run_id()
        assert run_id.startswith("run_")
        assert len(run_id) == 16  # "run_" (4) + 12 hex chars
        assert re.fullmatch(r"run_[0-9a-f]{12}", run_id)
