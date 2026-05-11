"""Benchmark runner (orchestrator) for the Bench'd harness."""

import json
import time
import uuid
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from benchd_harness.adapters.base import BaseAdapter
from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem
from benchd_harness.scoring.deterministic import score_response, score_retrieval_recall, compute_retrieval_metrics, compute_faithfulness, ScoreResult
from benchd_harness.manifest import build_manifest, TraceRecord, generate_run_id, estimate_tokens
from benchd_harness.signing.local import LocalSigner, SignedManifest

HARNESS_VERSION = "0.1.0"


class BenchmarkRunner:
    """Orchestrates a benchmark run: ingest → recall → answer → judge → manifest → sign."""

    def __init__(
        self,
        adapter: BaseAdapter,
        benchmark: BaseBenchmark,
        signer: Optional[LocalSigner] = None,
        output_dir: Optional[Path] = None,
        use_llm_judge: bool = False,
        llm_judge_config: Optional[object] = None,
    ):
        self.adapter = adapter
        self.benchmark = benchmark
        self.signer = signer if signer is not None else LocalSigner()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.use_llm_judge = use_llm_judge
        self.llm_judge_config = llm_judge_config

    def run(self) -> SignedManifest:
        """
        Execute the full benchmark pipeline:
        1. Generate run_id
        2. Call adapter.setup()
        3. For each benchmark item:
           a. Call adapter.reset()
           b. Call adapter.ingest(item.ingest_history)
           c. Call adapter.recall(item.recall_query)
           d. If LLM judge enabled: generate answer from retrieved memories
           e. Score: deterministic first, then LLM judge if enabled
           f. Record the trace
        4. Call adapter.teardown()
        5. Build manifest from traces
        6. Sign manifest
        7. Save to output_dir if specified
        8. Return SignedManifest
        """
        run_id = generate_run_id()
        started_at = datetime.now(timezone.utc)
        items = self.benchmark.load_items()
        total = len(items)

        mode_label = "full pipeline (answerer + judge)" if self.use_llm_judge else "deterministic only"

        print(f"\n{'='*60}")
        print(f"  Bench'd Harness v{HARNESS_VERSION}")
        print(f"  Run:       {run_id}")
        print(f"  Benchmark: {self.benchmark.name} ({self.benchmark.version})")
        print(f"  Adapter:   {self.adapter.name}")
        print(f"  Questions: {total}")
        print(f"  Scoring:   {mode_label}")
        print(f"{'='*60}\n")

        self.adapter.setup()

        traces: list[TraceRecord] = []
        passed = 0
        failed = 0
        pending = 0

        for i, item in enumerate(items, 1):
            trace = self._run_item(item)
            traces.append(trace)

            if trace.scored_correct:
                status_label = "PASS"
                passed += 1
            elif trace.status == "pending_llm_judge":
                status_label = "PENDING"
                pending += 1
            else:
                status_label = "FAIL"
                failed += 1

            # Truncate query for display
            query_display = item.recall_query
            if len(query_display) > 55:
                query_display = query_display[:52] + "..."

            print(f"  [{i:>{len(str(total))}}/{total}] {item.dimension:<10} {query_display:<57} {status_label}")

        self.adapter.teardown()
        completed_at = datetime.now(timezone.utc)

        elapsed = (completed_at - started_at).total_seconds()
        latencies = [t.latency_ms for t in traces]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        total_recall_tok = sum(t.recall_tokens for t in traces)
        mean_ingest_tok = sum(t.ingest_tokens for t in traces) / len(traces) if traces else 0.0

        print(f"\n{'='*60}")
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Passed: {passed}  Failed: {failed}  Pending: {pending}")
        print(f"  --- Efficiency ---")
        print(f"  Mean latency:       {mean_lat:.1f} ms")
        print(f"  Total recall tokens: {total_recall_tok}")
        print(f"  Mean ingest tokens:  {mean_ingest_tok:.0f}")
        if passed > 0:
            print(f"  Tokens/correct ans:  {total_recall_tok / passed:.1f}")
        print(f"{'='*60}\n")

        # Build judge config metadata for manifest
        judge_metadata = None
        if self.use_llm_judge and self.llm_judge_config:
            from benchd_harness.scoring.llm_judge import LLMJudgeConfig
            cfg = self.llm_judge_config if isinstance(self.llm_judge_config, LLMJudgeConfig) else LLMJudgeConfig()
            judge_metadata = {
                "answerer_model": cfg.answerer_model,
                "judge_model": cfg.judge_model,
                "temperature": cfg.temperature,
            }

        manifest = build_manifest(
            run_id=run_id,
            system_name=self.adapter.name,
            adapter_name=self.adapter.name,
            adapter_version=self.adapter.version,
            benchmark_slug=self.benchmark.slug,
            benchmark_name=self.benchmark.name,
            benchmark_version=self.benchmark.version,
            harness_version=HARNESS_VERSION,
            started_at=started_at,
            completed_at=completed_at,
            traces=traces,
            judge_metadata=judge_metadata,
        )

        signed = self.signer.sign_manifest(manifest)

        if self.output_dir is not None:
            saved_path = self._save_results(signed, run_id)
            print(f"  Results saved to: {saved_path}\n")

        return signed

    @staticmethod
    def _estimate_ingest_tokens(ingest_history: list) -> int:
        """Estimate token count for the ingest history."""
        import json as _json
        text = _json.dumps(ingest_history, default=str)
        return estimate_tokens(text)

    def _run_item(self, item: BenchmarkItem) -> TraceRecord:
        """Run a single benchmark item and return its trace."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        raw_recall = ""
        generated_answer = ""
        answerer_tokens = 0
        judge_tokens = 0

        ingest_tok = self._estimate_ingest_tokens(item.ingest_history)

        try:
            self.adapter.reset()
            t0 = time.perf_counter()
            self.adapter.ingest(item.ingest_history)
            raw_recall = self.adapter.recall(item.recall_query)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            latency_ms = 0.0
            raw_recall = f"[ADAPTER ERROR] {type(exc).__name__}: {exc}"
            return TraceRecord(
                trace_id=trace_id,
                question_id=item.question_id,
                dimension=item.dimension,
                ingest_history=item.ingest_history,
                query=item.recall_query,
                raw_recall=raw_recall,
                generated_answer="",
                response=raw_recall,
                expected_answer=str(item.expected_answer),
                scored_correct=False,
                scoring_method=item.scoring_method,
                score=0.0,
                max_score=item.max_score,
                judge_reasoning=f"Adapter raised an exception: {exc}",
                status="scored",
                latency_ms=latency_ms,
                recall_tokens=estimate_tokens(raw_recall),
                ingest_tokens=ingest_tok,
                retrieval_hit=False,
            )

        recall_tok = estimate_tokens(raw_recall)

        # Compute full retrieval metrics
        retrieval_hit = score_retrieval_recall(raw_recall, str(item.expected_answer))
        ret_metrics = compute_retrieval_metrics(raw_recall, str(item.expected_answer))
        # Helper: common metric fields for all TraceRecord paths
        _metric_fields = dict(
            latency_ms=latency_ms,
            recall_tokens=recall_tok,
            ingest_tokens=ingest_tok,
            retrieval_hit=retrieval_hit,
            partial_hit=ret_metrics.partial_hit,
            word_overlap=ret_metrics.word_overlap,
            answer_density=ret_metrics.answer_density,
            compression_ratio=ret_metrics.compression_ratio,
        )

        # Full pipeline: answerer LLM generates answer from retrieved memories
        if self.use_llm_judge:
            try:
                from benchd_harness.scoring.llm_judge import (
                    generate_answer,
                    judge_answer,
                    LLMJudgeConfig,
                )

                cfg = (
                    self.llm_judge_config
                    if isinstance(self.llm_judge_config, LLMJudgeConfig)
                    else LLMJudgeConfig()
                )

                # Step 2: Generate answer from retrieved memories
                ans_result = generate_answer(
                    query=item.recall_query,
                    retrieved_memories=raw_recall,
                    config=cfg,
                )
                generated_answer = ans_result.answer
                answerer_tokens = ans_result.input_tokens + ans_result.output_tokens

                # Step 3: Judge the answer
                judge_result = judge_answer(
                    query=item.recall_query,
                    expected_answer=str(item.expected_answer),
                    given_answer=generated_answer,
                    config=cfg,
                )
                judge_tokens = judge_result.input_tokens + judge_result.output_tokens

                return TraceRecord(
                    trace_id=trace_id,
                    question_id=item.question_id,
                    dimension=item.dimension,
                    ingest_history=item.ingest_history,
                    query=item.recall_query,
                    raw_recall=raw_recall,
                    generated_answer=generated_answer,
                    response=generated_answer,
                    expected_answer=str(item.expected_answer),
                    scored_correct=judge_result.correct,
                    scoring_method="llm",
                    score=item.max_score if judge_result.correct else 0.0,
                    max_score=item.max_score,
                    judge_reasoning=judge_result.reasoning,
                    status="scored",
                    **_metric_fields,
                    **self._faithfulness_metrics(raw_recall, generated_answer, str(item.expected_answer)),
                )

            except Exception as exc:
                # If judge fails, fall back to deterministic scoring
                generated_answer = f"[JUDGE ERROR] {exc}"

        # Deterministic-only scoring (or fallback)
        response_to_score = generated_answer if generated_answer and not generated_answer.startswith("[JUDGE ERROR]") else raw_recall

        result: ScoreResult = score_response(
            response=response_to_score,
            expected_answer=str(item.expected_answer),
            scoring_method=item.scoring_method,
            max_score=item.max_score,
        )

        return TraceRecord(
            trace_id=trace_id,
            question_id=item.question_id,
            dimension=item.dimension,
            ingest_history=item.ingest_history,
            query=item.recall_query,
            raw_recall=raw_recall,
            generated_answer=generated_answer,
            response=response_to_score,
            expected_answer=str(item.expected_answer),
            scored_correct=result.scored_correct,
            scoring_method=result.scoring_method,
            score=result.score,
            max_score=result.max_score,
            judge_reasoning=result.judge_reasoning,
            status=result.status,
            **_metric_fields,
            **self._faithfulness_metrics(raw_recall, generated_answer or raw_recall, str(item.expected_answer)),
        )

    @staticmethod
    def _faithfulness_metrics(raw_recall: str, generated_answer: str, expected: str) -> dict:
        """Compute faithfulness fields for TraceRecord."""
        f = compute_faithfulness(raw_recall, generated_answer, expected)
        return {
            "grounded": f["grounded"],
            "hallucination_risk": f["hallucination_risk"],
            "abstained": f["abstained"],
        }

    def _save_results(self, signed: SignedManifest, run_id: str) -> Path:
        """Save signed manifest to output_dir/run_id/manifest.signed.json."""
        assert self.output_dir is not None
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "manifest.signed.json"
        manifest_path.write_text(signed.to_json(indent=2) + "\n")

        return manifest_path
