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

# ─── ProofMeter integration ────────────────────────────────
try:
    from benchd_harness.proofmeter import ProofMeterClient, ProofMeterRunContext
    from benchd_harness.proofmeter.pricing import (
        estimate_cost_cents, PRICING_VERSION,
        get_pricing_source, get_price_book_id, get_price_book_hash, is_usage_only,
    )
    _PROOFMETER_AVAILABLE = True
except ImportError:
    _PROOFMETER_AVAILABLE = False


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
        budget_cents: Optional[int] = None,
    ):
        self.adapter = adapter
        self.benchmark = benchmark
        self.signer = signer if signer is not None else LocalSigner()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.use_llm_judge = use_llm_judge
        self.llm_judge_config = llm_judge_config
        self.budget_cents = budget_cents
        self._meter_ctx: Optional["ProofMeterRunContext"] = None
        self._meter_client: Optional["ProofMeterClient"] = None

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
        budget_label = f"${self.budget_cents / 100:.2f}" if self.budget_cents else "unlimited"

        print(f"\n{'='*60}")
        print(f"  Bench'd Harness v{HARNESS_VERSION}")
        print(f"  Run:       {run_id}")
        print(f"  Benchmark: {self.benchmark.name} ({self.benchmark.version})")
        print(f"  Adapter:   {self.adapter.name}")
        print(f"  Questions: {total}")
        print(f"  Scoring:   {mode_label}")
        print(f"  Budget:    {budget_label}")
        print(f"{'='*60}\n")

        # ─── ProofMeter: Authorize budget ───────────────────────
        self._setup_proofmeter(run_id)

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

        # ─── ProofMeter: Settle and attach to manifest ─────────
        self._settle_proofmeter(manifest, run_id)

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

                # ProofMeter: record answerer spend
                self._record_spend(
                    provider=self._extract_provider(cfg.answerer_model),
                    model=self._extract_model(cfg.answerer_model),
                    input_tokens=ans_result.input_tokens,
                    output_tokens=ans_result.output_tokens,
                    question_id=item.question_id,
                    benchmark_slug=self.benchmark.slug,
                )

                # Step 3: Judge the answer
                judge_result = judge_answer(
                    query=item.recall_query,
                    expected_answer=str(item.expected_answer),
                    given_answer=generated_answer,
                    config=cfg,
                )
                judge_tokens = judge_result.input_tokens + judge_result.output_tokens

                # ProofMeter: record judge spend
                self._record_spend(
                    provider=self._extract_provider(cfg.judge_model),
                    model=self._extract_model(cfg.judge_model),
                    input_tokens=judge_result.input_tokens,
                    output_tokens=judge_result.output_tokens,
                    question_id=item.question_id,
                    benchmark_slug=self.benchmark.slug,
                )

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

    # ─── Model ID Helpers ──────────────────────────────────────

    @staticmethod
    def _extract_provider(model_id: str) -> str:
        """Extract provider from model ID like 'openai/gpt-4o' → 'openai'."""
        if "/" in model_id:
            return model_id.split("/", 1)[0]
        return "openai"  # default

    @staticmethod
    def _extract_model(model_id: str) -> str:
        """Extract model name from model ID like 'openai/gpt-4o' → 'gpt-4o'."""
        if "/" in model_id:
            return model_id.split("/", 1)[1]
        return model_id

    # ─── ProofMeter Integration ─────────────────────────────────

    def _setup_proofmeter(self, run_id: str) -> None:
        """Initialize ProofMeter budget authorization if configured."""
        if not _PROOFMETER_AVAILABLE:
            return
        if not self.budget_cents:
            return

        import os
        api_key = os.environ.get("PROOFMETER_API_KEY") or os.environ.get("VS_API_KEY")
        namespace = os.environ.get("PROOFMETER_NAMESPACE") or os.environ.get("VS_NAMESPACE")
        if not api_key or not namespace:
            print("  [ProofMeter] Skipped — PROOFMETER_API_KEY or PROOFMETER_NAMESPACE not set")
            return

        try:
            self._meter_client = ProofMeterClient(
                api_key=api_key,
                namespace_id=namespace,
            )
            self._meter_client.connect()

            budget = self._meter_client.authorize_budget(
                agent_id="benchd-runner",
                max_budget_cents=self.budget_cents,
                task_id=run_id,
            )

            self._meter_ctx = ProofMeterRunContext(
                namespace_id=namespace,
                agent_id="benchd-runner",
                task_id=run_id,
                capability_id=budget.capability_id,
                max_budget_cents=self.budget_cents,
            )

            print(f"  [ProofMeter] Budget authorized: ${self.budget_cents / 100:.2f} (cap={budget.capability_id[:16]}...)")
        except Exception as exc:
            print(f"  [ProofMeter] Authorization failed: {exc}")
            self._meter_ctx = None

    def _record_spend(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        question_id: str,
        benchmark_slug: str,
    ) -> None:
        """
        Record a usage event for an LLM call during the benchmark run.

        The receipt PROVES: provider, model, token counts, timestamp.
        Cost is ESTIMATED separately using the active pricing table.
        """
        if not self._meter_client or not self._meter_ctx:
            return

        if input_tokens == 0 and output_tokens == 0:
            return

        # Cost is an estimate, not a fact — clearly labeled
        estimated_cost = estimate_cost_cents(provider, model, input_tokens, output_tokens)

        try:
            receipt = self._meter_client.record_spend(
                actor_id=self._meter_ctx.agent_id,
                provider_id=provider,
                usage_unit="tokens",
                usage_quantity=input_tokens + output_tokens,
                cost_cents=estimated_cost,  # VS API needs this; labeled as estimate in manifest
                task_id=self._meter_ctx.task_id,
                capability_id=self._meter_ctx.capability_id,
                endpoint_class="chat",
                metadata={
                    "question_id": question_id,
                    "benchmark": benchmark_slug,
                    "adapter": self.adapter.name,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "pricing_version": PRICING_VERSION,
                    "pricing_basis": get_pricing_source(),
                    "cost_confidence": "estimated",
                },
            )

            # Track proven usage totals
            self._meter_ctx.total_input_tokens += input_tokens
            self._meter_ctx.total_output_tokens += output_tokens
            self._meter_ctx.total_tokens += input_tokens + output_tokens
            self._meter_ctx.estimated_spend_cents += estimated_cost
            self._meter_ctx.receipt_count += 1
            self._meter_ctx.receipts.append(receipt)

            # Budget enforcement (based on estimated cost under declared pricing)
            if (
                self._meter_ctx.max_budget_cents
                and self._meter_ctx.estimated_spend_cents >= self._meter_ctx.max_budget_cents
            ):
                self._meter_ctx.budget_exceeded = True
                print(f"\n  [ProofMeter] BUDGET EXCEEDED (estimated): ${self._meter_ctx.estimated_spend_cents / 100:.2f} >= ${self._meter_ctx.max_budget_cents / 100:.2f}")

        except Exception as exc:
            # Non-fatal — don't break the benchmark run for metering failures
            pass

    def _settle_proofmeter(self, manifest: dict, run_id: str) -> None:
        """Settle the run and attach ProofMeter metadata to the manifest."""
        if not self._meter_client or not self._meter_ctx:
            return

        try:
            settlement = self._meter_client.settle(
                task_id=run_id,
                capability_id=self._meter_ctx.capability_id,
            )
            self._meter_ctx.settlement = settlement

            model_breakdown = self._build_model_breakdown()

            # Manifest separates PROVEN USAGE from ESTIMATED COST
            manifest["proofmeter"] = {
                "version": "1.1",
                "spec_url": "https://benchd.ai/methodology/receipt-spec",
                "capability_id": self._meter_ctx.capability_id,
                "receipt_count": self._meter_ctx.receipt_count,
                "budget_exceeded": self._meter_ctx.budget_exceeded,

                # ── Proven usage (signed, verifiable) ────────────
                "proven_usage": {
                    "total_input_tokens": self._meter_ctx.total_input_tokens,
                    "total_output_tokens": self._meter_ctx.total_output_tokens,
                    "total_tokens": self._meter_ctx.total_tokens,
                    "by_model": model_breakdown,
                },

                # ── Cost estimate (derived, not a fact) ──────────
                "cost_estimate": {
                    "estimated_total_usd": f"{self._meter_ctx.estimated_spend_cents / 100:.4f}" if not is_usage_only() else None,
                    "cost_confidence": "usage_only" if is_usage_only() else "estimated",
                    "pricing_basis": get_pricing_source(),
                    "price_book_id": get_price_book_id(),
                    "price_book_hash": get_price_book_hash(),
                    "pricing_version": PRICING_VERSION,
                    "note": "Usage only — no cost calculated." if is_usage_only() else "Estimated from declared pricing table. Actual billed amount may differ based on customer contract, credits, or discounts.",
                },

                # ── Budget (denominated in declared pricing) ─────
                "budget": {
                    "authorized_cents": self._meter_ctx.max_budget_cents,
                    "authorized_usd": f"{self._meter_ctx.max_budget_cents / 100:.2f}" if self._meter_ctx.max_budget_cents else None,
                    "pricing_basis": get_pricing_source(),
                    "price_book_id": get_price_book_id(),
                },

                # ── Settlement (crypto) ──────────────────────────
                "settlement": {
                    "settlement_id": settlement.settlement_id,
                    "merkle_root": settlement.merkle_root,
                    "signature": settlement.signature,
                    "status": settlement.status,
                },
            }

            est_usd = self._meter_ctx.estimated_spend_cents / 100
            tok = self._meter_ctx.total_tokens
            print(f"\n  [ProofMeter] Settled: {tok:,} tokens | ~${est_usd:.2f} (estimated, {get_pricing_source()}) | {self._meter_ctx.receipt_count} receipts")

        except Exception as exc:
            print(f"\n  [ProofMeter] Settlement failed: {exc}")
            model_breakdown = self._build_model_breakdown()
            manifest["proofmeter"] = {
                "version": "1.1",
                "spec_url": "https://benchd.ai/methodology/receipt-spec",
                "capability_id": self._meter_ctx.capability_id,
                "receipt_count": self._meter_ctx.receipt_count,
                "budget_exceeded": self._meter_ctx.budget_exceeded,
                "proven_usage": {
                    "total_input_tokens": self._meter_ctx.total_input_tokens,
                    "total_output_tokens": self._meter_ctx.total_output_tokens,
                    "total_tokens": self._meter_ctx.total_tokens,
                    "by_model": model_breakdown,
                },
                "cost_estimate": {
                    "estimated_total_usd": f"{self._meter_ctx.estimated_spend_cents / 100:.4f}" if not is_usage_only() else None,
                    "cost_confidence": "usage_only" if is_usage_only() else "estimated",
                    "pricing_basis": get_pricing_source(),
                    "price_book_id": get_price_book_id(),
                    "price_book_hash": get_price_book_hash(),
                    "pricing_version": PRICING_VERSION,
                },
                "settlement": {
                    "status": "failed",
                    "error": str(exc),
                },
            }

        finally:
            self._meter_client.close()

    def _build_model_breakdown(self) -> dict:
        """Build per-model USAGE breakdown from receipts (proven facts, not cost)."""
        if not self._meter_ctx:
            return {}

        breakdown: dict[str, dict] = {}
        for receipt in self._meter_ctx.receipts:
            meta = {}
            if hasattr(receipt, 'metadata') and receipt.metadata:
                meta = receipt.metadata if isinstance(receipt.metadata, dict) else {}

            model = meta.get("model", receipt.provider_id)
            key = f"{receipt.provider_id}/{model}"

            in_tok = meta.get("input_tokens", 0)
            out_tok = meta.get("output_tokens", 0)

            if key not in breakdown:
                breakdown[key] = {
                    "provider": receipt.provider_id,
                    "model": model,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

            entry = breakdown[key]
            entry["calls"] += 1
            entry["input_tokens"] += in_tok
            entry["output_tokens"] += out_tok
            entry["total_tokens"] += in_tok + out_tok

        return breakdown
