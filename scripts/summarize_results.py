#!/usr/bin/env python3
"""
Summarize Bench'd harness results into a formatted leaderboard table.

Reads manifest.signed.json files from the runs directory and prints
a ranked summary sorted by judged (nuance) score.

Usage:
    python scripts/summarize_results.py ./runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_manifests(runs_dir: Path) -> list[dict[str, Any]]:
    """Find and load all manifest.signed.json files."""
    manifests = []
    for mf in sorted(runs_dir.rglob("manifest.signed.json")):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "manifest" in data:
                manifests.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping {mf}: {e}", file=sys.stderr)
    return manifests


def extract_row(signed: dict[str, Any]) -> dict[str, Any]:
    """Extract a summary row from a signed manifest."""
    m = signed["manifest"]
    system = m["system"]
    scores = m["scores"]
    summary = m["summary"]
    efficiency = m["efficiency"]
    verified = scores.get("verified", {})
    nuance = scores.get("nuance", {})

    return {
        "system": system["name"],
        "adapter": system.get("adapter", system["name"]),
        "verified_overall": safe_float(verified.get("overall"), 0.0),
        "nuance_overall": safe_float(nuance.get("overall"), 0.0),
        "verified_recall": safe_float(verified.get("recall")),
        "verified_temporal": safe_float(verified.get("temporal")),
        "verified_reasoning": safe_float(verified.get("reasoning")),
        "nuance_recall": safe_float(nuance.get("recall")),
        "nuance_temporal": safe_float(nuance.get("temporal")),
        "nuance_reasoning": safe_float(nuance.get("reasoning")),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "pending": summary.get("pending_questions", 0),
        "total": summary.get("total_questions", 0),
        "mean_latency_ms": safe_float(efficiency.get("mean_latency_ms"), 0.0),
        "tokens_per_correct": safe_float(efficiency.get("tokens_per_correct_answer")),
        "total_recall_tokens": safe_float(efficiency.get("total_recall_tokens"), 0.0),
        "mean_ingest_tokens": safe_float(efficiency.get("mean_ingest_tokens"), 0.0),
        "started_at": m.get("started_at", ""),
        "completed_at": m.get("completed_at", ""),
        "run_id": m.get("run_id", ""),
        "benchmark_name": m["benchmark"]["name"],
        "benchmark_version": m["benchmark"].get("version", "?"),
        "judge_model": m.get("judge", {}).get("judge_model", "unknown"),
        "answerer_model": m.get("judge", {}).get("answerer_model", "unknown"),
        "temperature": m.get("judge", {}).get("temperature", 0.0),
    }


def fmt(val: float | None, width: int = 6, decimals: int = 1) -> str:
    if val is None:
        return "-".rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def fmt_int(val: int | None, width: int = 6) -> str:
    if val is None:
        return "-".rjust(width)
    return str(val).rjust(width)


def estimate_cost(rows: list[dict[str, Any]]) -> float:
    """Rough cost estimate based on token counts at ~$0.15/1M input tokens (gpt-4o-mini)."""
    total_tokens = 0
    for r in rows:
        # ingest tokens + recall tokens as rough proxy
        ingest = (r["mean_ingest_tokens"] or 0) * (r["total"] or 0)
        recall = r["total_recall_tokens"] or 0
        total_tokens += ingest + recall
    return total_tokens * 0.15 / 1_000_000


def print_leaderboard(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No results found.")
        return

    # Grab metadata from first row (all should share benchmark info)
    first = rows[0]
    total_q = first["total"]
    answerer = first["answerer_model"]
    judge_model = first["judge_model"]
    temp = first["temperature"]
    bench_name = first["benchmark_name"]
    bench_ver = first["benchmark_version"]

    # Sort by nuance_overall descending (the LLM-judge score)
    rows.sort(key=lambda r: r["nuance_overall"] or 0, reverse=True)

    print()
    print("BENCH'D RESULTS SUMMARY")
    print("=======================")
    print()
    print(f"{bench_name} v{bench_ver} ({total_q} questions, stratified)")
    print(f"Answerer: {answerer} | Judge: {judge_model} | Temp: {temp}")
    print()

    # Header
    hdr = (
        f"{'Rank':>4}  {'System':<22} {'Verified':>8}  {'Judged':>7}  "
        f"{'Passed':>6}  {'Failed':>6}  {'Pending':>7}  "
        f"{'Latency(ms)':>11}  {'Tokens/Correct':>14}"
    )
    sep = (
        f"{'----':>4}  {'------':<22} {'--------':>8}  {'------':>7}  "
        f"{'------':>6}  {'------':>6}  {'-------':>7}  "
        f"{'-----------':>11}  {'--------------':>14}"
    )

    print(hdr)
    print(sep)

    for rank, r in enumerate(rows, 1):
        tpc = fmt(r["tokens_per_correct"], width=14, decimals=0) if r["tokens_per_correct"] else "-".rjust(14)
        line = (
            f"{rank:>4}  {r['system']:<22} "
            f"{fmt(r['verified_overall'], 8)}  "
            f"{fmt(r['nuance_overall'], 7)}  "
            f"{fmt_int(r['passed'], 6)}  "
            f"{fmt_int(r['failed'], 6)}  "
            f"{fmt_int(r['pending'], 7)}  "
            f"{fmt(r['mean_latency_ms'], 11, 0)}  "
            f"{tpc}"
        )
        print(line)

    print()

    # Per-dimension breakdown for top system
    top = rows[0]
    print(f"Top system dimension breakdown: {top['system']}")
    print(f"  {'Dimension':<12} {'Verified':>10} {'Judged':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10}")
    for dim in ["recall", "temporal", "reasoning"]:
        v = top.get(f"verified_{dim}")
        n = top.get(f"nuance_{dim}")
        print(f"  {dim:<12} {fmt(v, 10)}  {fmt(n, 10)}")
    print()

    # Cost estimate
    cost = estimate_cost(rows)
    print(f"Estimated total API cost: ${cost:.2f}")
    print()

    # Run timestamps
    print("Run timestamps:")
    for r in rows:
        started = r["started_at"][:19].replace("T", " ") if r["started_at"] else "?"
        run_id = r["run_id"]
        print(f"  {r['system']:<22} {started}  ({run_id})")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Bench'd harness results into a leaderboard table.",
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Directory containing harness run outputs",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"Error: directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    manifests = load_manifests(runs_dir)
    if not manifests:
        print(f"No manifest.signed.json files found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(manifests)} run(s) in {runs_dir}")

    rows = [extract_row(m) for m in manifests]
    print_leaderboard(rows)


if __name__ == "__main__":
    main()
