#!/usr/bin/env python3
"""
Export harness manifest files to Bench'd frontend TypeScript data files.

Reads signed manifest JSON files produced by the benchd-harness, transforms
them into the Run and FailureTrace types expected by the Next.js frontend,
and writes generated TypeScript files.

Usage:
    python scripts/export_to_frontend.py ./runs/ --output-dir /path/to/benchd/lib/data/generated/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def find_manifest_files(runs_dir: Path) -> list[Path]:
    """Recursively find all manifest.signed.json files in the runs directory."""
    manifests = sorted(runs_dir.rglob("manifest.signed.json"))
    return manifests


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default if None or invalid."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def make_system_id(system_name: str) -> str:
    """Generate a stable system ID from a system name."""
    slug = system_name.lower().replace(" ", "-").replace("_", "-")
    return f"sys_{slug}"


def make_benchmark_id(benchmark_slug: str) -> str:
    """Generate a stable benchmark ID from a benchmark slug."""
    slug = benchmark_slug.lower().replace(" ", "-").replace("_", "-")
    return f"bench_{slug}"


def manifest_to_run(signed: dict[str, Any]) -> dict[str, Any]:
    """Convert a signed manifest JSON into a frontend Run object."""
    m = signed["manifest"]
    system = m["system"]
    benchmark = m["benchmark"]
    harness = m["harness"]
    judge = m["judge"]
    scores = m["scores"]
    summary = m["summary"]

    system_id = make_system_id(system["name"])
    benchmark_id = make_benchmark_id(benchmark["slug"])

    verified = scores.get("verified", {})
    nuance = scores.get("nuance", {})

    run_id = m["run_id"]
    merkle_root = signed.get("manifest_hash", "")
    signature = signed.get("signature", "")
    judge = judge if judge else {}

    return {
        "id": run_id,
        "systemId": system_id,
        "systemName": system["name"],
        "benchmarkId": benchmark_id,
        "benchmarkName": benchmark["name"],
        "harnessVersion": harness["version"],
        "judgeModel": judge.get("judge_model", "unknown"),
        "startedAt": m["started_at"],
        "completedAt": m["completed_at"],
        "status": "completed",
        "verifiedOverall": safe_float(verified.get("overall")),
        "nuanceOverall": safe_float(nuance.get("overall")),
        "signedReceiptUrl": f"/receipts/{run_id}.json",
        "merkleRoot": merkle_root,
        "signature": signature,
        "manifest": {
            "version": m["version"],
            "runId": run_id,
            "systemId": system_id,
            "systemName": system["name"],
            "benchmarkId": benchmark_id,
            "benchmarkName": benchmark["name"],
            "benchmarkVersion": benchmark.get("version", "1.0"),
            "harnessVersion": harness["version"],
            "judgeModel": judge.get("judge_model", "unknown"),
            "judgeTemperature": judge.get("temperature", 0.0),
            "startedAt": m["started_at"],
            "completedAt": m["completed_at"],
            "scores": {
                "verified": {
                    "recall": safe_float(verified.get("recall")),
                    "temporal": safe_float(verified.get("temporal")),
                    "reasoning": safe_float(verified.get("reasoning")),
                    "overall": safe_float(verified.get("overall")),
                },
                "nuance": {
                    "recall": safe_float(nuance.get("recall")),
                    "temporal": safe_float(nuance.get("temporal")),
                    "reasoning": safe_float(nuance.get("reasoning")),
                    "overall": safe_float(nuance.get("overall")),
                },
            },
            "questionCount": summary.get("total_questions", 0),
            "passCount": summary.get("passed", 0),
            "failCount": summary.get("failed", 0),
            "merkleRoot": merkle_root,
        },
    }


def extract_failures(signed: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract FailureTrace objects from traces where scored_correct is false."""
    m = signed["manifest"]
    run_id = m["run_id"]
    traces = m.get("traces", [])

    failures = []
    fail_counter = 0

    for i, trace in enumerate(traces):
        if trace.get("scored_correct", True):
            continue

        fail_counter += 1
        trace_id = trace.get("trace_id", f"trace_{run_id}_{i}")
        question_id = trace.get("question_id", f"q_{i}")

        # Build ingest history as ConversationTurn[]
        ingest_history = []
        for turn in trace.get("ingest_history", []):
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                ingest_history.append({
                    "role": turn["role"],
                    "content": turn["content"],
                })
            elif isinstance(turn, str):
                ingest_history.append({
                    "role": "user",
                    "content": turn,
                })

        # Build a short question summary from the query
        query = trace.get("query", "")
        question_summary = query[:80] + ("..." if len(query) > 80 else "")

        failure: dict[str, Any] = {
            "id": f"fail_{run_id}_{fail_counter:03d}",
            "runId": run_id,
            "questionId": question_id,
            "questionNumber": i + 1,
            "questionSummary": question_summary,
            "ingestHistory": ingest_history,
            "query": query,
            "response": trace.get("response", trace.get("generated_answer", "")),
            "expectedAnswer": trace.get("expected_answer", ""),
            "scoredCorrect": False,
            "scoringMethod": trace.get("scoring_method", "exact"),
            "judgeReasoning": trace.get("judge_reasoning"),
            "traceId": trace_id,
        }
        failures.append(failure)

    return failures


def to_ts_value(value: Any, indent: int = 0) -> str:
    """Convert a Python value to a TypeScript literal string."""
    prefix = "  " * indent

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Escape backslashes, backticks, and dollar signs for template-safe output
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    if isinstance(value, list):
        if not value:
            return "[]"
        items = []
        for item in value:
            items.append(f"{prefix}  {to_ts_value(item, indent + 1)}")
        return "[\n" + ",\n".join(items) + f"\n{prefix}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = []
        for k, v in value.items():
            ts_val = to_ts_value(v, indent + 1)
            entries.append(f"{prefix}  {k}: {ts_val}")
        return "{\n" + ",\n".join(entries) + f",\n{prefix}}}"

    return str(value)


def write_runs_file(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Write the generated_runs.ts file."""
    lines = ['import type { Run } from "@/lib/types";\n']
    lines.append("export const generatedRuns: Run[] = [")

    for run in runs:
        lines.append(f"  {to_ts_value(run, 1)},")

    lines.append("];\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_failures_file(failures: list[dict[str, Any]], output_path: Path) -> None:
    """Write the generated_failures.ts file."""
    lines = ['import type { FailureTrace } from "@/lib/types";\n']
    lines.append("export const generatedFailures: FailureTrace[] = [")

    for failure in failures:
        lines.append(f"  {to_ts_value(failure, 1)},")

    lines.append("];\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export harness manifest files to Bench'd frontend TypeScript data files.",
    )
    parser.add_argument(
        "runs_dir",
        type=str,
        help="Directory containing harness run outputs (searches recursively for manifest.signed.json files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated TypeScript files",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not runs_dir.is_dir():
        print(f"Error: Runs directory does not exist: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all manifest files
    manifest_files = find_manifest_files(runs_dir)
    if not manifest_files:
        print(f"No manifest.signed.json files found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(manifest_files)} manifest file(s) in {runs_dir}")
    print()

    # Parse all manifests
    all_runs: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    systems_seen: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []

    for mf in manifest_files:
        print(f"  Reading: {mf.relative_to(runs_dir)}")
        try:
            with open(mf, "r", encoding="utf-8") as f:
                signed = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"  Skipping {mf}: {e}")
            continue

        if "manifest" not in signed:
            errors.append(f"  Skipping {mf}: no 'manifest' key found")
            continue

        # Convert to Run
        run = manifest_to_run(signed)
        all_runs.append(run)

        # Track by system
        sys_name = run["systemName"]
        systems_seen.setdefault(sys_name, []).append(run)

        # Extract failures
        failures = extract_failures(signed)
        all_failures.extend(failures)

    print()

    # Report errors
    if errors:
        print("Warnings:")
        for err in errors:
            print(f"  {err}")
        print()

    # Sort runs by completedAt descending (most recent first)
    all_runs.sort(key=lambda r: r["completedAt"], reverse=True)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    runs_path = output_dir / "generated_runs.ts"
    failures_path = output_dir / "generated_failures.ts"

    write_runs_file(all_runs, runs_path)
    write_failures_file(all_failures, failures_path)

    # Print summary
    print("=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"  Manifests processed: {len(all_runs)}")
    print(f"  Failures extracted:  {len(all_failures)}")
    print(f"  Systems found:       {len(systems_seen)}")
    print()

    for sys_name, sys_runs in sorted(systems_seen.items()):
        avg_verified = sum(r["verifiedOverall"] for r in sys_runs) / len(sys_runs)
        avg_nuance = sum(r["nuanceOverall"] for r in sys_runs) / len(sys_runs)
        sys_failures = sum(1 for f in all_failures if f["runId"] in {r["id"] for r in sys_runs})
        print(f"  {sys_name}:")
        print(f"    Runs: {len(sys_runs)}")
        print(f"    Avg verified overall: {avg_verified:.1f}")
        print(f"    Avg nuance overall:   {avg_nuance:.1f}")
        print(f"    Failure traces:       {sys_failures}")
        print()

    print("Generated files:")
    print(f"  {runs_path}")
    print(f"  {failures_path}")
    print()
    print("To use in the frontend, import from these files:")
    print('  import { generatedRuns } from "@/lib/data/generated/generated_runs";')
    print('  import { generatedFailures } from "@/lib/data/generated/generated_failures";')


if __name__ == "__main__":
    main()
