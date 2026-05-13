"""
Bench'd Auto Runner — provisions, benchmarks, and collects results for all systems.

Reads system manifest YAML files from the systems/ directory, installs deps,
starts Docker containers, runs benchmarks, collects results, and tears down.

Usage:
  benchd auto-run --systems-dir ./systems --out ./runs --key ./keys/private.key
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import click

from benchd_harness.adapters import get_adapter
from benchd_harness.benchmarks import get_benchmark
from benchd_harness.signing.local import LocalSigner
from benchd_harness.runner import BenchmarkRunner
from benchd_harness.scoring.llm_judge import LLMJudgeConfig


def load_system_manifests(systems_dir: Path) -> list[dict[str, Any]]:
    """Load all YAML system manifests from a directory."""
    manifests = []
    for f in sorted(systems_dir.glob("*.yaml")):
        with open(f) as fh:
            manifest = yaml.safe_load(fh)
            manifest["_file"] = str(f)
            manifests.append(manifest)
    return manifests


def check_env(manifest: dict) -> bool:
    """Check if required environment variables are set."""
    env_reqs = manifest.get("env", {})
    for key, requirement in env_reqs.items():
        if requirement == "required" and not os.environ.get(key):
            return False
    return True


def start_docker(manifest: dict) -> str | None:
    """Start Docker container if specified. Returns container ID or None."""
    docker_config = manifest.get("docker")
    if not docker_config:
        return None

    image = docker_config["image"]
    name = f"benchd-{manifest['name'].replace(' ', '-')}"
    ports = docker_config.get("ports", [])
    env_vars = docker_config.get("env", {})
    wait_url = docker_config.get("wait_for")
    wait_timeout = docker_config.get("wait_timeout", 60)

    # Stop existing container if any
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    # Build docker run command
    cmd = ["docker", "run", "-d", "--name", name]
    for port in ports:
        cmd.extend(["-p", port])
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(image)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(click.style(f"  Docker failed: {result.stderr.strip()}", fg="red"))
        return None

    container_id = result.stdout.strip()[:12]

    # Wait for service to be ready
    if wait_url:
        import requests
        start = time.time()
        while time.time() - start < wait_timeout:
            try:
                resp = requests.get(wait_url, timeout=5)
                if resp.status_code < 500:
                    break
            except Exception:
                pass
            time.sleep(2)

    return container_id


def stop_docker(container_name: str) -> None:
    """Stop and remove a Docker container."""
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )


def run_system(
    manifest: dict,
    output_dir: Path,
    signer: LocalSigner,
    use_judge: bool = True,
    benchmarks: list[str] | None = None,
) -> list[dict]:
    """Run all benchmarks for a single system. Returns list of run results."""
    system_name = manifest["name"]
    adapter_name = manifest["adapter"]
    system_benchmarks = benchmarks or manifest.get("benchmarks", ["longmemeval-v1"])

    results = []

    for bench_slug in system_benchmarks:
        click.echo(f"\n  Running {bench_slug}...")

        try:
            adapter = get_adapter(adapter_name)
            benchmark = get_benchmark(bench_slug)

            llm_judge_config = None
            if use_judge:
                llm_judge_config = LLMJudgeConfig(
                    answerer_model="openai/gpt-4o-mini",
                    judge_model="openai/gpt-4o-mini",
                )

            runner = BenchmarkRunner(
                adapter=adapter,
                benchmark=benchmark,
                signer=signer,
                output_dir=output_dir,
                use_llm_judge=use_judge,
                llm_judge_config=llm_judge_config,
            )

            signed = runner.run()
            m = signed.manifest

            # Extract scores
            traces = m.get("traces", [])
            total = len(traces)
            correct = sum(1 for t in traces if t.get("scored_correct"))
            overall = (correct / total * 100) if total > 0 else 0

            # Per-dimension
            dims = {}
            for t in traces:
                d = t.get("dimension", "unknown")
                if d not in dims:
                    dims[d] = {"total": 0, "correct": 0}
                dims[d]["total"] += 1
                dims[d]["correct"] += 1 if t.get("scored_correct") else 0

            result = {
                "system": system_name,
                "benchmark": bench_slug,
                "run_id": m.get("run_id"),
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "overall": round(overall, 1),
                "dimensions": {
                    d: round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0
                    for d, v in dims.items()
                },
                "total_questions": total,
                "correct": correct,
                "status": "completed",
            }
            results.append(result)

            click.echo(click.style(
                f"  {bench_slug}: {overall:.1f}% ({correct}/{total})",
                fg="green",
            ))

        except Exception as e:
            click.echo(click.style(f"  {bench_slug}: FAILED — {e}", fg="red"))
            results.append({
                "system": system_name,
                "benchmark": bench_slug,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "overall": 0,
                "status": "failed",
                "error": str(e),
            })

    return results


def append_time_series(results: list[dict], time_series_path: Path) -> None:
    """Append run results to the time series JSON file."""
    existing = []
    if time_series_path.exists():
        existing = json.loads(time_series_path.read_text())

    existing.extend(results)

    time_series_path.write_text(json.dumps(existing, indent=2))


def auto_run(
    systems_dir: Path,
    output_dir: Path,
    key_path: Path,
    use_judge: bool = True,
    systems: list[str] | None = None,
    benchmarks: list[str] | None = None,
    time_series_path: Path | None = None,
) -> list[dict]:
    """
    Run benchmarks for all (or selected) systems.

    Returns list of all run results.
    """
    manifests = load_system_manifests(systems_dir)

    if systems:
        manifests = [m for m in manifests if m["name"] in systems]

    if not manifests:
        click.echo(click.style("No system manifests found.", fg="red"))
        return []

    signer = LocalSigner(private_key_path=key_path)

    all_results = []
    total = len(manifests)

    for i, manifest in enumerate(manifests, 1):
        system_name = manifest["name"]
        click.echo(click.style(
            f"\n{'='*60}\n[{i}/{total}] {system_name}\n{'='*60}",
            fg="cyan", bold=True,
        ))

        # Check env
        if not check_env(manifest):
            missing = [
                k for k, v in manifest.get("env", {}).items()
                if v == "required" and not os.environ.get(k)
            ]
            click.echo(click.style(
                f"  Skipping — missing env: {', '.join(missing)}", fg="yellow"
            ))
            continue

        # Start Docker if needed
        container_name = None
        docker_config = manifest.get("docker")
        if docker_config:
            click.echo(f"  Starting Docker: {docker_config['image']}...")
            container_name = start_docker(manifest)
            if container_name is None and docker_config:
                click.echo(click.style("  Skipping — Docker failed to start", fg="yellow"))
                continue

        try:
            results = run_system(
                manifest, output_dir, signer,
                use_judge=use_judge,
                benchmarks=benchmarks,
            )
            all_results.extend(results)
        finally:
            # Tear down Docker
            if container_name:
                docker_name = f"benchd-{manifest['name'].replace(' ', '-')}"
                click.echo(f"  Stopping Docker: {docker_name}")
                stop_docker(docker_name)

    # Save time series
    ts_path = time_series_path or (output_dir / "time_series.json")
    append_time_series(all_results, ts_path)

    # Summary
    completed = [r for r in all_results if r.get("status") == "completed"]
    failed = [r for r in all_results if r.get("status") == "failed"]

    click.echo(click.style(f"\n{'='*60}", fg="cyan"))
    click.echo(click.style("Auto-run complete", fg="green", bold=True))
    click.echo(f"  Systems: {total}")
    click.echo(f"  Runs completed: {len(completed)}")
    click.echo(f"  Runs failed: {len(failed)}")
    click.echo(f"  Time series: {ts_path}")
    click.echo(click.style(f"{'='*60}\n", fg="cyan"))

    return all_results
