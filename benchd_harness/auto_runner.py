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
from benchd_harness.runtime.base import RuntimeConfig, FailureType
from benchd_harness.runtime import get_runtime
from benchd_harness.runtime.isolation import run_isolation_check, classify_failure


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

        # Initialize runtime from manifest
        runtime_config_data = manifest.get("runtime", {})
        runtime_type = runtime_config_data.get("type", "python_library")

        # Build RuntimeConfig from manifest
        docker_config = manifest.get("docker") or {}
        isolation_config = manifest.get("isolation") or {}

        runtime_config = RuntimeConfig(
            type=runtime_type,
            command=runtime_config_data.get("command", ""),
            lifecycle=runtime_config_data.get("lifecycle", "per_run_long_lived"),
            startup_timeout_seconds=runtime_config_data.get("startup_timeout_seconds", 120),
            query_timeout_seconds=runtime_config_data.get("query_timeout_seconds", 60),
            isolation_strategy=isolation_config.get("strategy", "adapter_reset"),
            wipe_paths=isolation_config.get("wipe_paths", []),
            healthcheck_url=docker_config.get("wait_for", runtime_config_data.get("healthcheck_url", "")),
            docker_image=docker_config.get("image", runtime_config_data.get("docker_image", "")),
            docker_ports=docker_config.get("ports", []),
            docker_env=docker_config.get("env", {}),
            cwd=runtime_config_data.get("cwd", ""),
        )

        # Create runtime executor
        try:
            runtime = get_runtime(runtime_config)
        except ValueError:
            # Fallback to old Docker logic for manifests without runtime config
            runtime = None

        runtime_started = False
        healthcheck_passed = False

        if runtime:
            # Runtime lifecycle: prepare → start → healthcheck
            click.echo(f"  Runtime: {runtime_type}")

            prep = runtime.prepare()
            if not prep.success:
                click.echo(click.style(f"  Skipping — runtime prepare failed: {prep.message}", fg="yellow"))
                continue

            start_result = runtime.start()
            if not start_result.success:
                click.echo(click.style(f"  Skipping — runtime start failed: {start_result.message}", fg="yellow"))
                continue
            runtime_started = True
            if start_result.startup_ms > 0:
                click.echo(f"  Started in {start_result.startup_ms:.0f}ms")

            health = runtime.healthcheck()
            if not health.success:
                click.echo(click.style(f"  Skipping — healthcheck failed: {health.message}", fg="yellow"))
                runtime.stop()
                continue
            healthcheck_passed = True
        else:
            # Legacy: Start Docker if needed
            container_name = None
            if docker_config and docker_config.get("image"):
                click.echo(f"  Starting Docker: {docker_config['image']}...")
                container_name = start_docker(manifest)
                if container_name is None:
                    click.echo(click.style("  Skipping — Docker failed to start", fg="yellow"))
                    continue
            runtime_started = True
            healthcheck_passed = True

        try:
            # Run isolation check
            try:
                adapter_instance = get_adapter(manifest["adapter"])
                adapter_instance.setup()

                isolation_probe = run_isolation_check(adapter_instance.recall)
                if not isolation_probe.clean:
                    click.echo(click.style(
                        f"  ISOLATION FAILED: {isolation_probe.evidence}", fg="red"
                    ))
                    # Attempt wipe and retry
                    if runtime and runtime_config.isolation_strategy == "full_database_wipe":
                        click.echo("  Wiping database and retrying...")
                        runtime.wipe()
                        adapter_instance.teardown()
                        adapter_instance.setup()
                        isolation_probe = run_isolation_check(adapter_instance.recall)
                        if not isolation_probe.clean:
                            click.echo(click.style("  Isolation still failed after wipe. Skipping.", fg="red"))
                            continue
                    else:
                        click.echo(click.style("  Continuing with warning — results may be contaminated", fg="yellow"))
                else:
                    click.echo(click.style("  Isolation check: CLEAN", fg="green"))

                adapter_instance.teardown()
            except Exception as e:
                click.echo(click.style(f"  Isolation check error: {e} — continuing", fg="yellow"))

            results = run_system(
                manifest, output_dir, signer,
                use_judge=use_judge,
                benchmarks=benchmarks,
            )
            all_results.extend(results)
        finally:
            if runtime:
                runtime.cleanup()
                runtime.stop()
            elif 'container_name' in dir() and container_name:
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
