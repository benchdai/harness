"""Bench'd Harness CLI — neutral benchmark runner for AI memory systems."""

import json
import sys
from pathlib import Path

import click

from benchd_harness.adapters import get_adapter, _BUILTIN_ADAPTERS
from benchd_harness.benchmarks import get_benchmark, list_benchmarks
from benchd_harness.signing import generate_keypair, verify_signature
from benchd_harness.signing.local import LocalSigner, _fingerprint_from_public_key_hex
from benchd_harness.runner import BenchmarkRunner


@click.group()
@click.version_option(version="0.1.0", prog_name="benchd")
def cli():
    """Bench'd Harness — neutral benchmark runner for AI memory systems."""
    pass


@cli.command()
@click.option("--adapter", "-a", required=True, help="Adapter name (e.g., echo, null)")
@click.option("--benchmark", "-b", required=True, help="Benchmark slug (e.g., smoke-memory-v0)")
@click.option("--out", "-o", type=click.Path(), default="./runs", help="Output directory for results")
@click.option("--key", "-k", type=click.Path(exists=True), help="Path to private key file for signing")
@click.option("--max-items", "-n", type=int, default=None, help="Limit number of benchmark items (for quick test runs)")
@click.option("--judge/--no-judge", default=False, help="Enable LLM answerer + judge pipeline (requires API key)")
@click.option("--answerer-model", default="openai/gpt-4o-mini", help="Answerer LLM model ID")
@click.option("--judge-model", default="openai/gpt-4o-mini", help="Judge LLM model ID")
@click.option("--adapter-config", default=None, help="JSON string of adapter configuration (e.g., '{\"endpoint\": \"...\"}')")
def run(adapter, benchmark, out, key, max_items, judge, answerer_model, judge_model, adapter_config):
    """Run a benchmark against a memory system adapter."""
    # Parse adapter config JSON
    parsed_adapter_config = None
    if adapter_config:
        try:
            parsed_adapter_config = json.loads(adapter_config)
        except json.JSONDecodeError as exc:
            click.echo(click.style(f"Invalid --adapter-config JSON: {exc}", fg="red"), err=True)
            sys.exit(1)

    # Resolve adapter
    try:
        adapter_instance = get_adapter(adapter, adapter_config=parsed_adapter_config)
    except ValueError as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Resolve benchmark
    try:
        kwargs = {}
        if max_items is not None:
            kwargs["max_items"] = max_items
        benchmark_instance = get_benchmark(benchmark, **kwargs)
    except ValueError as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Create signer
    if key:
        signer = LocalSigner(private_key_path=Path(key))
    else:
        signer = LocalSigner()

    output_dir = Path(out)

    # Configure LLM judge if enabled
    llm_judge_config = None
    if judge:
        from benchd_harness.scoring.llm_judge import LLMJudgeConfig
        llm_judge_config = LLMJudgeConfig(
            answerer_model=answerer_model,
            judge_model=judge_model,
        )

    # Create runner and execute
    runner = BenchmarkRunner(
        adapter=adapter_instance,
        benchmark=benchmark_instance,
        signer=signer,
        output_dir=output_dir,
        use_llm_judge=judge,
        llm_judge_config=llm_judge_config,
    )

    try:
        signed = runner.run()
    except Exception as exc:
        click.echo(click.style(f"Run failed: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Build summary
    manifest = signed.manifest
    run_id = manifest.get("run_id", "unknown")
    system_name = manifest.get("system_name", adapter)
    benchmark_slug = manifest.get("benchmark_slug", benchmark)

    # Compute scores from traces
    traces = manifest.get("traces", [])
    total = len(traces)
    scored_traces = [t for t in traces if t.get("status") == "scored"]
    pending_traces = [t for t in traces if t.get("status") != "scored"]
    scored_count = len(scored_traces)
    pending_count = len(pending_traces)

    passed = sum(1 for t in scored_traces if t.get("scored_correct"))
    failed = scored_count - passed

    # Compute verified score
    total_score = sum(t.get("score", 0) for t in scored_traces)
    total_max = sum(t.get("max_score", 0) for t in scored_traces)
    if total_max > 0:
        verified_pct = (total_score / total_max) * 100
        score_display = f"{verified_pct:.1f} / 100"
    else:
        score_display = "N/A"

    # Determine nuance score status
    if pending_count > 0:
        nuance_display = "pending (LLM judge not connected)"
    else:
        nuance_display = "N/A"

    # Manifest file path
    manifest_path = output_dir / run_id / "manifest.signed.json"

    # Print summary
    separator = click.style("=" * 50, fg="cyan")
    click.echo(f"\n{separator}")
    click.echo(click.style(f"Run complete: {run_id}", fg="green", bold=True))
    click.echo(f"System: {system_name} | Benchmark: {benchmark_slug}")
    click.echo()
    click.echo(f"Verified Score:  {click.style(score_display, bold=True)}")
    click.echo(f"Nuance Score:    {nuance_display}")
    click.echo()
    click.echo(f"Questions: {total} total, {scored_count} scored, {pending_count} pending")
    click.echo(f"Passed: {click.style(str(passed), fg='green')} | Failed: {click.style(str(failed), fg='red')}")
    click.echo()
    click.echo(f"Manifest saved: {click.style(str(manifest_path), fg='blue')}")
    click.echo(separator)


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True))
def verify(manifest_path):
    """Verify a signed benchmark manifest."""
    # Load the signed manifest JSON
    path = Path(manifest_path)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        click.echo(click.style(f"Error reading manifest: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Verify signature
    is_valid = verify_signature(data)

    if is_valid:
        manifest = data.get("manifest", {})
        run_id = manifest.get("run_id", "unknown")
        system_name = manifest.get("system_name", "unknown")
        benchmark_slug = manifest.get("benchmark_slug", "unknown")
        signed_at = data.get("signed_at", "unknown")
        fingerprint = data.get("signing_key_fingerprint", "unknown")

        # Compute verified score
        traces = manifest.get("traces", [])
        scored_traces = [t for t in traces if t.get("status") == "scored"]
        total_score = sum(t.get("score", 0) for t in scored_traces)
        total_max = sum(t.get("max_score", 0) for t in scored_traces)
        if total_max > 0:
            verified_score = f"{(total_score / total_max) * 100:.1f}"
        else:
            verified_score = "N/A"

        click.echo(click.style("VERIFIED", fg="green", bold=True) + " — signature valid")
        click.echo(f"Run: {run_id}")
        click.echo(f"System: {system_name}")
        click.echo(f"Benchmark: {benchmark_slug}")
        click.echo(f"Verified Score: {verified_score}")
        click.echo(f"Signed at: {signed_at}")
        click.echo(f"Fingerprint: {fingerprint}")
    else:
        click.echo(click.style("INVALID", fg="red", bold=True) + " — signature verification failed", err=True)
        sys.exit(1)


@cli.group()
def keys():
    """Manage signing keys."""
    pass


@keys.command("generate")
@click.option("--out", "-o", type=click.Path(), default="./keys", help="Output directory for keypair")
def keys_generate(out):
    """Generate a new Ed25519 signing keypair."""
    output_dir = Path(out)

    try:
        private_path, public_path = generate_keypair(output_dir)
    except Exception as exc:
        click.echo(click.style(f"Error generating keypair: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Compute fingerprint from public key
    public_hex = public_path.read_text().strip()
    fingerprint = _fingerprint_from_public_key_hex(public_hex)

    click.echo(click.style("Keypair generated successfully!", fg="green", bold=True))
    click.echo()
    click.echo(f"  Private key: {click.style(str(private_path), fg='yellow')}")
    click.echo(f"  Public key:  {click.style(str(public_path), fg='blue')}")
    click.echo(f"  Fingerprint: {fingerprint}")
    click.echo()
    click.echo(click.style("Keep your private key safe — do not commit it to version control.", fg="yellow"))


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option("--api-url", default="https://benchd.ai/api/submit", help="Submission API endpoint")
def submit(manifest_path, api_url):
    """Submit a signed manifest to the Bench'd leaderboard."""
    import requests as req

    path = Path(manifest_path)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        click.echo(click.style(f"Error reading manifest: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Validate before submitting
    if "manifest" not in data or "signature" not in data:
        click.echo(click.style("Invalid manifest: missing 'manifest' or 'signature' field", fg="red"), err=True)
        sys.exit(1)

    # Verify signature locally first
    is_valid = verify_signature(data)
    if not is_valid:
        click.echo(click.style("Signature verification failed. Cannot submit.", fg="red"), err=True)
        sys.exit(1)

    # Extract summary for display
    manifest = data.get("manifest", {})
    if isinstance(manifest, str):
        manifest = json.loads(manifest)

    system_name = manifest.get("system", {}).get("name", "unknown")
    benchmark_name = manifest.get("benchmark", {}).get("name", "unknown")
    run_id = manifest.get("run_id", "unknown")
    scores = manifest.get("scores", {})
    nuance_overall = scores.get("nuance", {}).get("overall")
    verified_overall = scores.get("verified", {}).get("overall")

    click.echo(click.style("Manifest validated locally", fg="green"))
    click.echo(f"  Run:       {run_id}")
    click.echo(f"  System:    {system_name}")
    click.echo(f"  Benchmark: {benchmark_name}")
    if verified_overall is not None:
        click.echo(f"  Verified:  {verified_overall:.1f}")
    if nuance_overall is not None:
        click.echo(f"  Nuance:    {nuance_overall:.1f}")
    click.echo()

    # Submit to API
    click.echo(f"Submitting to {api_url}...")
    try:
        resp = req.post(
            api_url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code == 200 or resp.status_code == 201:
            click.echo(click.style("Submitted successfully!", fg="green", bold=True))
            result = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if result.get("url"):
                click.echo(f"  View at: {result['url']}")
        elif resp.status_code == 422:
            click.echo(click.style("Submission rejected — invalid manifest format", fg="red"), err=True)
            sys.exit(1)
        else:
            # API might not be connected yet — save locally
            click.echo(click.style(
                f"API returned {resp.status_code}. Manifest saved locally — "
                "you can also upload at https://benchd.ai/submit",
                fg="yellow"
            ))
    except req.exceptions.ConnectionError:
        click.echo(click.style(
            "Could not reach submission API. Manifest saved locally — "
            "upload at https://benchd.ai/submit",
            fg="yellow"
        ))
    except Exception as exc:
        click.echo(click.style(f"Submission error: {exc}", fg="yellow"))
        click.echo("Manifest saved locally — upload at https://benchd.ai/submit")


@cli.command("auto-run")
@click.option("--systems-dir", "-s", type=click.Path(exists=True), default="./systems", help="Directory with system YAML manifests")
@click.option("--out", "-o", type=click.Path(), default="./runs", help="Output directory for results")
@click.option("--key", "-k", type=click.Path(exists=True), default="./keys/private.key", help="Path to private key")
@click.option("--judge/--no-judge", default=True, help="Enable LLM judge pipeline")
@click.option("--system", "system_names", multiple=True, help="Run only specific systems (can repeat)")
@click.option("--benchmark", "benchmark_slugs", multiple=True, help="Run only specific benchmarks (can repeat)")
@click.option("--time-series", type=click.Path(), default=None, help="Path to time series JSON file")
def auto_run_cmd(systems_dir, out, key, judge, system_names, benchmark_slugs, time_series):
    """Run benchmarks for all systems defined in YAML manifests."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        click.echo(click.style("PyYAML required. Install: pip install pyyaml", fg="red"), err=True)
        sys.exit(1)

    from benchd_harness.auto_runner import auto_run

    results = auto_run(
        systems_dir=Path(systems_dir),
        output_dir=Path(out),
        key_path=Path(key),
        use_judge=judge,
        systems=list(system_names) if system_names else None,
        benchmarks=list(benchmark_slugs) if benchmark_slugs else None,
        time_series_path=Path(time_series) if time_series else None,
    )

    if not results:
        sys.exit(1)


@cli.command("list")
def list_available():
    """List available adapters and benchmarks."""
    click.echo(click.style("Adapters:", fg="cyan", bold=True))
    for name in sorted(_BUILTIN_ADAPTERS):
        click.echo(f"  - {name}")

    click.echo()

    click.echo(click.style("Benchmarks:", fg="cyan", bold=True))
    for slug in list_benchmarks():
        click.echo(f"  - {slug}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
