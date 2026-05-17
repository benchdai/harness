"""
Track baseline computation for Bench'd.

Computes population statistics (mean, p25, p75) from all scored systems
within a track. These baselines are used to populate expectation_profile
in StructuredScore objects, giving context like "this system is above/below
the track average."

Baselines are recomputed whenever new systems are scored, and cached
for UI consumption.
"""

import json
import statistics
from pathlib import Path
from typing import Optional


def compute_track_baselines(
    manifest_dir: Path,
    track_id: str,
    metric_id: str,
) -> dict:
    """
    Compute population baselines for a specific track+metric combination.

    Scans all signed manifests in manifest_dir, filters to those matching
    the track, and computes statistics on the overall verified score.

    Args:
        manifest_dir: Directory containing run_id/manifest.signed.json files.
        track_id: Track to filter by (e.g., "conversational", "knowledge-brain").
        metric_id: Benchmark slug to filter by (e.g., "knowledge-retrieval").

    Returns:
        Dict with keys: track_mean, track_p25, track_p75, sample_size, systems.
    """
    scores: list[float] = []
    systems: list[str] = []

    if not manifest_dir.exists():
        return _empty_baseline(track_id)

    for run_dir in manifest_dir.iterdir():
        if not run_dir.is_dir():
            continue

        manifest_path = run_dir / "manifest.signed.json"
        if not manifest_path.exists():
            continue

        try:
            data = json.loads(manifest_path.read_text())
            manifest = data.get("manifest", data)

            # Filter by metric
            benchmark_slug = manifest.get("benchmark", {}).get("slug", "")
            if benchmark_slug != metric_id:
                continue

            # Get the structured score track, or default to baseline
            structured = manifest.get("structured_score", {})
            manifest_track = structured.get("track_id", "baseline")

            # Include if track matches, or if querying baseline (include all)
            if track_id != "baseline" and manifest_track != track_id:
                continue

            # Extract the overall verified score
            verified_overall = manifest.get("scores", {}).get("verified", {}).get("overall")
            if verified_overall is not None:
                scores.append(verified_overall)
                system_name = manifest.get("system", {}).get("name", "unknown")
                systems.append(system_name)

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    if not scores:
        return _empty_baseline(track_id)

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    return {
        "track_id": track_id,
        "metric_id": metric_id,
        "track_mean": round(statistics.mean(sorted_scores), 2),
        "track_p25": round(_percentile(sorted_scores, 25), 2),
        "track_p75": round(_percentile(sorted_scores, 75), 2),
        "sample_size": n,
        "systems": systems,
    }


def compute_all_baselines(
    manifest_dir: Path,
    tracks: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
) -> dict[str, dict[str, dict]]:
    """
    Compute baselines for all track/metric combinations.

    Returns:
        Nested dict: baselines[track_id][metric_id] = baseline_dict
    """
    if tracks is None:
        tracks = ["conversational", "knowledge-brain", "graph", "agent-memory", "baseline"]

    if metrics is None:
        metrics = [
            "knowledge-retrieval",
            "knowledge-scale",
            "longmemeval",
            "locomo",
            "truth-arbitration",
            "memory-poisoning",
            "budget-curves",
            "reliability",
        ]

    baselines: dict[str, dict[str, dict]] = {}
    for track in tracks:
        baselines[track] = {}
        for metric in metrics:
            baselines[track][metric] = compute_track_baselines(manifest_dir, track, metric)

    return baselines


def save_baselines_cache(baselines: dict, cache_path: Path) -> None:
    """Write baselines to a JSON cache file for UI consumption."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(baselines, indent=2) + "\n")


def load_baselines_cache(cache_path: Path) -> Optional[dict]:
    """Load baselines from cache. Returns None if cache doesn't exist."""
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _percentile(sorted_data: list[float], pct: int) -> float:
    """Compute percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _empty_baseline(track_id: str) -> dict:
    """Return an empty baseline structure."""
    return {
        "track_id": track_id,
        "metric_id": None,
        "track_mean": None,
        "track_p25": None,
        "track_p75": None,
        "sample_size": 0,
        "systems": [],
    }
