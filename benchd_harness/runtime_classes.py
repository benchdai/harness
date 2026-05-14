"""
Bench'd Runtime Classes — standardized compute environments for fair benchmarking.

Every official run records its runtime class in the manifest so results
are only comparable within the same class.

Classes:
  S1 — Standard CPU (2 vCPU, 8GB RAM, no GPU, 30min timeout)
  S2 — Large CPU (4 vCPU, 16GB RAM, no GPU, 60min timeout)
  G1 — GPU-enabled (4 vCPU, 16GB RAM, 1x T4, 60min timeout)
  H1 — Hosted endpoint (no compute limit, measures vendor's infra)
  L1 — Local development (uncontrolled, not for official results)
"""

from dataclasses import dataclass


@dataclass
class RuntimeClass:
    """Defines compute constraints for a benchmark run."""
    name: str
    vcpu: int
    ram_gb: int
    gpu: str | None
    timeout_minutes: int
    description: str
    official: bool  # Can produce official leaderboard results


RUNTIME_CLASSES = {
    "S1": RuntimeClass(
        name="S1",
        vcpu=2,
        ram_gb=8,
        gpu=None,
        timeout_minutes=30,
        description="Standard CPU — default for OSS benchmarks",
        official=True,
    ),
    "S2": RuntimeClass(
        name="S2",
        vcpu=4,
        ram_gb=16,
        gpu=None,
        timeout_minutes=60,
        description="Large CPU — for memory-intensive systems",
        official=True,
    ),
    "G1": RuntimeClass(
        name="G1",
        vcpu=4,
        ram_gb=16,
        gpu="T4",
        timeout_minutes=60,
        description="GPU-enabled — for embedding-heavy systems",
        official=True,
    ),
    "H1": RuntimeClass(
        name="H1",
        vcpu=0,
        ram_gb=0,
        gpu=None,
        timeout_minutes=120,
        description="Hosted endpoint — vendor's infrastructure",
        official=True,
    ),
    "L1": RuntimeClass(
        name="L1",
        vcpu=0,
        ram_gb=0,
        gpu=None,
        timeout_minutes=0,
        description="Local development — uncontrolled, not for official results",
        official=False,
    ),
}


def get_runtime_class(name: str) -> RuntimeClass:
    """Get a runtime class by name."""
    rc = RUNTIME_CLASSES.get(name)
    if rc is None:
        available = ", ".join(sorted(RUNTIME_CLASSES))
        raise ValueError(f"Unknown runtime class {name!r}. Available: {available}")
    return rc


def get_current_runtime_class() -> str:
    """
    Detect the current runtime environment.

    Returns the best-matching runtime class name.
    Currently always returns L1 (local development).
    Future: detect Modal, Docker, CI environments.
    """
    import os

    if os.environ.get("MODAL_ENVIRONMENT"):
        return "S1"
    if os.environ.get("GITHUB_ACTIONS"):
        return "S1"
    if os.environ.get("BENCHD_RUNTIME_CLASS"):
        return os.environ["BENCHD_RUNTIME_CLASS"]

    return "L1"
