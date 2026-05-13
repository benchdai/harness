"""Benchmark layer for Bench'd harness — pluggable benchmark definitions."""

from typing import Any

from .base import BaseBenchmark, BenchmarkItem
from .smoke import SmokeBenchmark
from .longmemeval import LongMemEvalBenchmark
from .locomo import LoCoMoBenchmark
from .reliability import ReliabilityBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkItem",
    "SmokeBenchmark",
    "LongMemEvalBenchmark",
    "LoCoMoBenchmark",
    "get_benchmark",
    "list_benchmarks",
]

_BUILTIN_BENCHMARKS = {
    "smoke-memory-v0": SmokeBenchmark,
    "longmemeval-v1": LongMemEvalBenchmark,
    "locomo-v1": LoCoMoBenchmark,
    "reliability-v1": ReliabilityBenchmark,
}


def list_benchmarks() -> "list[str]":
    """Return slugs of all registered benchmarks."""
    return sorted(_BUILTIN_BENCHMARKS)


def get_benchmark(slug: str, **kwargs: Any) -> BaseBenchmark:
    """
    Factory function that returns a benchmark instance by slug.

    Args:
        slug: Benchmark identifier (e.g., 'smoke-memory-v0').
        **kwargs: Optional keyword arguments forwarded to the benchmark
            constructor (e.g., ``max_items=50`` for quick test runs).

    Returns:
        An instantiated BaseBenchmark subclass.

    Raises:
        ValueError: If the benchmark slug is not recognized.
    """
    cls = _BUILTIN_BENCHMARKS.get(slug)
    if cls is None:
        available = ", ".join(sorted(_BUILTIN_BENCHMARKS))
        raise ValueError(
            f"Unknown benchmark {slug!r}. Available benchmarks: {available}"
        )
    return cls(**kwargs)
