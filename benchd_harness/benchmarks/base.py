"""Abstract base class and data structures for benchmarks."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class BenchmarkItem:
    """A single benchmark question/test case."""

    question_id: str
    ingest_history: List[Dict[str, Any]]  # conversation turns to feed the memory system
    recall_query: str  # the question to ask after ingestion
    expected_answer: str  # correct answer
    scoring_method: str  # "exact" | "regex" | "llm"
    dimension: str  # "recall" | "temporal" | "reasoning"
    max_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Every benchmark must implement this interface."""

    @property
    @abstractmethod
    def slug(self) -> str:
        """Unique identifier for this benchmark (e.g., 'smoke-memory-v0')."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semver version string."""
        ...

    @property
    def description(self) -> str:
        """Optional longer description."""
        return ""

    @abstractmethod
    def load_items(self) -> List[BenchmarkItem]:
        """Load all benchmark items."""
        ...

    def get_dimensions(self) -> List[str]:
        """Return the dimensions this benchmark covers."""
        return ["recall", "temporal", "reasoning"]
