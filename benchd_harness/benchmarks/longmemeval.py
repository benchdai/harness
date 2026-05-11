"""
LongMemEval benchmark loader.

Downloads the oracle dataset from Hugging Face and converts
to Bench'd BenchmarkItem format.

Paper: https://arxiv.org/abs/2407.01528
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import json
from pathlib import Path
from typing import List, Optional
import urllib.request

from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
    "/resolve/main/longmemeval_oracle.json"
)
CACHE_DIR = Path.home() / ".cache" / "benchd" / "longmemeval"

# Map LongMemEval question types to Bench'd dimensions
DIMENSION_MAP = {
    "single-session-user": "recall",
    "single-session-assistant": "recall",
    "single-session-preference": "recall",
    "temporal-reasoning": "temporal",
    "knowledge-update": "temporal",
    "multi-session": "reasoning",
}

# Scoring method by dimension
SCORING_MAP = {
    "recall": "exact",
    "temporal": "exact",
    "reasoning": "llm",
}


class LongMemEvalBenchmark(BaseBenchmark):
    """
    LongMemEval — long-term memory evaluation benchmark.

    500 questions testing recall, temporal reasoning, and multi-session synthesis.
    Uses the oracle variant (minimal sessions containing evidence).

    Args:
        data_path: Path to local longmemeval_oracle.json. If None, downloads
            from HuggingFace.
        max_items: Limit number of items (for testing/quick runs). None = all 500.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        max_items: Optional[int] = None,
    ):
        self._data_path = data_path
        self._max_items = max_items

    @property
    def slug(self) -> str:
        return "longmemeval-v1"

    @property
    def name(self) -> str:
        return "LongMemEval"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return (
            "Long-term memory evaluation across conversation sessions "
            "(oracle variant, 500 questions)"
        )

    def _ensure_data(self) -> Path:
        """Download dataset if not cached locally."""
        if self._data_path and Path(self._data_path).exists():
            return Path(self._data_path)

        cache_path = CACHE_DIR / "longmemeval_oracle.json"
        if cache_path.exists():
            return cache_path

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LongMemEval dataset to {cache_path}...")
        urllib.request.urlretrieve(DATASET_URL, cache_path)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"Downloaded ({size_mb:.1f} MB)")
        return cache_path

    def load_items(self) -> List[BenchmarkItem]:
        """Load and convert LongMemEval instances to BenchmarkItems."""
        data_path = self._ensure_data()

        with open(data_path) as f:
            raw_data = json.load(f)

        if self._max_items and self._max_items < len(raw_data):
            # Stratified sample: take proportional items from each question type
            # so we cover all dimensions, not just whatever's first in the file
            from collections import defaultdict
            import random
            random.seed(42)  # deterministic sampling

            by_type: dict[str, list] = defaultdict(list)
            for item in raw_data:
                by_type[item.get("question_type", "unknown")].append(item)

            sampled: list = []
            total = len(raw_data)
            for qtype, items_of_type in by_type.items():
                # Proportional allocation, at least 1 per type
                n = max(1, round(len(items_of_type) / total * self._max_items))
                sampled.extend(random.sample(items_of_type, min(n, len(items_of_type))))

            # Trim to exact max_items if we overshot due to rounding
            raw_data = sampled[: self._max_items]

        items: List[BenchmarkItem] = []
        for instance in raw_data:
            question_type = instance.get("question_type", "unknown")
            dimension = DIMENSION_MAP.get(question_type, "reasoning")
            scoring_method = SCORING_MAP.get(dimension, "exact")

            # Flatten haystack_sessions into a single list of turns.
            # Each session is a list of turns; we attach session metadata
            # and answer-evidence flags to each turn.
            ingest_turns: List[dict] = []
            for session_idx, session in enumerate(
                instance.get("haystack_sessions", [])
            ):
                for turn in session:
                    ingest_turns.append(
                        {
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                            "metadata": {
                                "session_index": session_idx,
                                "has_answer": turn.get("has_answer", False),
                            },
                        }
                    )

            question_date = instance.get("question_date", "")

            items.append(
                BenchmarkItem(
                    question_id=instance["question_id"],
                    ingest_history=ingest_turns,
                    recall_query=instance["question"],
                    expected_answer=instance["answer"],
                    scoring_method=scoring_method,
                    dimension=dimension,
                    max_score=1.0,
                    metadata={
                        "question_type": question_type,
                        "question_date": question_date,
                        "num_sessions": len(
                            instance.get("haystack_sessions", [])
                        ),
                        "answer_session_ids": instance.get(
                            "answer_session_ids", []
                        ),
                    },
                )
            )

        return items
