"""
LoCoMo benchmark loader.

Downloads the LoCoMo-10 dataset from GitHub and converts
to Bench'd BenchmarkItem format.

Paper: https://arxiv.org/abs/2402.18397 (ACL 2024)
Repository: https://github.com/snap-research/locomo
"""

import json
import re
from pathlib import Path
from typing import List, Optional
import urllib.request

from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem

DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo"
    "/main/data/locomo10.json"
)
CACHE_DIR = Path.home() / ".cache" / "benchd" / "locomo"

# LoCoMo category integers → question type names
CATEGORY_NAME = {
    1: "single_hop",
    2: "temporal_reasoning",
    3: "multi_hop",
    4: "open_domain",
    # 5 = adversarial — excluded from evaluation
}

# Map LoCoMo question types to Bench'd dimensions
DIMENSION_MAP = {
    "single_hop": "recall",
    "temporal_reasoning": "temporal",
    "multi_hop": "reasoning",
    "open_domain": "reasoning",
}

# Scoring method by question type
SCORING_MAP = {
    "single_hop": "exact",
    "temporal_reasoning": "exact",
    "multi_hop": "llm",
    "open_domain": "llm",
}


class LoCoMoBenchmark(BaseBenchmark):
    """
    LoCoMo — long-term conversational memory benchmark.

    10 multi-session conversations with ~1500 QA pairs (excluding
    adversarial) testing single-hop recall, temporal reasoning,
    multi-hop reasoning, and open-domain comprehension.

    Args:
        data_path: Path to local locomo10.json. If None, downloads
            from GitHub.
        max_items: Limit number of items (for testing/quick runs). None = all.
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
        return "locomo-v1"

    @property
    def name(self) -> str:
        return "LoCoMo"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return (
            "Long-term conversational memory benchmark — 10 multi-session "
            "conversations with ~1500 QA pairs across recall, temporal, "
            "and reasoning dimensions (ACL 2024)"
        )

    def _ensure_data(self) -> Path:
        """Download dataset if not cached locally."""
        if self._data_path and Path(self._data_path).exists():
            return Path(self._data_path)

        cache_path = CACHE_DIR / "locomo10.json"
        if cache_path.exists():
            return cache_path

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LoCoMo dataset to {cache_path}...")
        urllib.request.urlretrieve(DATASET_URL, cache_path)
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"Downloaded ({size_mb:.1f} MB)")
        return cache_path

    @staticmethod
    def _extract_sessions(conversation: dict) -> List[dict]:
        """
        Extract ordered sessions from a conversation object.

        Returns a list of dicts, each with keys:
            session_index, date_time, turns
        where turns is a list of dialogue turn dicts.
        """
        sessions = []
        # Session keys follow the pattern session_1, session_2, ...
        session_nums = sorted(
            int(m.group(1))
            for key in conversation
            if (m := re.match(r"^session_(\d+)$", key))
        )
        for num in session_nums:
            turns = conversation.get(f"session_{num}", [])
            date_time = conversation.get(f"session_{num}_date_time", "")
            sessions.append({
                "session_index": num,
                "date_time": date_time,
                "turns": turns,
            })
        return sessions

    def load_items(self) -> List[BenchmarkItem]:
        """Load and convert LoCoMo instances to BenchmarkItems."""
        data_path = self._ensure_data()

        with open(data_path) as f:
            raw_data = json.load(f)

        # Flatten: one BenchmarkItem per QA pair across all conversations
        all_qa: List[dict] = []
        for conv_idx, entry in enumerate(raw_data):
            sample_id = entry.get("sample_id", f"conv-{conv_idx}")
            conversation = entry.get("conversation", {})
            sessions = self._extract_sessions(conversation)
            speaker_a = conversation.get("speaker_a", "")
            speaker_b = conversation.get("speaker_b", "")

            # Build ingestion turns for this conversation
            ingest_turns: List[dict] = []
            for session in sessions:
                for turn in session["turns"]:
                    ingest_turns.append({
                        "role": "user" if turn.get("speaker") == speaker_a else "assistant",
                        "content": turn.get("text", ""),
                        "metadata": {
                            "session_index": session["session_index"],
                            "session_date_time": session["date_time"],
                            "dia_id": turn.get("dia_id", ""),
                            "speaker": turn.get("speaker", ""),
                        },
                    })

            for qa in entry.get("qa", []):
                category = qa.get("category")
                # Skip adversarial questions (category 5)
                if category == 5:
                    continue
                question_type = CATEGORY_NAME.get(category, "unknown")
                all_qa.append({
                    "sample_id": sample_id,
                    "conv_idx": conv_idx,
                    "question_type": question_type,
                    "category": category,
                    "qa": qa,
                    "ingest_turns": ingest_turns,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "num_sessions": len(sessions),
                })

        if self._max_items and self._max_items < len(all_qa):
            # Stratified sample: proportional items from each question type
            from collections import defaultdict
            import random
            random.seed(42)

            by_type: dict[str, list] = defaultdict(list)
            for item in all_qa:
                by_type[item["question_type"]].append(item)

            sampled: list = []
            total = len(all_qa)
            for qtype, items_of_type in by_type.items():
                n = max(1, round(len(items_of_type) / total * self._max_items))
                sampled.extend(random.sample(items_of_type, min(n, len(items_of_type))))

            # Trim to exact max_items if we overshot due to rounding
            all_qa = sampled[: self._max_items]

        items: List[BenchmarkItem] = []
        for idx, entry in enumerate(all_qa):
            qa = entry["qa"]
            question_type = entry["question_type"]
            dimension = DIMENSION_MAP.get(question_type, "reasoning")
            scoring_method = SCORING_MAP.get(question_type, "exact")

            items.append(
                BenchmarkItem(
                    question_id=f"locomo-{entry['conv_idx']}-{idx}",
                    ingest_history=entry["ingest_turns"],
                    recall_query=qa["question"],
                    expected_answer=str(qa["answer"]),
                    scoring_method=scoring_method,
                    dimension=dimension,
                    max_score=1.0,
                    metadata={
                        "question_type": question_type,
                        "category": entry["category"],
                        "sample_id": entry["sample_id"],
                        "evidence": qa.get("evidence", []),
                        "speaker_a": entry["speaker_a"],
                        "speaker_b": entry["speaker_b"],
                        "num_sessions": entry["num_sessions"],
                    },
                )
            )

        return items
