"""EchoAdapter — deterministic test adapter with simple keyword matching."""

from typing import Any, Dict, List

from .base import BaseAdapter


class EchoAdapter(BaseAdapter):
    """
    A deterministic test adapter that stores ingested turns in memory
    and retrieves the most relevant one via simple keyword matching.

    Designed to pass smoke-memory-v0 where answers are literally present
    in ingested conversation content.
    """

    def __init__(self) -> None:
        self._turns: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "echo"

    @property
    def version(self) -> str:
        return "0.1.0"

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        self._turns.extend(turns)

    def recall(self, query: str) -> str:
        if not self._turns:
            return ""

        query_words = _tokenize(query)
        if not query_words:
            return ""

        best_turn: Dict[str, Any] = self._turns[0]
        best_score: int = -1

        for turn in self._turns:
            content = turn.get("content", "")
            content_words = _tokenize(content)
            score = len(query_words & content_words)
            if score > best_score:
                best_score = score
                best_turn = turn

        return best_turn.get("content", "")

    def reset(self) -> None:
        self._turns.clear()


def _tokenize(text: str) -> set:
    """Lowercase a string and split into a set of alphanumeric words."""
    return {w for w in text.lower().split() if w.isalnum()}
