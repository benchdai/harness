"""NullAdapter — always returns empty string. Useful for proving failure traces."""

from typing import Any, Dict, List

from .base import BaseAdapter


class NullAdapter(BaseAdapter):
    """Adapter that stores nothing and always returns an empty string."""

    @property
    def name(self) -> str:
        return "null"

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        pass

    def recall(self, query: str) -> str:
        return ""
