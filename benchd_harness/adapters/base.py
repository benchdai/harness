"""Abstract base class for all memory system adapters."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """Every memory system adapter must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter identifier (e.g., 'mem0-local', 'echo')."""
        ...

    @property
    def version(self) -> Optional[str]:
        """Optional adapter version."""
        return None

    @abstractmethod
    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Ingest conversation turns into the memory system.

        Each turn is a dict with:
          - role: "user" | "assistant" | "system"
          - content: str
          - timestamp: Optional[str] (ISO datetime)
          - metadata: Optional[dict]
        """
        ...

    @abstractmethod
    def recall(self, query: str) -> str:
        """
        Query the memory system and return a plain string response.
        """
        ...

    def reset(self) -> None:
        """Reset the adapter state between benchmark questions. Override if needed."""
        pass

    def setup(self) -> None:
        """Called once before a benchmark run. Override for initialization."""
        pass

    def teardown(self) -> None:
        """Called once after a benchmark run. Override for cleanup."""
        pass
