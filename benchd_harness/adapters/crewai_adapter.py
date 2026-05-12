"""
CrewAI Memory adapter for Bench'd harness.

Uses CrewAI's long-term memory storage to ingest and recall conversation data.

Requires:
  pip install crewai crewai-tools

  export OPENAI_API_KEY=sk-...  (or OPENROUTER_API_KEY)
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class CrewAIMemoryAdapter(BaseAdapter):
    """Adapter for CrewAI's built-in memory system."""

    @property
    def name(self) -> str:
        return "crewai-memory"

    @property
    def version(self) -> Optional[str]:
        try:
            import crewai
            return getattr(crewai, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._memory = None
        self._llm = None

    def setup(self) -> None:
        try:
            from crewai.memory.long_term.long_term_memory import LongTermMemory
            from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
        except ImportError:
            raise RuntimeError(
                "crewai package not installed. Install with:\n"
                "  pip install crewai"
            )

        self._memory = LongTermMemory()
        self._LongTermMemoryItem = LongTermMemoryItem

    def reset(self) -> None:
        if self._memory is not None:
            try:
                self._memory.reset()
            except Exception:
                pass

    def teardown(self) -> None:
        self.reset()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for i, turn in enumerate(turns):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")

            item = self._LongTermMemoryItem(
                agent=role,
                task=f"conversation_turn_{i}",
                expected_output="",
                datetime=timestamp,
                quality=1.0,
                metadata={"role": role, "index": i},
                output=content,
            )
            self._memory.save(item)

    def recall(self, query: str) -> str:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._memory.search(query, latest_n=10)
            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, dict):
                        texts.append(r.get("output", r.get("text", str(r))))
                    else:
                        texts.append(str(r))
                return "\n".join(texts)
            return str(results)
        except Exception as e:
            return f"[recall error: {e}]"
