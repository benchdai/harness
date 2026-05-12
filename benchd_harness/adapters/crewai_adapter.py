"""
CrewAI Memory adapter for Bench'd harness.

Uses CrewAI's unified Memory class with remember/recall/reset.

Requires:
  pip install crewai

  export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class CrewAIMemoryAdapter(BaseAdapter):
    """Adapter for CrewAI's unified memory system."""

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

    def setup(self) -> None:
        try:
            from crewai.memory import Memory
        except ImportError:
            raise RuntimeError(
                "crewai package not installed. Install with:\n"
                "  pip install crewai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("CrewAI Memory requires OPENAI_API_KEY for embeddings.")

        self._memory = Memory()

    def reset(self) -> None:
        if self._memory is not None:
            try:
                self._memory.reset()
            except Exception:
                pass

    def teardown(self) -> None:
        if self._memory is not None:
            try:
                self._memory.close()
            except Exception:
                pass
        self._memory = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            self._memory.remember(f"[{role}]: {content}")

    def recall(self, query: str) -> str:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._memory.recall(query)
            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, str):
                        texts.append(r)
                    elif hasattr(r, "content"):
                        texts.append(str(r.content))
                    elif hasattr(r, "text"):
                        texts.append(str(r.text))
                    elif isinstance(r, dict):
                        texts.append(r.get("content", r.get("text", str(r))))
                    else:
                        texts.append(str(r))
                return "\n".join(texts)
            return str(results) if results else ""
        except Exception as e:
            return f"[recall error: {e}]"
