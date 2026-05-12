"""
LangMem adapter for Bench'd harness.

Uses LangGraph's InMemoryStore with embeddings for semantic memory storage
and retrieval. This is the recommended in-process approach.

Requires:
  pip install langmem langgraph

  export OPENAI_API_KEY=sk-...
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LangMemAdapter(BaseAdapter):
    """Adapter for LangChain's LangMem SDK using InMemoryStore."""

    @property
    def name(self) -> str:
        return "langmem"

    @property
    def version(self) -> Optional[str]:
        try:
            import langmem
            return getattr(langmem, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._store = None
        self._namespace = ("memories",)

    def setup(self) -> None:
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError:
            raise RuntimeError(
                "langmem/langgraph not installed. Install with:\n"
                "  pip install langmem langgraph"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LangMem adapter requires OPENAI_API_KEY for embeddings.")

        self._store = InMemoryStore(
            index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        )

    def reset(self) -> None:
        if self._store is not None:
            try:
                from langgraph.store.memory import InMemoryStore
                self._store = InMemoryStore(
                    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
                )
            except Exception:
                pass

    def teardown(self) -> None:
        self._store = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._store is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")

            self._store.put(
                self._namespace,
                str(uuid.uuid4()),
                {
                    "content": f"[{role}]: {content}",
                    "role": role,
                    "timestamp": timestamp,
                },
            )

    def recall(self, query: str) -> str:
        if self._store is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._store.search(self._namespace, query=query, limit=10)
            texts = []
            for item in results:
                val = item.value
                if isinstance(val, dict):
                    texts.append(val.get("content", str(val)))
                else:
                    texts.append(str(val))
            return "\n".join(texts) if texts else ""
        except Exception as e:
            return f"[recall error: {e}]"
