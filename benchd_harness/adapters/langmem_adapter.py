"""
LangMem adapter for Bench'd harness.

Uses LangMem's memory management SDK for storing and retrieving
conversation memories with semantic search.

Requires:
  pip install langmem

  export OPENAI_API_KEY=sk-...  (or OPENROUTER_API_KEY)
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LangMemAdapter(BaseAdapter):
    """Adapter for LangChain's LangMem SDK."""

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

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self._model_name = model
        self._client = None
        self._thread_id = "benchd-eval"

    def setup(self) -> None:
        try:
            from langmem import Client
        except ImportError:
            raise RuntimeError(
                "langmem package not installed. Install with:\n"
                "  pip install langmem"
            )

        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "LangMem adapter requires an API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
            )

        self._client = Client()

    def reset(self) -> None:
        # Create a new thread for each question
        import uuid
        self._thread_id = f"benchd-{uuid.uuid4().hex[:8]}"

    def teardown(self) -> None:
        self._client = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        messages = []
        for turn in turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })

        try:
            # LangMem extracts memories from conversation messages
            self._client.add_messages(
                thread_id=self._thread_id,
                messages=messages,
            )
        except Exception as e:
            # If add_messages isn't available, try alternative API
            try:
                for msg in messages:
                    self._client.add_memory(
                        content=msg["content"],
                        metadata={"role": msg["role"], "thread": self._thread_id},
                    )
            except Exception:
                raise RuntimeError(f"Failed to ingest: {e}")

    def recall(self, query: str) -> str:
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._client.search_memories(
                query=query,
                thread_id=self._thread_id,
            )
            if isinstance(results, list):
                texts = [
                    r.get("content", r.get("text", str(r)))
                    if isinstance(r, dict) else str(r)
                    for r in results
                ]
                return "\n".join(texts)
            return str(results)
        except Exception as e:
            return f"[recall error: {e}]"
