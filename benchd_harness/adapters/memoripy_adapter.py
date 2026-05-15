"""
Memoripy adapter for Bench'd harness.

Memoripy is an AI memory layer with short- and long-term storage,
semantic clustering, and optional persistent storage.

Requires:
  pip install memoripy

  export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class MemoripyAdapter(BaseAdapter):
    """Adapter for Memoripy memory layer."""

    @property
    def name(self) -> str:
        return "memoripy"

    @property
    def version(self) -> Optional[str]:
        try:
            import memoripy
            return getattr(memoripy, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._memory = None

    def setup(self) -> None:
        try:
            from memoripy import MemoryManager, JSONStorage
        except ImportError:
            raise RuntimeError(
                "memoripy not installed. Install with:\n"
                "  pip install memoripy"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Memoripy requires OPENAI_API_KEY for embeddings.")

        self._memory = MemoryManager(
            api_key=api_key,
            chat_model="gpt-4o-mini",
            chat_model_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_model_provider="openai",
            storage=JSONStorage("/tmp/benchd_memoripy.json"),
        )

    def reset(self) -> None:
        if self._memory is not None:
            try:
                # Reinitialize with fresh storage
                from memoripy import MemoryManager, JSONStorage
                import os as _os
                try:
                    _os.remove("/tmp/benchd_memoripy.json")
                except FileNotFoundError:
                    pass
                api_key = _os.environ.get("OPENAI_API_KEY", "")
                self._memory = MemoryManager(
                    api_key=api_key,
                    chat_model="gpt-4o-mini",
                    chat_model_provider="openai",
                    embedding_model="text-embedding-3-small",
                    embedding_model_provider="openai",
                    storage=JSONStorage("/tmp/benchd_memoripy.json"),
                )
            except Exception:
                pass

    def teardown(self) -> None:
        import os as _os
        try:
            _os.remove("/tmp/benchd_memoripy.json")
        except Exception:
            pass
        self._memory = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            try:
                self._memory.add_interaction(
                    prompt=f"[{role}]: {content}",
                    output=content if role == "assistant" else "",
                )
            except Exception:
                # Some versions use different API
                try:
                    self._memory.add_memory(f"[{role}]: {content}")
                except Exception:
                    pass

    def recall(self, query: str) -> str:
        if self._memory is None:
            raise RuntimeError("Adapter not initialized.")

        try:
            results = self._memory.retrieve_relevant_interactions(query, top_k=5)
            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, dict):
                        texts.append(r.get("prompt", r.get("text", str(r))))
                    elif isinstance(r, str):
                        texts.append(r)
                    elif hasattr(r, "prompt"):
                        texts.append(str(r.prompt))
                    else:
                        texts.append(str(r))
                return "\n".join(texts)
            return str(results) if results else ""
        except Exception as e:
            return f"[recall error: {e}]"
