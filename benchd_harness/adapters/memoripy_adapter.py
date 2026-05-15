"""
Memoripy adapter for Bench'd harness.

Requires:
  pip install memoripy

  export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class MemoripyAdapter(BaseAdapter):

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
            from memoripy.implemented_models import OpenAIChatModel, OpenAIEmbeddingModel
        except ImportError:
            raise RuntimeError("memoripy not installed. pip install memoripy")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Memoripy requires OPENAI_API_KEY.")

        chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-4o-mini")
        embed_model = OpenAIEmbeddingModel(api_key=api_key, model_name="text-embedding-3-small")

        # Clean storage
        import pathlib
        pathlib.Path("/tmp/benchd_memoripy.json").unlink(missing_ok=True)

        self._memory = MemoryManager(
            chat_model=chat_model,
            embedding_model=embed_model,
            storage=JSONStorage("/tmp/benchd_memoripy.json"),
        )

    def reset(self) -> None:
        if self._memory:
            try:
                from memoripy import MemoryManager, JSONStorage
                from memoripy.implemented_models import OpenAIChatModel, OpenAIEmbeddingModel
                import pathlib
                pathlib.Path("/tmp/benchd_memoripy.json").unlink(missing_ok=True)
                api_key = os.environ.get("OPENAI_API_KEY", "")
                chat_model = OpenAIChatModel(api_key=api_key, model_name="gpt-4o-mini")
                embed_model = OpenAIEmbeddingModel(api_key=api_key, model_name="text-embedding-3-small")
                self._memory = MemoryManager(
                    chat_model=chat_model,
                    embedding_model=embed_model,
                    storage=JSONStorage("/tmp/benchd_memoripy.json"),
                )
            except Exception:
                pass

    def teardown(self) -> None:
        import pathlib
        pathlib.Path("/tmp/benchd_memoripy.json").unlink(missing_ok=True)
        self._memory = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if not self._memory:
            raise RuntimeError("Not initialized.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            try:
                self._memory.add_interaction(
                    prompt=f"[{role}]: {content}",
                    output=content if role == "assistant" else "",
                )
            except Exception:
                pass

    def recall(self, query: str) -> str:
        if not self._memory:
            raise RuntimeError("Not initialized.")

        try:
            results = self._memory.retrieve_relevant_interactions(query, top_k=5)
            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, dict):
                        texts.append(r.get("prompt", r.get("text", str(r))))
                    elif hasattr(r, "prompt"):
                        texts.append(str(r.prompt))
                    else:
                        texts.append(str(r))
                return "\n".join(texts)
            return str(results) if results else ""
        except Exception as e:
            return f"[recall error: {e}]"
