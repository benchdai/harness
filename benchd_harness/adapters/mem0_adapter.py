"""
Mem0 adapter for Bench'd harness.

Requires:
  pip install mem0ai

  Either:
    export OPENAI_API_KEY=sk-...
  Or (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...

Mem0 has native OpenRouter support for the LLM layer.
Embeddings route through OpenRouter when OPENAI_API_KEY is also set.
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class Mem0Adapter(BaseAdapter):
    """
    Adapter for Mem0 (mem0.ai) — open-source memory layer for LLM apps.

    Mem0 uses OpenAI for memory extraction and embeddings.
    Set OPENAI_API_KEY environment variable before use.
    """

    @property
    def name(self) -> str:
        return "mem0-local"

    @property
    def version(self) -> Optional[str]:
        try:
            import mem0

            return getattr(mem0, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self, user_id: str = "benchd_user"):
        self._user_id = user_id
        self._memory = None  # lazy init

    def setup(self) -> None:
        """Initialize Mem0. Supports OpenAI or OpenRouter."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "Mem0 requires an LLM API key. Set one of:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )

        try:
            from mem0 import Memory
        except ImportError:
            raise RuntimeError(
                "mem0ai package not installed. Install it with: "
                "pip install mem0ai"
            )

        # If using OpenRouter, also set OPENAI_API_KEY for the embedder
        # (embedder doesn't auto-detect OpenRouter) and point it at OpenRouter
        if openrouter_key and not openai_key:
            os.environ["OPENAI_API_KEY"] = openrouter_key
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "openai/gpt-4o-mini",
                },
            },
        }

        # Use OpenRouter base URL for LLM if available
        if openrouter_key:
            config["llm"]["config"]["openrouter_base_url"] = "https://openrouter.ai/api/v1"

        self._memory = Memory.from_config(config)

    def teardown(self) -> None:
        """Clean up Mem0 state."""
        if self._memory is not None:
            try:
                self._memory.delete_all(user_id=self._user_id)
            except Exception:
                pass

    def reset(self) -> None:
        """Clear all memories between benchmark questions."""
        if self._memory is not None:
            try:
                self._memory.delete_all(user_id=self._user_id)
            except Exception:
                pass

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into Mem0.

        Converts our turn format to Mem0's message format and calls add().
        Mem0 will extract facts/memories from the conversation using its LLM.
        """
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        messages = []
        for turn in turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })

        if messages:
            self._memory.add(messages, user_id=self._user_id)

    def recall(self, query: str) -> str:
        """
        Query Mem0's memory and return the most relevant results as a string.

        Mem0.search returns a list of memory dicts. We concatenate the
        'memory' field from the top results into a single response string.
        """
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._memory.search(
                query,
                filters={"user_id": self._user_id},
                top_k=10,
            )
        except Exception as e:
            # Don't let search errors crash the run — return empty
            # so the scorer marks it as a failure with context
            return f"[search error: {e}]"

        if not results:
            return ""

        # Results can be a list of dicts, each with a 'memory' key,
        # or sometimes a dict with a 'results' key wrapping the list.
        if isinstance(results, dict) and "results" in results:
            results = results["results"]

        if not isinstance(results, list):
            return str(results)

        memories: list[str] = []
        for r in results:
            if isinstance(r, dict):
                mem_text = r.get("memory", r.get("text", ""))
                if mem_text:
                    memories.append(str(mem_text))
            elif isinstance(r, str):
                memories.append(r)

        return ". ".join(memories) if memories else ""
