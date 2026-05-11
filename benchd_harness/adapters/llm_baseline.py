"""
LLM Baseline adapter — tests raw context window against benchmarks.

No memory system. Just feeds the full conversation history into an LLM
and asks it to answer the query directly. This is the baseline that
every memory system must beat to justify existing.

Uses OpenRouter (or any OpenAI-compatible API).

Requires:
  export OPENROUTER_API_KEY=sk-or-...
  or
  export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LLMBaselineAdapter(BaseAdapter):
    """
    Baseline: feed full conversation into LLM context, ask question directly.
    No memory extraction, no vector store, no retrieval — just raw context.
    """

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self._model = model
        self._turns: List[Dict[str, Any]] = []
        self._client: Any = None

    @property
    def name(self) -> str:
        return "llm-baseline"

    @property
    def version(self) -> Optional[str]:
        return self._model

    def setup(self) -> None:
        """Initialize OpenAI client."""
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "LLM baseline requires an API key. "
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package required. pip install openai"
            )

        base_url = (
            "https://openrouter.ai/api/v1"
            if os.environ.get("OPENROUTER_API_KEY")
            else None
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def teardown(self) -> None:
        self._client = None

    def reset(self) -> None:
        self._turns = []

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """Just store the turns in memory. No processing."""
        self._turns.extend(turns)

    def recall(self, query: str) -> str:
        """
        Feed ALL ingested turns as context to the LLM, then ask the query.
        Returns the LLM's direct answer.
        """
        if self._client is None:
            raise RuntimeError(
                "Adapter not initialized. Call setup() first."
            )

        # Build the conversation context
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are answering questions based on the conversation "
                    "history provided below. Answer concisely and directly "
                    "using only information from the conversation. "
                    "If the answer is not in the conversation, say "
                    "'Information not found in conversation history.'"
                ),
            }
        ]

        # Format all turns into a single user message with the full history
        history_lines: List[str] = []
        for turn in self._turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_lines.append(f"[{role}]: {content}")

        history_text = "\n\n".join(history_lines)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Here is the conversation history:\n\n"
                    f"{history_text}\n\n"
                    f"Based on this conversation history, "
                    f"answer this question:\n{query}"
                ),
            }
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
                max_tokens=256,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM ERROR] {e}"
