"""
LangChain Memory adapter for Bench'd harness.

Uses LangChain's ChatMessageHistory to store conversation turns, then
recalls by loading the full history (with smart truncation) and using
the LLM to answer queries from that context.

This replaces the deprecated ConversationSummaryBufferMemory approach
which would crash on long conversations due to accumulated summarization
LLM calls.

Requires:
  pip install langchain langchain-openai langchain-core

  Either:
    export OPENAI_API_KEY=sk-...
  Or (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LangChainMemoryAdapter(BaseAdapter):
    """
    Adapter using LangChain's ChatMessageHistory + LLM recall.

    Ingests conversation turns into an in-memory message store, then
    on recall() loads and truncates the history to fit context, and
    uses the LLM to answer the query.
    """

    @property
    def name(self) -> str:
        return "langchain-memory"

    @property
    def version(self) -> Optional[str]:
        try:
            import langchain_core
            return getattr(langchain_core, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self, model: str = "openai/gpt-4o-mini", max_context_chars: int = 60000):
        self._model_name = model
        self._max_context_chars = max_context_chars
        self._llm = None
        self._messages: list = []

    def setup(self) -> None:
        """Initialize ChatOpenAI."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "LangChain Memory adapter requires an LLM API key. Set one of:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError(
                "langchain-openai package not installed. Install with:\n"
                "  pip install langchain-openai langchain-core"
            )

        # Configure LLM
        model_for_langchain = self._model_name
        if "/" in model_for_langchain:
            model_for_langchain = model_for_langchain.split("/", 1)[1]

        llm_kwargs: dict[str, Any] = {
            "model": model_for_langchain,
            "temperature": 0,
        }

        if openrouter_key:
            llm_kwargs["api_key"] = openrouter_key
            llm_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        elif openai_key:
            llm_kwargs["api_key"] = openai_key

        self._llm = ChatOpenAI(**llm_kwargs)
        self._messages = []

    def reset(self) -> None:
        """Clear memory between benchmark questions."""
        self._messages = []

    def teardown(self) -> None:
        """Clean up."""
        self._messages = []

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Store conversation turns as LangChain message objects.
        """
        if self._llm is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "system":
                self._messages.append(SystemMessage(content=content))
            elif role == "user":
                self._messages.append(HumanMessage(content=content))
            elif role == "assistant":
                self._messages.append(AIMessage(content=content))

    def recall(self, query: str) -> str:
        """
        Build context from stored messages and use the LLM to answer.

        Truncates from the beginning if the conversation is too long,
        keeping the most recent messages (which tend to be most relevant
        for memory recall benchmarks).
        """
        if self._llm is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        if not self._messages:
            return ""

        # Build a text representation of the conversation history
        # with smart truncation
        lines = []
        total_chars = 0

        # Walk messages in reverse to prioritize recent context
        for msg in reversed(self._messages):
            role_label = "User"
            if hasattr(msg, "type"):
                if msg.type == "ai":
                    role_label = "Assistant"
                elif msg.type == "system":
                    role_label = "System"
                elif msg.type == "human":
                    role_label = "User"

            line = f"{role_label}: {msg.content}"
            total_chars += len(line)

            if total_chars > self._max_context_chars:
                lines.append("[... earlier conversation truncated ...]")
                break

            lines.append(line)

        # Reverse back to chronological order
        lines.reverse()
        context = "\n".join(lines)

        # Use LangChain's LLM to answer from context
        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = (
            "You are a helpful assistant. Based ONLY on the following conversation "
            "memory, answer the question. If the answer is not in the memory, say "
            "you don't know. Be concise and factual."
        )

        user_prompt = (
            f"Conversation memory:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            return f"[recall error: {e}]"
