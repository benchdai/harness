"""
LlamaIndex Memory adapter for Bench'd harness.

Uses LlamaIndex's ChatMemoryBuffer, which stores conversation messages
and manages a token-limited window of chat history.

Requires:
  pip install llama-index llama-index-llms-openai

  Either:
    export OPENAI_API_KEY=sk-...
  Or (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...

Limitations:
  - ChatMemoryBuffer is a sliding-window buffer over chat messages, not a
    semantic memory store. It does not extract facts or build a knowledge graph.
  - Recall works by dumping the buffered messages and using the LLM to
    answer the query from that context.
  - Token limit determines how many messages are retained in the window.
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LlamaIndexMemoryAdapter(BaseAdapter):
    """
    Adapter for LlamaIndex ChatMemoryBuffer.

    Ingests conversation turns via put(), then on recall() retrieves
    all buffered messages and uses the LLM to answer the query.
    """

    @property
    def name(self) -> str:
        return "llamaindex-memory"

    @property
    def version(self) -> Optional[str]:
        try:
            import llama_index.core
            return getattr(llama_index.core, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self, model: str = "openai/gpt-4o-mini", token_limit: int = 8000):
        self._model_name = model
        self._token_limit = token_limit
        self._llm = None
        self._memory = None

    def setup(self) -> None:
        """Initialize OpenAI LLM and ChatMemoryBuffer."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "LlamaIndex Memory adapter requires an LLM API key. Set one of:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )

        try:
            from llama_index.core.memory import ChatMemoryBuffer
        except ImportError:
            raise RuntimeError(
                "llama-index-core package not installed. Install with:\n"
                "  pip install llama-index llama-index-llms-openai"
            )

        # Use OpenAI-like LLM for OpenRouter, or direct OpenAI
        if openrouter_key:
            try:
                from llama_index.llms.openai_like import OpenAILike
            except ImportError:
                raise RuntimeError(
                    "llama-index-llms-openai-like package not installed. "
                    "Install with: pip install llama-index-llms-openai-like"
                )
            self._llm = OpenAILike(
                model=self._model_name,
                api_key=openrouter_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0,
                is_chat_model=True,
            )
        else:
            try:
                from llama_index.llms.openai import OpenAI as LlamaOpenAI
            except ImportError:
                raise RuntimeError(
                    "llama-index-llms-openai package not installed. "
                    "Install with: pip install llama-index-llms-openai"
                )
            self._llm = LlamaOpenAI(
                model=self._model_name,
                api_key=openai_key,
                temperature=0,
            )

        self._memory = ChatMemoryBuffer.from_defaults(
            token_limit=self._token_limit,
        )

    def reset(self) -> None:
        """Clear memory between benchmark questions."""
        if self._memory is not None:
            self._memory.reset()

    def teardown(self) -> None:
        """Clean up."""
        if self._memory is not None:
            self._memory.reset()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into LlamaIndex ChatMemoryBuffer via put().

        Each turn is converted to a ChatMessage and added to the buffer.
        """
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole
        except ImportError:
            from llama_index.core.llms import ChatMessage, MessageRole

        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }

        for turn in turns:
            role_str = turn.get("role", "user")
            content = turn.get("content", "")
            role = role_map.get(role_str, MessageRole.USER)

            msg = ChatMessage(role=role, content=content)
            self._memory.put(msg)

    def recall(self, query: str) -> str:
        """
        Retrieve buffered messages and use the LLM to answer the query.

        ChatMemoryBuffer.get() returns messages within the token limit.
        We format them as context and ask the LLM to answer.
        """
        if self._memory is None or self._llm is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            messages = self._memory.get_all()
        except Exception as e:
            return f"[memory retrieval error: {e}]"

        if not messages:
            return ""

        # Format messages into a context string
        context_lines = []
        for msg in messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", str(msg))
            # role may be a MessageRole enum
            role_str = role.value if hasattr(role, "value") else str(role)
            context_lines.append(f"{role_str.capitalize()}: {content}")

        context = "\n".join(context_lines)

        # Use the LLM to answer the query from the stored context
        try:
            from llama_index.core.base.llms.types import ChatMessage, MessageRole
        except ImportError:
            from llama_index.core.llms import ChatMessage, MessageRole

        prompt_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a helpful assistant. Based ONLY on the following "
                    "conversation memory, answer the question. If the answer is "
                    "not in the memory, say you don't know. Be concise and factual."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    f"Conversation memory:\n{context}\n\n"
                    f"Question: {query}"
                ),
            ),
        ]

        try:
            response = self._llm.chat(prompt_messages)
            # response is a ChatResponse; extract the message content
            if hasattr(response, "message"):
                return str(response.message.content)
            return str(response)
        except Exception as e:
            return f"[recall error: {e}]"
