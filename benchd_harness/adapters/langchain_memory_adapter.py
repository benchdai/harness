"""
LangChain Memory adapter for Bench'd harness.

Uses LangChain's ConversationSummaryBufferMemory, which summarizes older
messages and keeps recent ones in a buffer. This actually extracts and
compresses information (via an LLM) rather than just buffering raw text.

Requires:
  pip install langchain langchain-openai langchain-community

  Either:
    export OPENAI_API_KEY=sk-...
  Or (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...

Limitations:
  - ConversationSummaryBufferMemory is designed for chat continuity, not
    semantic search. Recall works by dumping the memory buffer and using
    the LLM to answer the query from that context.
  - The summary quality depends on the backing LLM.
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LangChainMemoryAdapter(BaseAdapter):
    """
    Adapter for LangChain ConversationSummaryBufferMemory.

    Ingests conversation turns via save_context(), then on recall() loads
    the memory variables (summary + buffer) and uses the LLM to answer
    the query from the stored context.
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

    def __init__(self, model: str = "openai/gpt-4o-mini", max_token_limit: int = 2000):
        self._model_name = model
        self._max_token_limit = max_token_limit
        self._llm = None
        self._memory = None

    def setup(self) -> None:
        """Initialize ChatOpenAI and ConversationSummaryBufferMemory."""
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
                "  pip install langchain langchain-openai langchain-community"
            )

        try:
            from langchain_classic.memory import ConversationSummaryBufferMemory
        except ImportError:
            raise RuntimeError(
                "langchain-classic package not installed (provides memory classes). "
                "Install with: pip install langchain"
            )

        # Configure LLM — use OpenRouter if available, else direct OpenAI
        # Strip provider prefix for LangChain tokenizer compatibility
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

        self._memory = ConversationSummaryBufferMemory(
            llm=self._llm,
            max_token_limit=self._max_token_limit,
            return_messages=False,
            human_prefix="User",
            ai_prefix="Assistant",
        )

    def reset(self) -> None:
        """Clear memory between benchmark questions."""
        if self._memory is not None:
            self._memory.clear()

    def teardown(self) -> None:
        """Clean up."""
        if self._memory is not None:
            self._memory.clear()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into LangChain memory via save_context().

        save_context() takes (input_dict, output_dict) pairs, so we pair up
        user/assistant turns. System messages are prepended to the next user
        message. Unpaired turns are handled gracefully.
        """
        if self._memory is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Collect turns into (user_input, assistant_output) pairs
        pending_system = ""
        pending_user = None

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "system":
                # Accumulate system text to prepend to next user message
                pending_system += f"[System: {content}] "
            elif role == "user":
                # If there's a pending user without a paired assistant response,
                # save it with an empty assistant response
                if pending_user is not None:
                    self._memory.save_context(
                        {"input": pending_user},
                        {"output": ""},
                    )
                pending_user = pending_system + content
                pending_system = ""
            elif role == "assistant":
                if pending_user is not None:
                    self._memory.save_context(
                        {"input": pending_user},
                        {"output": content},
                    )
                    pending_user = None
                else:
                    # Assistant message without a preceding user message —
                    # save with empty input
                    self._memory.save_context(
                        {"input": pending_system or "(no user input)"},
                        {"output": content},
                    )
                    pending_system = ""

        # Handle trailing unpaired user message
        if pending_user is not None:
            self._memory.save_context(
                {"input": pending_user},
                {"output": ""},
            )

    def recall(self, query: str) -> str:
        """
        Load memory context and use the LLM to answer the query.

        LangChain's load_memory_variables returns the summary + recent buffer.
        We feed that as context to the LLM along with the query.
        """
        if self._memory is None or self._llm is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            mem_vars = self._memory.load_memory_variables({})
        except Exception as e:
            return f"[memory load error: {e}]"

        # The memory key is typically "history"
        context = ""
        for key, value in mem_vars.items():
            if isinstance(value, str):
                context += value
            elif isinstance(value, list):
                # If return_messages=True, we get message objects
                context += "\n".join(str(m) for m in value)

        if not context.strip():
            return ""

        # Use the LLM to answer the query from the stored context
        prompt = (
            "You are a helpful assistant. Based ONLY on the following conversation "
            "memory, answer the question. If the answer is not in the memory, say "
            "you don't know. Be concise and factual.\n\n"
            f"Conversation memory:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            response = self._llm.invoke(prompt)
            # response is an AIMessage; extract content
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            return f"[recall error: {e}]"
