"""
Letta (formerly MemGPT) adapter for Bench'd harness.

Requires:
  pip install letta-client

  A running Letta server (default: http://localhost:8283).
  Start the server with:
    letta server
  Or via Docker:
    docker run -p 8283:8283 letta/letta:latest

  Optionally:
    export LETTA_BASE_URL=http://localhost:8283  (default)
    export LETTA_API_KEY=...                      (if using Letta Cloud)

Letta is a stateful agent framework. Each benchmark run creates a fresh agent.
Conversation turns are sent as user messages. Recall queries are sent as
messages and the assistant's response is captured.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class LettaAdapter(BaseAdapter):
    """
    Adapter for Letta — stateful LLM agent framework.

    Letta runs as a server; the adapter communicates via the Python client SDK.
    Each benchmark run creates a dedicated agent that is deleted on teardown.

    Prerequisites:
      1. Install: pip install letta-client
      2. Start the Letta server: letta server  (or Docker)
      3. Optionally set LETTA_BASE_URL (default http://localhost:8283)
    """

    @property
    def name(self) -> str:
        return "letta"

    @property
    def version(self) -> Optional[str]:
        try:
            import letta_client

            return getattr(letta_client, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self._base_url = base_url or os.environ.get(
            "LETTA_BASE_URL", "http://localhost:8283"
        )
        self._model = model
        self._client = None  # lazy init
        self._agent_id: Optional[str] = None

    def setup(self) -> None:
        """
        Initialize the Letta client and verify the server is reachable.
        Creates a fresh agent for this benchmark run.
        """
        try:
            from letta_client import Letta
        except ImportError:
            raise RuntimeError(
                "letta-client package not installed. Install it with:\n"
                "  pip install letta-client"
            )

        api_key = os.environ.get("LETTA_API_KEY")

        self._client = Letta(
            base_url=self._base_url,
            api_key=api_key,
        )

        # Verify the server is reachable by hitting the health endpoint.
        try:
            self._client.health()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Letta server at {self._base_url}.\n"
                f"Start it with: letta server\n"
                f"Or via Docker: docker run -p 8283:8283 letta/letta:latest\n"
                f"Error: {e}"
            )

        self._create_agent()

    def _create_agent(self) -> None:
        """Create a fresh Letta agent for this benchmark run."""
        agent_name = f"benchd-{uuid.uuid4().hex[:8]}"

        # Build create kwargs — configure model if using OpenRouter.
        create_kwargs: Dict[str, Any] = {
            "name": agent_name,
            "include_base_tools": True,
        }

        # If an OpenRouter key is available, configure the LLM to route
        # through OpenRouter.
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            create_kwargs["llm_config"] = {
                "model": self._model,
                "model_endpoint_type": "openai",
                "model_endpoint": "https://openrouter.ai/api/v1",
                "context_window": 128000,
            }
            # Letta may also need the key set as an env var for the server
            # to pick up; document this for users.

        agent = self._client.agents.create(**create_kwargs)
        self._agent_id = agent.id

    def _delete_agent(self) -> None:
        """Delete the current agent, if one exists."""
        if self._client is not None and self._agent_id is not None:
            try:
                self._client.agents.delete(self._agent_id)
            except Exception:
                pass
            self._agent_id = None

    def teardown(self) -> None:
        """Delete the benchmark agent and clean up."""
        self._delete_agent()

    def reset(self) -> None:
        """
        Reset state between benchmark questions.
        Deletes the current agent and creates a fresh one.
        """
        self._delete_agent()
        if self._client is not None:
            self._create_agent()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Send conversation turns to the Letta agent.

        Each turn is sent as a user message. Assistant turns are sent with a
        prefix indicating they are prior assistant responses, so the agent
        can incorporate them into its memory.
        """
        if self._client is None or self._agent_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role == "user":
                message_text = content
            elif role == "assistant":
                message_text = f"[Previous assistant response]: {content}"
            elif role == "system":
                message_text = f"[System context]: {content}"
            else:
                message_text = content

            try:
                self._client.agents.messages.create(
                    agent_id=self._agent_id,
                    messages=[{"role": "user", "content": message_text}],
                )
            except Exception as e:
                # Log but continue — partial ingest is better than failing
                # the entire run.
                pass

    def recall(self, query: str) -> str:
        """
        Send a recall query to the Letta agent and return its response.

        The agent processes the query using its built-in memory (core memory,
        archival memory, recall memory) and returns a natural language response.
        """
        if self._client is None or self._agent_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[{"role": "user", "content": query}],
            )
        except Exception as e:
            return f"[recall error: {e}]"

        # Extract assistant messages from the response.
        assistant_texts: list[str] = []
        if hasattr(response, "messages"):
            for msg in response.messages:
                # Check for AssistantMessage type
                msg_type = getattr(msg, "message_type", None)
                if msg_type == "assistant_message":
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and content:
                        assistant_texts.append(content)
                    elif isinstance(content, list):
                        # Content may be a list of content blocks
                        for block in content:
                            text = getattr(block, "text", None)
                            if text:
                                assistant_texts.append(text)

        if assistant_texts:
            return " ".join(assistant_texts)

        # Fallback: try to get any text from the response
        return str(response) if not assistant_texts else ""
