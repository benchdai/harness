"""
Julep adapter for Bench'd harness.

Connects to a Julep server to use its session memory for conversation
storage and retrieval.

Requires:
  pip install julep

  export JULEP_API_KEY=...
  export JULEP_API_URL=...  (optional, defaults to cloud)
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class JulepAdapter(BaseAdapter):
    """Adapter for Julep's session memory."""

    @property
    def name(self) -> str:
        return "julep"

    @property
    def version(self) -> Optional[str]:
        try:
            import julep
            return getattr(julep, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._client = None
        self._agent_id = None
        self._session_id = None

    def setup(self) -> None:
        try:
            from julep import Julep
        except ImportError:
            raise RuntimeError(
                "julep package not installed. Install with:\n"
                "  pip install julep"
            )

        api_key = os.environ.get("JULEP_API_KEY")
        if not api_key:
            raise RuntimeError("Julep requires JULEP_API_KEY.")

        base_url = os.environ.get("JULEP_API_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self._client = Julep(**kwargs)

        # Create an agent for benchmarking
        agent = self._client.agents.create(
            name="benchd-evaluator",
            about="Bench'd benchmark evaluation agent",
        )
        self._agent_id = agent.id

        # Create a session
        session = self._client.sessions.create(agent=self._agent_id)
        self._session_id = session.id

    def reset(self) -> None:
        if self._client and self._agent_id:
            try:
                session = self._client.sessions.create(agent=self._agent_id)
                self._session_id = session.id
            except Exception:
                pass

    def teardown(self) -> None:
        self._client = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._client is None or self._session_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            messages.append({"role": role, "content": content})

        try:
            # Send messages to session to build memory
            for msg in messages:
                self._client.sessions.chat(
                    session_id=self._session_id,
                    messages=[msg],
                )
        except Exception as e:
            raise RuntimeError(f"Julep ingest failed: {e}")

    def recall(self, query: str) -> str:
        if self._client is None or self._session_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            response = self._client.sessions.chat(
                session_id=self._session_id,
                messages=[{"role": "user", "content": query}],
            )
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            return str(response)
        except Exception as e:
            return f"[recall error: {e}]"
