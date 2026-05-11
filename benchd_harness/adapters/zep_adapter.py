"""
Zep adapter for Bench'd harness.

Requires:
  pip install zep-python

  Either Zep Cloud or a self-hosted Zep server:

  Zep Cloud:
    export ZEP_API_KEY=z_...
    (No ZEP_API_URL needed — the SDK defaults to Zep Cloud.)

  Self-hosted Zep (open source):
    export ZEP_API_URL=http://localhost:8000
    export ZEP_API_KEY=...  (if auth is enabled)

    Start Zep via Docker Compose:
      git clone https://github.com/getzep/zep.git
      cd zep && docker compose up

Zep stores conversation history per session, extracts facts automatically,
and supports semantic search over messages, facts, and summaries.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class ZepAdapter(BaseAdapter):
    """
    Adapter for Zep — long-term memory for AI assistants.

    Zep uses sessions to group conversations. Each benchmark question gets
    a fresh session. Messages are ingested into the session; recall queries
    use Zep's search API to find relevant facts, messages, and summaries.

    Prerequisites:
      1. Install: pip install zep-python
      2. Set ZEP_API_KEY (required for Zep Cloud, optional for self-hosted)
      3. For self-hosted: set ZEP_API_URL (e.g., http://localhost:8000)
    """

    @property
    def name(self) -> str:
        return "zep"

    @property
    def version(self) -> Optional[str]:
        try:
            import zep_python

            return getattr(zep_python, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(
        self,
        user_id: str = "benchd_user",
    ):
        self._user_id = user_id
        self._client = None  # lazy init
        self._session_id: Optional[str] = None

    def setup(self) -> None:
        """
        Initialize the Zep client and verify connectivity.
        Creates a user and an initial session for the benchmark run.
        """
        try:
            from zep_python.client import Zep
        except ImportError:
            raise RuntimeError(
                "zep-python package not installed. Install it with:\n"
                "  pip install zep-python"
            )

        api_key = os.environ.get("ZEP_API_KEY")
        api_url = os.environ.get("ZEP_API_URL")

        if not api_key and not api_url:
            raise RuntimeError(
                "Zep requires configuration. Set one of:\n"
                "  Zep Cloud:      export ZEP_API_KEY=z_...\n"
                "  Self-hosted:    export ZEP_API_URL=http://localhost:8000\n"
                "                  export ZEP_API_KEY=...  (if auth enabled)"
            )

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_url:
            kwargs["base_url"] = api_url

        try:
            self._client = Zep(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Zep client: {e}\n"
                f"Ensure Zep is running and credentials are correct."
            )

        # Ensure the benchmark user exists.
        try:
            self._client.user.get(self._user_id)
        except Exception:
            try:
                self._client.user.add(user_id=self._user_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create Zep user '{self._user_id}': {e}\n"
                    f"Ensure the Zep server is reachable."
                )

        self._create_session()

    def _create_session(self) -> None:
        """Create a fresh Zep session for a benchmark question."""
        self._session_id = f"benchd-{uuid.uuid4().hex[:12]}"
        self._client.memory.add_session(
            session_id=self._session_id,
            user_id=self._user_id,
        )

    def _delete_session(self) -> None:
        """Delete the current session's memory, if one exists."""
        if self._client is not None and self._session_id is not None:
            try:
                self._client.memory.delete(self._session_id)
            except Exception:
                pass
            self._session_id = None

    def teardown(self) -> None:
        """Delete session and optionally clean up the user."""
        self._delete_session()
        # Optionally delete the user to leave no state behind.
        if self._client is not None:
            try:
                self._client.user.delete(self._user_id)
            except Exception:
                pass

    def reset(self) -> None:
        """Reset state between benchmark questions by creating a new session."""
        self._delete_session()
        if self._client is not None:
            self._create_session()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Add conversation turns to the Zep session.

        Zep ingests messages with role_type and content. It automatically
        extracts facts and builds summaries from the conversation.
        """
        if self._client is None or self._session_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            from zep_python import Message
        except ImportError:
            raise RuntimeError("zep-python package not available.")

        # Map our turn roles to Zep role_type values.
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
        }

        messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            zep_role = role_map.get(role, "user")

            messages.append(
                Message(
                    role_type=zep_role,
                    content=content,
                )
            )

        if messages:
            try:
                self._client.memory.add(
                    self._session_id,
                    messages=messages,
                )
            except Exception as e:
                # Partial failure is better than crashing the run.
                pass

    def recall(self, query: str) -> str:
        """
        Search Zep's memory for information relevant to the query.

        Uses Zep's search_sessions API to find matching facts, messages,
        and summaries. Returns a concatenated string of results.
        """
        if self._client is None or self._session_id is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        results_parts: list[str] = []

        # Strategy 1: Search across sessions for facts and messages.
        try:
            search_response = self._client.memory.search_sessions(
                text=query,
                session_ids=[self._session_id],
                search_scope="facts",
                search_type="similarity",
                limit=10,
            )

            if search_response and search_response.results:
                for result in search_response.results:
                    # Extract fact text
                    if result.fact and result.fact.fact:
                        results_parts.append(result.fact.fact)
                    # Extract message content
                    if result.message and result.message.content:
                        results_parts.append(result.message.content)
                    # Extract summary
                    if result.summary and hasattr(result.summary, "content"):
                        summary_text = getattr(result.summary, "content", None)
                        if summary_text:
                            results_parts.append(summary_text)
        except Exception:
            pass

        # Strategy 2: Also search messages directly.
        if not results_parts:
            try:
                search_response = self._client.memory.search_sessions(
                    text=query,
                    session_ids=[self._session_id],
                    search_scope="messages",
                    search_type="similarity",
                    limit=10,
                )

                if search_response and search_response.results:
                    for result in search_response.results:
                        if result.message and result.message.content:
                            results_parts.append(result.message.content)
                        if result.fact and result.fact.fact:
                            results_parts.append(result.fact.fact)
            except Exception:
                pass

        # Strategy 3: Fall back to retrieving the full memory (facts + summary).
        if not results_parts:
            try:
                memory = self._client.memory.get(self._session_id)

                if memory.relevant_facts:
                    for fact in memory.relevant_facts:
                        if fact.fact:
                            results_parts.append(fact.fact)

                if memory.facts:
                    for fact_str in memory.facts:
                        if fact_str and fact_str not in results_parts:
                            results_parts.append(fact_str)

                if memory.summary and hasattr(memory.summary, "content"):
                    summary_text = getattr(memory.summary, "content", None)
                    if summary_text:
                        results_parts.append(summary_text)
            except Exception as e:
                return f"[recall error: {e}]"

        if not results_parts:
            return ""

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_parts: list[str] = []
        for part in results_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return ". ".join(unique_parts)
