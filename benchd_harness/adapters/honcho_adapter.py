"""
Honcho adapter for Bench'd harness.

Requires:
  pip install honcho-ai

  Set your Honcho API key:
    export HONCHO_API_KEY=...

  Optionally set environment and base URL:
    export HONCHO_ENVIRONMENT=local|production
    export HONCHO_BASE_URL=http://localhost:8000

Honcho is a user-context management platform that builds long-term memory
from conversations. It uses "peers" to track different conversation participants,
sessions to group messages, and automatically extracts conclusions (facts)
from conversations.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class HonchoAdapter(BaseAdapter):
    """
    Adapter for Honcho (honcho.dev) — user context management for AI apps.

    Honcho requires a running server (cloud or self-hosted).
    Messages are ingested into sessions with peer attribution.
    Recall uses session context and search.

    Prerequisites:
      1. Install: pip install honcho-ai
      2. Set HONCHO_API_KEY environment variable
    """

    @property
    def name(self) -> str:
        return "honcho"

    @property
    def version(self) -> Optional[str]:
        try:
            import honcho

            return getattr(honcho, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self) -> None:
        self._client = None  # lazy init
        self._session = None
        self._session_id: str = ""
        self._user_peer_id: str = "benchd-user"
        self._assistant_peer_id: str = "benchd-assistant"

    def setup(self) -> None:
        """Initialize the Honcho client and create a session."""
        try:
            from honcho import Honcho
        except ImportError:
            raise RuntimeError(
                "honcho-ai package not installed. Install it with:\n"
                "  pip install honcho-ai"
            )

        api_key = os.environ.get("HONCHO_API_KEY")
        base_url = os.environ.get("HONCHO_BASE_URL")
        environment = os.environ.get("HONCHO_ENVIRONMENT")

        if not api_key and not base_url:
            raise RuntimeError(
                "Honcho requires configuration. Set one of:\n"
                "  Honcho Cloud:    export HONCHO_API_KEY=...\n"
                "  Self-hosted:     export HONCHO_BASE_URL=http://localhost:8000"
            )

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if environment:
            kwargs["environment"] = environment

        try:
            self._client = Honcho(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Honcho client: {e}")

        # Ensure peers exist (get-or-create semantics).
        try:
            self._client.peer(self._user_peer_id)
            self._client.peer(self._assistant_peer_id)
        except Exception:
            pass  # Peers may already exist or be auto-created.

        self._create_session()

    def _create_session(self) -> None:
        """Create a fresh Honcho session for a benchmark question."""
        if self._client is None:
            return
        self._session_id = f"benchd-{uuid.uuid4().hex[:12]}"
        try:
            self._session = self._client.session(self._session_id)
        except Exception as e:
            raise RuntimeError(f"Failed to create Honcho session: {e}")

    def _delete_session(self) -> None:
        """Delete the current session, if one exists."""
        if self._session is not None:
            try:
                self._session.delete()
            except Exception:
                pass
            self._session = None
            self._session_id = ""

    def teardown(self) -> None:
        """Clean up the session."""
        self._delete_session()

    def reset(self) -> None:
        """Reset state between benchmark questions by creating a new session."""
        self._delete_session()
        if self._client is not None:
            self._create_session()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Add conversation turns to the Honcho session.

        Honcho uses peer_id to attribute messages to participants.
        User messages go to the user peer, assistant messages to the
        assistant peer, system messages to the user peer.
        """
        if self._client is None or self._session is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            from honcho import MessageCreateParams
        except ImportError:
            raise RuntimeError("honcho-ai package not available.")

        peer_map = {
            "user": self._user_peer_id,
            "assistant": self._assistant_peer_id,
            "system": self._user_peer_id,
        }

        messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            peer_id = peer_map.get(role, self._user_peer_id)

            messages.append(
                MessageCreateParams(
                    content=content,
                    peer_id=peer_id,
                )
            )

        if messages:
            try:
                self._session.add_messages(messages)
            except Exception:
                # Don't crash the run on ingest failure.
                pass

    def recall(self, query: str) -> str:
        """
        Query Honcho's memory for information relevant to the query.

        Uses two strategies:
        1. Session context (includes summaries and conclusions).
        2. Session search for semantically relevant messages.
        """
        if self._client is None or self._session is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        results_parts: list[str] = []

        # Strategy 1: Get session context (summaries + conclusions).
        try:
            ctx = self._session.context(
                summary=True,
                search_query=query,
                peer_target=self._user_peer_id,
            )
            if ctx and hasattr(ctx, "context") and ctx.context:
                results_parts.append(str(ctx.context))
            elif ctx and isinstance(ctx, str):
                results_parts.append(ctx)
        except Exception:
            pass

        # Strategy 2: Search session messages.
        try:
            search_results = self._session.search(query=query, limit=10)
            if search_results:
                for msg in search_results:
                    content = getattr(msg, "content", None)
                    if content and content not in results_parts:
                        results_parts.append(content)
        except Exception:
            pass

        # Strategy 3: Fall back to global workspace search.
        if not results_parts:
            try:
                search_results = self._client.search(query=query, limit=10)
                if search_results:
                    for msg in search_results:
                        content = getattr(msg, "content", None)
                        if content:
                            results_parts.append(content)
            except Exception:
                pass

        if not results_parts:
            return ""

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for part in results_parts:
            if part not in seen:
                seen.add(part)
                unique.append(part)

        return ". ".join(unique)
