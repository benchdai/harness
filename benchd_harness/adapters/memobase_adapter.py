"""
Memobase adapter for Bench'd harness.

Requires:
  pip install memobase

  Set your Memobase API key:
    export MEMOBASE_API_KEY=...

  Optionally set a custom server URL:
    export MEMOBASE_URL=http://localhost:8019

Memobase is a user profile memory system that builds structured user profiles
from conversations. It extracts topics, preferences, and facts into a
searchable profile, and supports event-based semantic search.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class MemobaseAdapter(BaseAdapter):
    """
    Adapter for Memobase — user memory for personalized AI.

    Memobase builds user profiles from chat conversations. It ingests
    OpenAI-compatible message lists, extracts structured profile data,
    and supports context retrieval and event search.

    Prerequisites:
      1. Install: pip install memobase
      2. Set MEMOBASE_API_KEY environment variable
      3. Optionally set MEMOBASE_URL for self-hosted instances
    """

    @property
    def name(self) -> str:
        return "memobase"

    @property
    def version(self) -> Optional[str]:
        try:
            import memobase

            return getattr(memobase, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self) -> None:
        self._client = None  # lazy init
        self._user = None
        self._user_id: str = ""

    def setup(self) -> None:
        """Initialize the Memobase client and create a user."""
        try:
            from memobase import MemoBaseClient
        except ImportError:
            raise RuntimeError(
                "memobase package not installed. Install it with:\n"
                "  pip install memobase"
            )

        api_key = os.environ.get("MEMOBASE_API_KEY")
        project_url = os.environ.get("MEMOBASE_URL", "https://api.memobase.dev")

        if not api_key:
            raise RuntimeError(
                "Memobase requires an API key. Set it with:\n"
                "  export MEMOBASE_API_KEY=..."
            )

        try:
            self._client = MemoBaseClient(
                api_key=api_key,
                project_url=project_url,
            )
            self._client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Memobase client: {e}\n"
                f"Ensure the Memobase server is reachable at {project_url}."
            )

        self._create_user()

    def _create_user(self) -> None:
        """Create a fresh Memobase user for a benchmark question."""
        if self._client is None:
            return
        self._user_id = f"benchd-{uuid.uuid4().hex[:12]}"
        try:
            self._user = self._client.get_or_create_user(self._user_id)
        except Exception as e:
            raise RuntimeError(f"Failed to create Memobase user: {e}")

    def _delete_user(self) -> None:
        """Delete the current user's data, if one exists."""
        if self._client is not None and self._user_id:
            try:
                self._client.delete_user(self._user_id)
            except Exception:
                pass
            self._user = None
            self._user_id = ""

    def teardown(self) -> None:
        """Clean up the user."""
        self._delete_user()

    def reset(self) -> None:
        """Reset state between benchmark questions by creating a new user."""
        self._delete_user()
        if self._client is not None:
            self._create_user()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into Memobase.

        Converts turns to OpenAI-compatible messages and inserts them
        as a ChatBlob. Then flushes to trigger profile extraction.
        """
        if self._client is None or self._user is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            from memobase import ChatBlob
            from memobase.core.blob import OpenAICompatibleMessage
        except ImportError:
            raise RuntimeError("memobase package not available.")

        # Memobase only supports "user" and "assistant" roles.
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "user",  # Map system to user.
        }

        messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            mapped_role = role_map.get(role, "user")

            messages.append(
                OpenAICompatibleMessage(
                    role=mapped_role,
                    content=content,
                )
            )

        if messages:
            try:
                blob = ChatBlob(messages=messages)
                self._user.insert(blob, sync=True)
                # Flush to trigger profile extraction.
                self._user.flush(sync=True)
            except Exception:
                # Don't crash the run on ingest failure.
                pass

    def recall(self, query: str) -> str:
        """
        Query Memobase's memory for information relevant to the query.

        Uses two strategies:
        1. User context (structured profile + relevant events).
        2. Event search for semantically similar content.
        """
        if self._client is None or self._user is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        results_parts: list[str] = []

        # Strategy 1: Get user context (profile + events combined).
        try:
            context = self._user.context(max_token_size=2000)
            if context:
                results_parts.append(context)
        except Exception:
            pass

        # Strategy 2: Search events by semantic similarity.
        if not results_parts:
            try:
                events = self._user.search_event(query=query, topk=10)
                if events:
                    for event in events:
                        if event.event_data:
                            event_str = str(event.event_data)
                            if event_str:
                                results_parts.append(event_str)
            except Exception:
                pass

        # Strategy 3: Fall back to profile.
        if not results_parts:
            try:
                profiles = self._user.profile(max_token_size=2000)
                if profiles:
                    for p in profiles:
                        desc = getattr(p, "describe", None)
                        if desc is not None:
                            # describe is a property that returns a string
                            profile_text = str(desc)
                            if profile_text:
                                results_parts.append(profile_text)
                        else:
                            results_parts.append(
                                f"{p.topic}/{p.sub_topic}: {p.content}"
                            )
            except Exception:
                pass

        if not results_parts:
            return ""

        return ". ".join(results_parts)
