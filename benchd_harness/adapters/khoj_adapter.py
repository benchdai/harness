"""
Khoj adapter for Bench'd harness.

Connects to a Khoj instance (self-hosted or cloud) to ingest conversations
and query memory via the Khoj API.

Requires:
  A running Khoj server (self-hosted or cloud)

  export KHOJ_URL=http://localhost:42110  (default)
  export KHOJ_API_KEY=...  (optional, for cloud)
"""

import os
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


class KhojAdapter(BaseAdapter):
    """Adapter for Khoj personal AI memory."""

    @property
    def name(self) -> str:
        return "khoj"

    @property
    def version(self) -> Optional[str]:
        return "api-v1"

    def __init__(self):
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None

    def setup(self) -> None:
        self._base_url = os.environ.get("KHOJ_URL", "http://localhost:42110")
        api_key = os.environ.get("KHOJ_API_KEY")

        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

        # Verify connection
        try:
            resp = self._session.get(f"{self._base_url}/api/health", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Khoj at {self._base_url}.\n"
                "Start Khoj with: khoj --anonymous-mode\n"
                "Or set KHOJ_URL to your Khoj instance."
            )

    def reset(self) -> None:
        # Khoj doesn't have a simple reset — we'll use a fresh conversation
        pass

    def teardown(self) -> None:
        if self._session:
            self._session.close()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Ingest conversation turns as chat messages
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            try:
                # Use Khoj's chat API to build conversation history
                self._session.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "q": content if role == "user" else f"[{role}]: {content}",
                        "stream": False,
                    },
                    timeout=30,
                )
            except Exception:
                # Also try indexing as a document
                try:
                    self._session.post(
                        f"{self._base_url}/api/v1/index/update",
                        json={
                            "t": content,
                            "metadata": {"role": role},
                        },
                        timeout=30,
                    )
                except Exception:
                    pass

    def recall(self, query: str) -> str:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            # Use Khoj's search API
            resp = self._session.get(
                f"{self._base_url}/api/search",
                params={"q": query, "t": "all", "n": 5},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()

            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, dict):
                        texts.append(r.get("entry", r.get("content", str(r))))
                    else:
                        texts.append(str(r))
                return "\n".join(texts[:5])
            return str(results)
        except Exception as e:
            return f"[recall error: {e}]"
