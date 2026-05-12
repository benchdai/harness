"""
Khoj adapter for Bench'd harness.

Connects to a running Khoj server to ingest conversations via file upload
and query memory via semantic search.

Requires:
  A running Khoj server (self-hosted or cloud)

  Start with: khoj --anonymous-mode
  Or: docker run -p 42110:42110 khoj

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
        """Clear plaintext content index."""
        if self._session and self._base_url:
            try:
                self._session.delete(
                    f"{self._base_url}/api/content",
                    params={"t": "plaintext"},
                    timeout=10,
                )
            except Exception:
                pass

    def teardown(self) -> None:
        self.reset()
        if self._session:
            self._session.close()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Build conversation as plaintext
        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")
            prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{prefix}{role}: {content}")

        text_content = "\n".join(lines)

        # Upload as plaintext file via PUT /api/content
        try:
            self._session.put(
                f"{self._base_url}/api/content",
                files=[("files", ("conversation.txt", text_content, "text/plain"))],
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"Khoj ingest failed: {e}")

    def recall(self, query: str) -> str:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            resp = self._session.get(
                f"{self._base_url}/api/search",
                params={"q": query, "t": "plaintext", "n": 5},
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
