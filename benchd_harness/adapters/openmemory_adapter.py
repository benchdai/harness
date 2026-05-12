"""
OpenMemory (by Mem0) adapter for Bench'd harness.

Connects to a local OpenMemory MCP server for cross-tool memory sharing.
OpenMemory is Mem0's self-hosted memory server.

Requires:
  A running OpenMemory server

  export OPENMEMORY_URL=http://localhost:8765  (default)
"""

import os
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


class OpenMemoryAdapter(BaseAdapter):
    """Adapter for Mem0's OpenMemory local server."""

    @property
    def name(self) -> str:
        return "openmemory"

    @property
    def version(self) -> Optional[str]:
        return "api-v1"

    def __init__(self):
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None

    def setup(self) -> None:
        self._base_url = os.environ.get("OPENMEMORY_URL", "http://localhost:8765")

        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

        # Verify connection
        try:
            resp = self._session.get(f"{self._base_url}/health", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to OpenMemory at {self._base_url}.\n"
                "Start OpenMemory: https://github.com/mem0ai/open-memory\n"
                "Or set OPENMEMORY_URL to your instance."
            )

    def reset(self) -> None:
        if self._session and self._base_url:
            try:
                self._session.delete(f"{self._base_url}/memories", timeout=10)
            except Exception:
                pass

    def teardown(self) -> None:
        self.reset()
        if self._session:
            self._session.close()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Build conversation as messages
        messages = []
        for turn in turns:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", ""),
            })

        try:
            self._session.post(
                f"{self._base_url}/memories",
                json={"messages": messages},
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"OpenMemory ingest failed: {e}")

    def recall(self, query: str) -> str:
        if self._session is None or self._base_url is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            resp = self._session.post(
                f"{self._base_url}/search",
                json={"query": query, "limit": 10},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()

            if isinstance(results, list):
                texts = []
                for r in results:
                    if isinstance(r, dict):
                        texts.append(r.get("memory", r.get("content", str(r))))
                    else:
                        texts.append(str(r))
                return "\n".join(texts)
            if isinstance(results, dict) and "results" in results:
                return "\n".join(
                    r.get("memory", str(r)) for r in results["results"]
                )
            return str(results)
        except Exception as e:
            return f"[recall error: {e}]"
