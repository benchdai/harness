"""
AnythingLLM adapter for Bench'd harness.

AnythingLLM is a privacy-first AI productivity tool with document ingestion
and persistent memory. Uses HTTP API.

Requires:
  A running AnythingLLM instance
  export ANYTHING_LLM_URL=http://localhost:3001  (default)
  export ANYTHING_LLM_API_KEY=...
"""

import os
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


class AnythingLLMAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "anything-llm"

    @property
    def version(self) -> Optional[str]:
        return "api-v1"

    def __init__(self):
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None
        self._workspace_slug: str = "benchd-eval"

    def setup(self) -> None:
        self._base_url = os.environ.get("ANYTHING_LLM_URL", "http://localhost:3001")
        api_key = os.environ.get("ANYTHING_LLM_API_KEY")

        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

        # Verify connection
        try:
            resp = self._session.get(f"{self._base_url}/api/ping", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to AnythingLLM at {self._base_url}.\n"
                "Start with: docker run -d -p 3001:3001 mintplexlabs/anythingllm"
            )

        # Create workspace
        try:
            self._session.post(
                f"{self._base_url}/api/v1/workspace/new",
                json={"name": self._workspace_slug},
                timeout=10,
            )
        except Exception:
            pass

    def reset(self) -> None:
        if self._session and self._base_url:
            try:
                self._session.delete(
                    f"{self._base_url}/api/v1/workspace/{self._workspace_slug}",
                    timeout=10,
                )
                self._session.post(
                    f"{self._base_url}/api/v1/workspace/new",
                    json={"name": self._workspace_slug},
                    timeout=10,
                )
            except Exception:
                pass

    def teardown(self) -> None:
        self.reset()
        if self._session:
            self._session.close()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if not self._session or not self._base_url:
            raise RuntimeError("Not initialized.")

        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")

        text = "\n\n".join(lines)

        try:
            self._session.post(
                f"{self._base_url}/api/v1/workspace/{self._workspace_slug}/update-embeddings",
                json={"adds": [{"content": text, "metadata": {"source": "benchd"}}]},
                timeout=30,
            )
        except Exception as e:
            raise RuntimeError(f"AnythingLLM ingest failed: {e}")

    def recall(self, query: str) -> str:
        if not self._session or not self._base_url:
            raise RuntimeError("Not initialized.")

        try:
            resp = self._session.post(
                f"{self._base_url}/api/v1/workspace/{self._workspace_slug}/chat",
                json={"message": query, "mode": "query"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("textResponse", str(data))
        except Exception as e:
            return f"[recall error: {e}]"
