"""
RagFlow adapter for Bench'd harness.

RagFlow is an open-source RAG engine with deep document understanding.
Runs as a Docker service with REST API.

Requires:
  docker compose up -d  (from ragflow repo)
  export RAGFLOW_URL=http://localhost:9380
  export RAGFLOW_API_KEY=...
"""

import os
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


class RagFlowAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "ragflow"

    @property
    def version(self) -> Optional[str]:
        return "api-v1"

    def __init__(self):
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None
        self._dataset_id: Optional[str] = None

    def setup(self) -> None:
        self._base_url = os.environ.get("RAGFLOW_URL", "http://localhost:9380")
        api_key = os.environ.get("RAGFLOW_API_KEY")

        if not api_key:
            raise RuntimeError("RagFlow requires RAGFLOW_API_KEY.")

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

        try:
            resp = self._session.get(f"{self._base_url}/api/v1/datasets", timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to RagFlow at {self._base_url}.\n"
                "Start RagFlow: docker compose up -d (from ragflow repo)"
            )

    def reset(self) -> None:
        self._dataset_id = None

    def teardown(self) -> None:
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
            resp = self._session.post(
                f"{self._base_url}/api/v1/datasets",
                json={"name": "benchd-eval", "chunk_method": "naive"},
                timeout=30,
            )
            data = resp.json()
            self._dataset_id = data.get("data", {}).get("id")

            if self._dataset_id:
                self._session.post(
                    f"{self._base_url}/api/v1/datasets/{self._dataset_id}/documents",
                    json={"documents": [{"content": text, "name": "benchd-input.txt"}]},
                    timeout=30,
                )
        except Exception as e:
            raise RuntimeError(f"RagFlow ingest failed: {e}")

    def recall(self, query: str) -> str:
        if not self._session or not self._base_url:
            raise RuntimeError("Not initialized.")

        try:
            resp = self._session.post(
                f"{self._base_url}/api/v1/retrieval",
                json={"question": query, "dataset_ids": [self._dataset_id] if self._dataset_id else []},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            chunks = data.get("data", {}).get("chunks", [])
            texts = [c.get("content", str(c)) for c in chunks[:5]]
            return "\n".join(texts) if texts else ""
        except Exception as e:
            return f"[recall error: {e}]"
