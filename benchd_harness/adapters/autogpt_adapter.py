"""
AutoGPT Memory adapter for Bench'd harness.

Uses AutoGPT's memory subsystem with vector-store backed retrieval.

Requires:
  pip install chromadb sentence-transformers

  export OPENAI_API_KEY=sk-...  (or OPENROUTER_API_KEY for recall LLM)
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class AutoGPTMemoryAdapter(BaseAdapter):
    """
    Adapter simulating AutoGPT's memory pattern: embed conversation turns
    into a local vector store (ChromaDB), retrieve on recall.
    """

    @property
    def name(self) -> str:
        return "autogpt-memory"

    @property
    def version(self) -> Optional[str]:
        try:
            import chromadb
            return getattr(chromadb, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._collection = None
        self._client = None

    def setup(self) -> None:
        try:
            import chromadb
        except ImportError:
            raise RuntimeError(
                "chromadb not installed. Install with:\n"
                "  pip install chromadb"
            )

        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=f"benchd_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        if self._client is not None:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.create_collection(
                name=f"benchd_{uuid.uuid4().hex[:8]}",
                metadata={"hnsw:space": "cosine"},
            )

    def teardown(self) -> None:
        self._collection = None
        self._client = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._collection is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        documents = []
        ids = []
        metadatas = []

        for i, turn in enumerate(turns):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")

            documents.append(f"[{role}]: {content}")
            ids.append(f"turn_{i}_{uuid.uuid4().hex[:6]}")
            metadatas.append({"role": role, "index": i, "timestamp": timestamp})

        if documents:
            self._collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

    def recall(self, query: str) -> str:
        if self._collection is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(10, self._collection.count()),
            )
            if results and results.get("documents"):
                docs = results["documents"][0]  # first query's results
                return "\n".join(docs)
            return ""
        except Exception as e:
            return f"[recall error: {e}]"
