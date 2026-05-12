"""
Microsoft GraphRAG adapter for Bench'd harness.

Uses GraphRAG's indexing and query APIs to ingest conversation data
and retrieve information via knowledge graph queries.

Requires:
  pip install graphrag

  export GRAPHRAG_API_KEY=...  (OpenAI key for embeddings)
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class GraphRAGAdapter(BaseAdapter):
    """Adapter for Microsoft GraphRAG."""

    @property
    def name(self) -> str:
        return "graphrag"

    @property
    def version(self) -> Optional[str]:
        try:
            import graphrag
            return getattr(graphrag, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._workdir: Optional[Path] = None
        self._texts: list[str] = []

    def setup(self) -> None:
        try:
            import graphrag
        except ImportError:
            raise RuntimeError(
                "graphrag package not installed. Install with:\n"
                "  pip install graphrag"
            )

        api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GraphRAG requires an OpenAI API key for embeddings.\n"
                "Set OPENAI_API_KEY or GRAPHRAG_API_KEY."
            )

        self._workdir = Path(tempfile.mkdtemp(prefix="benchd_graphrag_"))
        self._texts = []

    def reset(self) -> None:
        self._texts = []
        if self._workdir and self._workdir.exists():
            import shutil
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = Path(tempfile.mkdtemp(prefix="benchd_graphrag_"))

    def teardown(self) -> None:
        if self._workdir and self._workdir.exists():
            import shutil
            shutil.rmtree(self._workdir, ignore_errors=True)

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._workdir is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Build conversation text
        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")
            prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{prefix}{role}: {content}")

        text = "\n".join(lines)
        self._texts.append(text)

        # Write to input directory for GraphRAG
        input_dir = self._workdir / "input"
        input_dir.mkdir(exist_ok=True)
        (input_dir / "conversation.txt").write_text("\n\n".join(self._texts))

    def recall(self, query: str) -> str:
        if self._workdir is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            from graphrag.query.cli import run_local_search
        except ImportError:
            try:
                # Try alternative import path
                from graphrag.api import query as graphrag_query
                result = graphrag_query(
                    root_dir=str(self._workdir),
                    query=query,
                    method="local",
                )
                if hasattr(result, "response"):
                    return str(result.response)
                return str(result)
            except Exception as e:
                return f"[recall error: {e}]"

        try:
            result = run_local_search(
                root_dir=str(self._workdir),
                query=query,
            )
            if hasattr(result, "response"):
                return str(result.response)
            return str(result)
        except Exception as e:
            return f"[recall error: {e}]"
