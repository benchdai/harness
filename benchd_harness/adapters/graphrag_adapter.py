"""
Microsoft GraphRAG adapter for Bench'd harness.

Uses GraphRAG's indexing and local search APIs. Note: GraphRAG is designed
for batch document indexing, not conversational memory. Indexing is expensive
(multiple LLM calls per document). This adapter exists to test how well
graph-based RAG compares against purpose-built memory systems.

Requires:
  pip install graphrag

  export OPENAI_API_KEY=sk-...  (for embeddings + entity extraction)
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class GraphRAGAdapter(BaseAdapter):
    """Adapter for Microsoft GraphRAG."""

    @property
    def name(self) -> str:
        return "microsoft-graphrag"

    @property
    def version(self) -> Optional[str]:
        try:
            import graphrag
            return getattr(graphrag, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._workdir: Optional[Path] = None
        self._indexed = False

    def setup(self) -> None:
        try:
            import graphrag  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "graphrag package not installed. Install with:\n"
                "  pip install graphrag"
            )

        api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GraphRAG requires an OpenAI API key.\n"
                "Set OPENAI_API_KEY or GRAPHRAG_API_KEY."
            )

        self._workdir = Path(tempfile.mkdtemp(prefix="benchd_graphrag_"))
        self._indexed = False

        # Initialize GraphRAG workspace
        try:
            from graphrag.api import initialize_project_at
            initialize_project_at(self._workdir)
        except ImportError:
            # Fallback: create minimal structure manually
            (self._workdir / "input").mkdir(exist_ok=True)
            (self._workdir / "output").mkdir(exist_ok=True)

    def reset(self) -> None:
        self._indexed = False
        if self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = Path(tempfile.mkdtemp(prefix="benchd_graphrag_"))
            (self._workdir / "input").mkdir(exist_ok=True)
            (self._workdir / "output").mkdir(exist_ok=True)

    def teardown(self) -> None:
        if self._workdir and self._workdir.exists():
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

        # Write to input directory
        input_dir = self._workdir / "input"
        input_dir.mkdir(exist_ok=True)
        (input_dir / "conversation.txt").write_text(text)

        # Run indexing
        try:
            from graphrag.config.load_config import load_config
            from graphrag.api import build_index

            config = load_config(self._workdir)
            asyncio.get_event_loop().run_until_complete(build_index(config=config))
            self._indexed = True
        except Exception as e:
            # Indexing may fail on short texts — record but don't crash
            self._indexed = False

    def recall(self, query: str) -> str:
        if self._workdir is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        if not self._indexed:
            return ""

        try:
            import pandas as pd
            from graphrag.config.load_config import load_config
            from graphrag.api import local_search

            config = load_config(self._workdir)
            output_dir = self._workdir / "output"

            entities = pd.read_parquet(output_dir / "entities.parquet")
            communities = pd.read_parquet(output_dir / "communities.parquet")
            community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
            text_units = pd.read_parquet(output_dir / "text_units.parquet")
            relationships = pd.read_parquet(output_dir / "relationships.parquet")

            response, context = asyncio.get_event_loop().run_until_complete(
                local_search(
                    config=config,
                    entities=entities,
                    communities=communities,
                    community_reports=community_reports,
                    text_units=text_units,
                    relationships=relationships,
                    covariates=None,
                    community_level=2,
                    response_type="Single Paragraph",
                    query=query,
                )
            )
            return str(response)
        except Exception as e:
            return f"[recall error: {e}]"
