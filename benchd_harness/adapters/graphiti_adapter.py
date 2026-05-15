"""
Graphiti adapter for Bench'd harness.

Graphiti (by Zep) is a graph-based temporal memory layer that builds
knowledge graphs from conversations. It requires Neo4j as its graph
database backend.

Requires:
  pip install graphiti-core

  External infrastructure:
    - Neo4j database (v5.x recommended)
      Docker quick-start:
        docker run -d --name neo4j -p 7687:7687 -p 7474:7474 \
          -e NEO4J_AUTH=neo4j/password neo4j:5

  Environment variables:
    NEO4J_URI       - Neo4j bolt URI (default: bolt://localhost:7687)
    NEO4J_USER      - Neo4j username (default: neo4j)
    NEO4J_PASSWORD  - Neo4j password (required)

    Either:
      OPENAI_API_KEY      - OpenAI API key
    Or (OpenRouter):
      OPENROUTER_API_KEY  - OpenRouter API key

    GRAPHITI_LLM_MODEL   - LLM model (default: gpt-4o-mini / openai/gpt-4o-mini)
    GRAPHITI_GROUP_ID    - Group ID for isolating benchmark data (default: benchd)
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


import threading

# Dedicated event loop running in a background thread for async operations
_LOOP = None
_THREAD = None

def _get_loop():
    """Get or create a dedicated event loop in a background thread."""
    global _LOOP, _THREAD
    if _LOOP is None or not _LOOP.is_running():
        _LOOP = asyncio.new_event_loop()
        _THREAD = threading.Thread(target=_LOOP.run_forever, daemon=True)
        _THREAD.start()
    return _LOOP

def _run_async(coro):
    """Run an async coroutine on the dedicated background loop."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)


class GraphitiAdapter(BaseAdapter):
    """
    Adapter for Graphiti — Zep's temporal knowledge graph memory layer.

    Graphiti extracts entities, relationships, and temporal information from
    conversations and stores them in a Neo4j graph database. Queries use
    hybrid search (BM25 + cosine similarity + BFS) with cross-encoder reranking.

    Prerequisites:
      1. A running Neo4j instance (v5.x recommended)
      2. An OpenAI or OpenRouter API key for LLM and embedding operations

    Environment variables:
      NEO4J_URI           - bolt://localhost:7687 (default)
      NEO4J_USER          - neo4j (default)
      NEO4J_PASSWORD      - required
      OPENAI_API_KEY      - OpenAI key
      OPENROUTER_API_KEY  - OpenRouter key (alternative)
      GRAPHITI_LLM_MODEL  - model name override
      GRAPHITI_GROUP_ID   - namespace for benchmark data (default: benchd)
    """

    @property
    def name(self) -> str:
        return "graphiti"

    @property
    def version(self) -> Optional[str]:
        try:
            import graphiti_core

            return getattr(graphiti_core, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self, group_id: Optional[str] = None):
        self._group_id = group_id or os.environ.get("GRAPHITI_GROUP_ID", "benchd")
        self._client = None  # Graphiti instance, lazily created in setup()
        self._episode_count = 0

    def setup(self) -> None:
        """
        Initialize the Graphiti client with Neo4j and LLM configuration.

        Raises RuntimeError if required dependencies or configuration are missing.
        """
        # --- Validate dependencies ---
        try:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client.openai_client import OpenAIClient
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        except ImportError:
            raise RuntimeError(
                "graphiti-core package not installed. Install it with: "
                "pip install graphiti-core"
            )

        # --- Validate Neo4j configuration ---
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")

        if not neo4j_password:
            raise RuntimeError(
                "Graphiti requires a Neo4j database. Set the following env vars:\n"
                "  export NEO4J_URI=bolt://localhost:7687\n"
                "  export NEO4J_USER=neo4j\n"
                "  export NEO4J_PASSWORD=your_password\n\n"
                "Quick-start with Docker:\n"
                "  docker run -d --name neo4j -p 7687:7687 -p 7474:7474 \\\n"
                "    -e NEO4J_AUTH=neo4j/password neo4j:5"
            )

        # --- Validate LLM API key ---
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "Graphiti requires an LLM API key. Set one of:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )

        # --- Configure LLM client ---
        if openrouter_key and not openai_key:
            api_key = openrouter_key
            base_url = "https://openrouter.ai/api/v1"
            default_model = "openai/gpt-4o-mini"
        else:
            api_key = openai_key
            base_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"

        model = os.environ.get("GRAPHITI_LLM_MODEL", default_model)

        llm_config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        llm_client = OpenAIClient(config=llm_config)

        # --- Configure embedder ---
        # Graphiti uses OpenAI embeddings. When using OpenRouter, we still
        # point embeddings at OpenRouter (which proxies embedding models).
        embedder_config = OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # --- Create Graphiti client ---
        self._client = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
        )

        # Build indices and constraints in Neo4j
        _run_async(self._client.build_indices_and_constraints())
        self._episode_count = 0

    def teardown(self) -> None:
        """Close the Graphiti client connection."""
        if self._client is not None:
            try:
                _run_async(self._client.close())
            except Exception:
                pass
            self._client = None

    def reset(self) -> None:
        """
        Reset state between benchmark questions.

        Graphiti stores data in Neo4j, so a full reset would require clearing
        the graph. We use a fresh group_id for each question to isolate data,
        which avoids expensive graph deletion operations.
        """
        # Rotate group_id to isolate each question's data
        self._group_id = f"benchd_{uuid.uuid4().hex[:8]}"
        self._episode_count = 0

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into Graphiti as episodes.

        Each turn is added as a separate episode so that Graphiti can extract
        entities, relationships, and temporal information from the conversation.
        """
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp_str = turn.get("timestamp")

            if not content.strip():
                continue

            # Parse timestamp or use current time
            if timestamp_str:
                try:
                    ref_time = datetime.fromisoformat(timestamp_str)
                    if ref_time.tzinfo is None:
                        ref_time = ref_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    ref_time = datetime.now(timezone.utc)
            else:
                ref_time = datetime.now(timezone.utc)

            self._episode_count += 1
            episode_name = f"turn_{self._episode_count}_{role}"

            _run_async(
                self._client.add_episode(
                    name=episode_name,
                    episode_body=f"{role}: {content}",
                    source_description=f"Conversation turn ({role})",
                    reference_time=ref_time,
                    group_id=self._group_id,
                )
            )

    def recall(self, query: str) -> str:
        """
        Search Graphiti's knowledge graph for relevant information.

        Returns entity relationships (edges) that match the query, formatted
        as a plain string.
        """
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            results = _run_async(
                self._client.search(
                    query=query,
                    group_ids=[self._group_id],
                    num_results=10,
                )
            )
        except Exception as e:
            return f"[search error: {e}]"

        if not results:
            return ""

        # Results are EntityEdge objects with fact/name attributes
        facts: list[str] = []
        for edge in results:
            # EntityEdge has a 'fact' field that describes the relationship
            fact = getattr(edge, "fact", None) or getattr(edge, "name", None)
            if fact:
                facts.append(str(fact))

        return ". ".join(facts) if facts else ""
