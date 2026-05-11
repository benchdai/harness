"""
Cognee adapter for Bench'd harness.

Cognee is an open-source memory engine that processes text into knowledge graphs.
It uses LiteLLM under the hood, so it supports many LLM providers.

Requires:
  pip install cognee

  Either:
    export OPENAI_API_KEY=sk-...
  Or (OpenRouter):
    export OPENROUTER_API_KEY=sk-or-...

Cognee v1.0+ exposes a high-level API: remember(), recall(), forget().
The legacy API (add/cognify/search) also still works. This adapter uses
the v1.0 API when available, falling back to the legacy API.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


def _run_async(coro):
    """Run an async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running event loop (e.g. Jupyter).
        # Create a new loop in a thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class CogneeAdapter(BaseAdapter):
    """
    Adapter for Cognee — open-source memory engine with knowledge graphs.

    Cognee processes text through chunking, entity extraction, and graph
    construction. Queries are answered via graph-based retrieval.

    Environment variables:
      OPENAI_API_KEY      - OpenAI API key (preferred)
      OPENROUTER_API_KEY  - OpenRouter API key (alternative)
      COGNEE_LLM_MODEL    - LLM model name (default: openai/gpt-4o-mini for
                            OpenRouter, gpt-4o-mini for OpenAI)
    """

    @property
    def name(self) -> str:
        return "cognee"

    @property
    def version(self) -> Optional[str]:
        try:
            import cognee

            return getattr(cognee, "__version__", None) or cognee.get_cognee_version()
        except (ImportError, Exception):
            return None

    def __init__(self, dataset_name: str = "benchd_dataset"):
        self._dataset_name = dataset_name
        self._initialized = False

    def setup(self) -> None:
        """Initialize Cognee with LLM provider configuration."""
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if not openai_key and not openrouter_key:
            raise RuntimeError(
                "Cognee requires an LLM API key. Set one of:\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )

        try:
            import cognee
        except ImportError:
            raise RuntimeError(
                "cognee package not installed. Install it with: "
                "pip install cognee"
            )

        # Cognee uses LiteLLM under the hood, which supports OpenRouter
        # via the "openrouter/" prefix on model names.
        if openrouter_key and not openai_key:
            cognee.config.set_llm_api_key(openrouter_key)
            cognee.config.set_llm_endpoint("https://openrouter.ai/api/v1")
            cognee.config.set_llm_provider("openai")

            model = os.environ.get("COGNEE_LLM_MODEL", "openai/gpt-4o-mini")
            cognee.config.set_llm_model(model)

            # Embeddings: OpenRouter's embedding support is limited.
            # Use a small model and disable encoding_format parameter
            # by setting the embedding model explicitly.
            cognee.config.set_embedding_api_key(openrouter_key)
            cognee.config.set_embedding_endpoint("https://openrouter.ai/api/v1")
            cognee.config.set_embedding_model("openai/text-embedding-3-small")
            cognee.config.set_embedding_dimensions(1536)
        else:
            cognee.config.set_llm_api_key(openai_key)
            cognee.config.set_llm_provider("openai")

            model = os.environ.get("COGNEE_LLM_MODEL", "gpt-4o-mini")
            cognee.config.set_llm_model(model)

            cognee.config.set_embedding_api_key(openai_key)

        self._initialized = True

    def teardown(self) -> None:
        """Clean up Cognee state after a benchmark run."""
        if not self._initialized:
            return
        try:
            import cognee

            _run_async(cognee.prune.prune_data())
        except Exception:
            pass

    def reset(self) -> None:
        """Clear all Cognee data between benchmark questions."""
        if not self._initialized:
            return
        try:
            import cognee

            # forget(everything=True) wipes all memories
            _run_async(cognee.forget(everything=True))
        except Exception:
            # Fall back to prune if forget fails
            try:
                _run_async(cognee.prune.prune_data())
            except Exception:
                pass

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into Cognee.

        Converts turns to a text block and processes them through Cognee's
        knowledge graph pipeline (add + cognify, or remember).
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        import cognee

        # Build a conversation transcript for Cognee to process.
        lines: list[str] = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")

        text = "\n".join(lines)
        if not text.strip():
            return

        # Use the v1.0 remember() API which combines add + cognify
        try:
            _run_async(cognee.remember(text, dataset_name=self._dataset_name))
        except Exception:
            # Fall back to legacy add + cognify pipeline
            _run_async(cognee.add(text, dataset_name=self._dataset_name))
            _run_async(cognee.cognify(datasets=[self._dataset_name]))

    def recall(self, query: str) -> str:
        """
        Query Cognee for relevant memories and return as a plain string.

        Uses Cognee's recall() (v1.0) or search() (legacy) with graph-based
        completion to synthesize an answer from the knowledge graph.
        """
        if not self._initialized:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        import cognee

        try:
            # Try the v1.0 recall() API first
            results = _run_async(
                cognee.recall(
                    query,
                    datasets=[self._dataset_name],
                    top_k=10,
                )
            )
        except Exception:
            # Fall back to legacy search()
            try:
                from cognee.modules.search.types.SearchType import SearchType

                results = _run_async(
                    cognee.search(
                        query_text=query,
                        query_type=SearchType.GRAPH_COMPLETION,
                        datasets=[self._dataset_name],
                        top_k=10,
                    )
                )
            except Exception as e:
                return f"[search error: {e}]"

        if not results:
            return ""

        # Results can be various types depending on the API version.
        if isinstance(results, str):
            return results

        if isinstance(results, list):
            parts: list[str] = []
            for r in results:
                if isinstance(r, str):
                    parts.append(r)
                elif isinstance(r, dict):
                    # Legacy search results are dicts
                    text = r.get("text", r.get("content", r.get("answer", "")))
                    if text:
                        parts.append(str(text))
                elif hasattr(r, "answer"):
                    parts.append(str(r.answer))
                elif hasattr(r, "text"):
                    parts.append(str(r.text))
                elif hasattr(r, "content"):
                    parts.append(str(r.content))
                else:
                    parts.append(str(r))
            return ". ".join(parts) if parts else ""

        return str(results)
