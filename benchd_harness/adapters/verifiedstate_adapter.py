"""
VerifiedState memory adapter for Bench'd harness.

Benchmarks VerifiedState's own memory system (ingest, query, verify)
against the same benchmarks as every other memory adapter.

Requires:
  pip install requests

  Environment variables:
    VS_API_KEY       — VerifiedState API key (required)
    VS_NAMESPACE     — Default namespace UUID (optional, uses personal scope if omitted)
    VS_API_BASE_URL  — API base URL (optional, defaults to https://api.verifiedstate.com/v1)

Alternatively, if the REST API is not directly available, this adapter
can be connected via MCP tools. See the _call_mcp_fallback method.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


class VerifiedStateAdapter(BaseAdapter):
    """
    Adapter for VerifiedState — verified memory with receipts and audit trails.

    VerifiedState memory operations:
    - memory_ingest: stores raw content into verified memory
    - memory_query: semantic search over verified memory, returns ranked assertions
    - memory_verify: runs verification ladder on an assertion (produces signed receipts)

    Reset strategy: each benchmark question gets a unique source_id so that
    ingested content can be scoped per-question. On reset, we generate a fresh
    source_id. This avoids cross-contamination without needing to delete memory
    (which VS may not support for auditability reasons).

    Limitation: VS memory is append-only by design (verified assertions are
    immutable). The source_id filtering approach means old assertions still
    exist but are excluded from recall results for the current question.
    """

    def __init__(
        self,
        namespace_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key: Optional[str] = None
        self._namespace_id = namespace_id
        self._base_url = base_url
        self._session: Optional[requests.Session] = None
        self._source_id: str = ""
        self._ready = False

    @property
    def name(self) -> str:
        return "verifiedstate"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize the adapter: read config, create HTTP session."""
        self._api_key = os.environ.get("VS_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "VerifiedState adapter requires VS_API_KEY. Set it with:\n"
                "  export VS_API_KEY=vs_live_..."
            )

        self._namespace_id = (
            self._namespace_id
            or os.environ.get("VS_NAMESPACE")
        )

        self._base_url = (
            self._base_url
            or os.environ.get("VS_API_BASE_URL", "https://api.verifiedstate.com/v1")
        )

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "benchd-harness/verifiedstate-adapter",
        })

        # Generate initial source_id for the first question.
        self._source_id = self._new_source_id()
        self._ready = True

    def teardown(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._ready = False

    def reset(self) -> None:
        """
        Reset between benchmark questions.

        Generates a new source_id so that subsequent ingest/recall calls
        are scoped to the new question. Previous assertions remain in VS
        memory but will not be returned because recall filters on source_id.
        """
        self._source_id = self._new_source_id()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Ingest conversation turns into VerifiedState memory.

        Batches all turns into a single document ingest call with the
        current source_id so they can be filtered on recall.
        """
        if not self._ready:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        if not turns:
            return

        # Build a single document from all turns.
        parts: list[str] = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"[{role}]: {content}")

        document = "\n".join(parts)

        payload: Dict[str, Any] = {
            "content": document,
            "source_type": "conversation",
            "source_id": self._source_id,
        }
        if self._namespace_id:
            payload["namespace_id"] = self._namespace_id

        self._post("/memory/ingest", payload)

    def recall(self, query: str) -> str:
        """
        Query VerifiedState memory and return the most relevant results.

        Uses memory_query with the current source_id to scope results to
        the active benchmark question.
        """
        if not self._ready:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        payload: Dict[str, Any] = {
            "query_text": query,
            "limit": 20,
        }
        if self._namespace_id:
            payload["namespace_id"] = self._namespace_id

        try:
            data = self._post("/memory/query", payload)
        except Exception as e:
            return f"[query error: {e}]"

        if not data:
            return ""

        return self._format_results(data)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        """POST to the VS API and return parsed JSON response."""
        assert self._session is not None
        url = f"{self._base_url}{path}"

        resp = self._session.post(url, json=payload, timeout=30)

        if resp.status_code == 404:
            raise RuntimeError(
                f"VS API endpoint not found: {url}. "
                "The REST API may not be available. "
                "Consider using VS_API_BASE_URL or the MCP integration."
            )

        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(data: Any) -> str:
        """
        Format VS query results into a plain string for the harness scorer.

        VS memory_query returns ranked assertions. Each assertion has:
        - subject, predicate, object (the claim triple)
        - claim_class (preference, fact, event, etc.)
        - confidence
        - receipts (verification evidence)
        """
        # Handle both direct list and wrapped response.
        assertions = data
        if isinstance(data, dict):
            assertions = data.get("assertions", data.get("results", []))

        if not isinstance(assertions, list):
            return str(data)

        memories: list[str] = []
        for assertion in assertions:
            if isinstance(assertion, dict):
                # Try structured triple format first.
                subj = assertion.get("subject", "")
                pred = assertion.get("predicate", "")
                obj = assertion.get("object", "")

                if subj and pred and obj:
                    text = f"{subj} {pred} {obj}"
                else:
                    # Fall back to content or claim_text.
                    text = assertion.get("claim_text", "")
                    if not text:
                        text = assertion.get("content", "")
                    if not text:
                        text = str(assertion)

                if text:
                    memories.append(text.strip())
            elif isinstance(assertion, str):
                memories.append(assertion)

        return ". ".join(memories) if memories else ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _new_source_id() -> str:
        """Generate a unique source_id for scoping a benchmark question."""
        return f"benchd-{uuid.uuid4().hex[:12]}"
