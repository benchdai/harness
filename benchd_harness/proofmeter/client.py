"""
ProofMeter client — HTTP interface to VerifiedState meter tools.

This client implements the Authorize → Meter → Sign → Settle → Verify loop
by calling the VerifiedState API directly. The same operations are available
as MCP tools (meter_authorize, meter_spend, meter_settle, meter_verify).

Usage:
    from benchd_harness.proofmeter import ProofMeterClient

    client = ProofMeterClient(api_key="vs_live_...", namespace_id="ns_...")
    budget = client.authorize_budget(agent_id="benchd-runner", max_budget_cents=500)
    receipt = client.record_spend(...)
    settlement = client.settle(task_id="run_abc123")

Patent Pending.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from .types import (
    BudgetAuthorization,
    SpendReceipt,
    Settlement,
    ProofMeterRunContext,
)


class ProofMeterClient:
    """Client for VerifiedState's Proof Meter spend attestation API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        namespace_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("PROOFMETER_API_KEY") or os.environ.get("VS_API_KEY", "")
        self._namespace_id = namespace_id or os.environ.get("PROOFMETER_NAMESPACE") or os.environ.get("VS_NAMESPACE", "")
        self._base_url = base_url or os.environ.get(
            "PROOFMETER_BASE_URL", os.environ.get("VS_API_BASE_URL", "https://api.proofmeter.com")
        )
        self._session: Optional[requests.Session] = None

    def connect(self) -> None:
        """Initialize HTTP session."""
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "proofmeter-python/0.1.0 (benchd-harness)",
        })

    def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None

    # ─── Budget Authorization ───────────────────────────────────

    def authorize_budget(
        self,
        agent_id: str,
        max_budget_cents: int,
        task_id: Optional[str] = None,
        scope: Optional[dict] = None,
        expires_at: Optional[str] = None,
    ) -> BudgetAuthorization:
        """
        Create a signed budget capability granting the agent permission to spend.

        Args:
            agent_id: Agent principal being granted the budget.
            max_budget_cents: Maximum spend allowed in cents.
            task_id: Optional task/run ID to scope the budget.
            scope: Optional restrictions (allowed_providers, allowed_endpoint_classes).
            expires_at: Optional ISO 8601 expiration timestamp.

        Returns:
            BudgetAuthorization with the capability_id for subsequent spend calls.
        """
        payload: dict[str, Any] = {
            "namespace_id": self._namespace_id,
            "authorized_agent_id": agent_id,
            "max_budget_cents": max_budget_cents,
        }
        if scope:
            payload["scope"] = scope
        if expires_at:
            payload["expires_at"] = expires_at
        if task_id:
            payload["metadata"] = {"task_id": task_id, "source": "benchd-harness"}

        data = self._post("/v1/capabilities", payload)

        return BudgetAuthorization(
            capability_id=data.get("capability_id", data.get("id", "")),
            namespace_id=self._namespace_id,
            agent_id=agent_id,
            max_budget_cents=max_budget_cents,
            remaining_cents=data.get("remaining_cents", max_budget_cents),
            currency=data.get("currency", "USD"),
            expires_at=expires_at,
            status=data.get("status", "active"),
        )

    # ─── Spend Recording ────────────────────────────────────────

    def record_spend(
        self,
        actor_id: str,
        provider_id: str,
        usage_unit: str,
        usage_quantity: float,
        cost_cents: int,
        task_id: Optional[str] = None,
        capability_id: Optional[str] = None,
        endpoint_class: Optional[str] = None,
        metadata: Optional[dict] = None,
        occurred_at: Optional[str] = None,
    ) -> SpendReceipt:
        """
        Record a spend event and get a signed, hash-chained receipt.

        Args:
            actor_id: Agent or principal incurring the cost.
            provider_id: Provider that served the request (e.g., "openai").
            usage_unit: Unit of usage (e.g., "input_tokens", "output_tokens").
            usage_quantity: Amount of usage.
            cost_cents: Cost in cents.
            task_id: Run/task ID for grouping.
            capability_id: Budget capability to debit.
            endpoint_class: Endpoint class (chat, embedding, etc.).
            metadata: Arbitrary metadata (question_id, benchmark, etc.).
            occurred_at: ISO 8601 timestamp (defaults to now).

        Returns:
            SpendReceipt with receipt_id, hash, and signature.
        """
        if occurred_at is None:
            occurred_at = datetime.now(timezone.utc).isoformat()

        payload: dict[str, Any] = {
            "namespace_id": self._namespace_id,
            "actor_id": actor_id,
            "provider_id": provider_id,
            "usage_unit": usage_unit,
            "usage_quantity": usage_quantity,
            "cost_cents": cost_cents,
            "occurred_at": occurred_at,
        }
        if task_id:
            payload["task_id"] = task_id
        if capability_id:
            payload["capability_id"] = capability_id
        if endpoint_class:
            payload["endpoint_class"] = endpoint_class
        if metadata:
            payload["metadata"] = metadata

        data = self._post("/v1/receipts", payload)

        return SpendReceipt(
            receipt_id=data.get("receipt_id", data.get("id", "")),
            namespace_id=self._namespace_id,
            actor_id=actor_id,
            provider_id=provider_id,
            usage_unit=usage_unit,
            usage_quantity=usage_quantity,
            cost_cents=cost_cents,
            occurred_at=occurred_at,
            task_id=task_id,
            capability_id=capability_id,
            endpoint_class=endpoint_class,
            event_hash=data.get("event_hash"),
            previous_hash=data.get("previous_hash"),
            signature=data.get("signature"),
            status=data.get("status", "verified"),
        )

    # ─── Budget Check ───────────────────────────────────────────

    def check_budget(self, capability_id: str) -> dict:
        """
        Check remaining budget for a capability.

        Returns:
            Dict with remaining_cents, total_spent_cents, receipt_count, etc.
        """
        return self._post("/v1/budget/" + capability_id, {})

    # ─── Settlement ─────────────────────────────────────────────

    def settle(
        self,
        task_id: Optional[str] = None,
        capability_id: Optional[str] = None,
    ) -> Settlement:
        """
        Settle outstanding receipts into a Merkle-rooted settlement batch.

        Args:
            task_id: Settle receipts for this task/run.
            capability_id: Settle receipts for this capability.

        Returns:
            Settlement with merkle_root, signature, total_cost, receipt_count.
        """
        payload: dict[str, Any] = {"namespace_id": self._namespace_id}
        if task_id:
            payload["task_id"] = task_id
        if capability_id:
            payload["capability_id"] = capability_id

        data = self._post("/v1/settlements", payload)

        return Settlement(
            settlement_id=data.get("settlement_id", data.get("id", "")),
            namespace_id=self._namespace_id,
            total_cost_cents=data.get("total_cost_cents", 0),
            receipt_count=data.get("receipt_count", 0),
            merkle_root=data.get("merkle_root"),
            signature=data.get("signature"),
            status=data.get("status", "settled"),
            task_id=task_id,
            capability_id=capability_id,
        )

    # ─── Verification ───────────────────────────────────────────

    def verify_receipt(self, receipt_id: str) -> dict:
        """
        Cryptographically verify a spend receipt.

        Returns:
            Dict with valid, hash_valid, signature_valid, chain_valid, etc.
        """
        return self._post("/v1/receipts/" + receipt_id + "/verify", {})

    # ─── Receipt Listing ────────────────────────────────────────

    def list_receipts(
        self,
        task_id: Optional[str] = None,
        capability_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """List spend receipts with optional filters."""
        payload: dict[str, Any] = {
            "namespace_id": self._namespace_id,
            "limit": limit,
        }
        if task_id:
            payload["task_id"] = task_id
        if capability_id:
            payload["capability_id"] = capability_id
        if actor_id:
            payload["actor_id"] = actor_id

        data = self._post("/v1/receipts", payload)
        if isinstance(data, list):
            return data
        return data.get("receipts", data.get("results", []))

    # ─── HTTP ───────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> Any:
        """POST to the VS meter API."""
        if not self._session:
            self.connect()
        assert self._session is not None

        url = f"{self._base_url}{path}"
        resp = self._session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
