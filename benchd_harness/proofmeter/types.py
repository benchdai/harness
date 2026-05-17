"""
ProofMeter data types v1.1

Design principles:
1. Receipts attest to PROVABLE FACTS only (tokens, provider, model, timestamps)
2. Cost is ALWAYS a derived estimate unless invoice-reconciled
3. Pricing is a declared snapshot, not a truth claim
4. Private pricing never leaks into shared/exported receipts
5. Non-token usage units are first-class (API calls, images, compute seconds)
6. Every receipt is hash-chained and independently verifiable
7. Budget enforcement is denominated in the declared pricing basis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ─── Pricing ─────────────────────────────────────────────────

@dataclass
class PricingSnapshot:
    """
    Declares which pricing was used to compute cost estimates.

    This is NOT a truth claim — it's the declared basis for computation.
    Different parties can re-derive cost from the same receipt using
    their own pricing. The price_book_hash allows reproducibility.
    """
    basis: str = "public_estimate"
    # Pricing basis options:
    #   "public_estimate"     — ProofMeter's public list prices (default)
    #   "customer_price_book" — customer-supplied rates
    #   "usage_only"          — no cost calculation
    #   "invoice_reconciled"  — matched to actual billing (future)

    price_book_id: str = ""              # e.g., "proofmeter_public_2026_05"
    price_book_hash: Optional[str] = None  # SHA-256 of the pricing table for reproducibility
    effective_date: Optional[str] = None   # when this price book took effect
    currency: str = "USD"

    # Rates (null if usage_only or if private and not exported)
    input_rate_per_1m: Optional[float] = None
    output_rate_per_1m: Optional[float] = None

    # Privacy: when True, raw rates are excluded from shared/exported receipts
    rates_private: bool = False


@dataclass
class CostEstimate:
    """
    Derived cost — computed from usage + pricing snapshot.

    Explicitly labeled with confidence level. NEVER presented as "Cost:"
    without qualification.
    """
    estimated_cost_usd: Optional[float] = None
    cost_confidence: str = "estimated"
    # Confidence levels:
    #   "estimated"          — computed from declared price book (default)
    #   "customer_supplied"  — customer provided their own rates
    #   "invoice_reconciled" — matched to actual invoice/billing (future)
    #   "usage_only"         — no cost calculated

    pricing_snapshot: Optional[PricingSnapshot] = None
    reconciliation_status: str = "unreconciled"
    # Reconciliation:
    #   "unreconciled"  — not matched to invoice
    #   "reconciled"    — matched to actual billing
    #   "not_applicable" — usage_only mode

    # Display hint for dashboards
    display_prefix: str = "Estimated cost"
    # Options: "Estimated cost", "Private estimate", "Actual billed cost", "Usage only"


# ─── Usage Units ─────────────────────────────────────────────

@dataclass
class UsageUnit:
    """
    Single unit of measured usage. Supports tokens, API calls, images,
    compute seconds, storage, and any future unit type.
    """
    unit_type: str       # input_tokens, output_tokens, api_calls, images, compute_seconds, etc.
    quantity: float
    verified: bool = True  # whether this was observed from the API response


@dataclass
class ProvenUsage:
    """
    Provable usage facts — the core of what gets signed.
    Supports both token-based and non-token usage.
    """
    provider: str
    model: str
    endpoint_class: str              # chat, embedding, rerank, image, browser, tool, etc.
    units: list[UsageUnit] = field(default_factory=list)
    latency_ms: float = 0.0
    occurred_at: str = ""

    # Convenience accessors for the common token case
    @property
    def input_tokens(self) -> int:
        return int(next((u.quantity for u in self.units if u.unit_type == "input_tokens"), 0))

    @property
    def output_tokens(self) -> int:
        return int(next((u.quantity for u in self.units if u.unit_type == "output_tokens"), 0))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ─── Core Objects ────────────────────────────────────────────

@dataclass
class BudgetAuthorization:
    """Signed budget capability granting an agent permission to spend."""
    capability_id: str
    namespace_id: str
    agent_id: str
    max_budget_cents: int
    remaining_cents: int
    currency: str = "USD"
    expires_at: Optional[str] = None
    status: str = "active"

    # Budget is denominated in a specific pricing basis
    pricing_basis: str = "public_estimate"
    price_book_id: str = ""
    price_book_hash: Optional[str] = None

    # Scope restrictions
    allowed_providers: Optional[list[str]] = None
    allowed_endpoint_classes: Optional[list[str]] = None

    # Revocation
    revoked: bool = False
    revoked_at: Optional[str] = None


@dataclass
class SpendReceipt:
    """
    Signed, hash-chained receipt for a single usage event.

    PROVES: provider, model, usage units, timestamp, chain position.
    ESTIMATES: cost (via an attached pricing snapshot).
    NEVER CLAIMS: actual billed amount (unless invoice_reconciled).
    """
    receipt_id: str
    namespace_id: str
    actor_id: str

    # ── Provable usage (signed, verifiable) ──────────────────
    proven_usage: Optional[ProvenUsage] = None

    # ── Cost estimate (derived, not a fact) ──────────────────
    cost_estimate: Optional[CostEstimate] = None

    # ── Context ──────────────────────────────────────────────
    task_id: Optional[str] = None
    capability_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    # ── Replay protection ────────────────────────────────────
    idempotency_key: Optional[str] = None  # prevents duplicate receipts

    # ── Schema version (for forward compatibility) ─────────
    schema_version: str = "proofmeter.receipt.v1.1"

    # ── Signing key (for key rotation) ───────────────────────
    signing_key_id: Optional[str] = None  # which key signed this receipt

    # ── Chain integrity ──────────────────────────────────────
    sequence_number: Optional[int] = None  # monotonic per-capability
    event_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    status: str = "verified"

    # ── Verification scope (what CAN be verified) ────────────
    verification_scope: Optional[dict[str, bool]] = None
    # Default: {
    #   "usage_verified": true,     — tokens from API response
    #   "price_book_verified": true, — price book hash matches
    #   "cost_estimated": true,      — math is reproducible
    #   "invoice_reconciled": false,  — not matched to billing
    #   "provider_attested": false,   — provider did not sign response
    # }


@dataclass
class Settlement:
    """Merkle-rooted settlement batch for a completed run."""
    settlement_id: str
    namespace_id: str

    # ── Proven aggregates ────────────────────────────────────
    total_usage: list[UsageUnit] = field(default_factory=list)
    receipt_count: int = 0

    # ── Cost estimate (derived) ──────────────────────────────
    cost_estimate: Optional[CostEstimate] = None

    # ── Schema + signing ────────────────────────────────────
    schema_version: str = "proofmeter.settlement.v1.1"
    signing_key_id: Optional[str] = None

    # ── Crypto ───────────────────────────────────────────────
    merkle_root: Optional[str] = None
    signature: Optional[str] = None
    status: str = "settled"

    task_id: Optional[str] = None
    capability_id: Optional[str] = None


@dataclass
class ProofMeterRunContext:
    """ProofMeter context that travels with a benchmark run."""
    namespace_id: str
    agent_id: str
    task_id: str  # = run_id
    capability_id: Optional[str] = None
    max_budget_cents: Optional[int] = None
    pricing_basis: str = "public_estimate"

    # ── Running totals (proven) ──────────────────────────────
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    receipt_count: int = 0
    sequence_counter: int = 0  # monotonic receipt counter

    # ── Running totals (estimated) ───────────────────────────
    estimated_spend_cents: int = 0

    receipts: list[SpendReceipt] = field(default_factory=list)
    settlement: Optional[Settlement] = None
    budget_exceeded: bool = False
