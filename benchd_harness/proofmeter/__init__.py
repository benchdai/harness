"""
ProofMeter — Cryptographic spend attestation for AI agent actions.

This submodule implements the Authorize → Meter → Sign → Settle → Verify
loop for tracking and proving the cost of benchmark runs.

Can be imported independently from the rest of the harness:

    from benchd_harness.proofmeter import ProofMeterClient

Patent Pending. ProofMeter is a patent-pending method for separating
cryptographically-attested resource consumption from policy-dependent
valuation in AI agent systems.
"""

from .client import ProofMeterClient
from .types import (
    BudgetAuthorization,
    SpendReceipt,
    Settlement,
    ProofMeterRunContext,
    ProvenUsage,
    UsageUnit,
    PricingSnapshot,
    CostEstimate,
)

__all__ = [
    "ProofMeterClient",
    "BudgetAuthorization",
    "SpendReceipt",
    "Settlement",
    "ProofMeterRunContext",
    "ProvenUsage",
    "UsageUnit",
    "PricingSnapshot",
    "CostEstimate",
]
