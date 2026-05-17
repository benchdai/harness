"""
Model pricing table for ProofMeter cost estimation.

Prices in USD per 1M tokens. Updated May 2026.

IMPORTANT: These are LIST PRICES from provider public pricing pages.
Customers may pay different rates due to:
  - Enterprise volume discounts
  - Committed-use agreements
  - Proxy/aggregator pricing (OpenRouter, etc.)
  - Free tier credits
  - Regional pricing

ProofMeter receipts always record:
  1. Exact token counts (provable from API response)
  2. Cost estimate with pricing_source label
  3. Customer can override with their own pricing table

The receipt format distinguishes:
  - pricing_source: "list_price" (our estimate)
  - pricing_source: "customer_provided" (their rates)
  - pricing_source: "provider_reported" (if API returns actual cost)
"""

from __future__ import annotations

from typing import Optional

# (input_per_1m, output_per_1m) in USD
PRICING_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "o3": (10.00, 40.00),
        "o3-mini": (1.10, 4.40),
        "o4-mini": (1.10, 4.40),
    },
    "anthropic": {
        "claude-opus-4-6": (15.00, 75.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5": (0.80, 4.00),
    },
    "google": {
        "gemini-2.5-pro": (1.25, 10.00),
        "gemini-2.5-flash": (0.15, 0.60),
    },
}

PRICING_VERSION = "2026-05"

# Usage-only mode flag
_usage_only_mode: bool = False

# Active pricing table — starts as list prices, can be overridden
_active_pricing: dict[str, dict[str, tuple[float, float]]] = dict(PRICING_TABLE)
_active_pricing_source: str = "public_estimate"
_active_price_book_id: str = f"proofmeter_public_{PRICING_VERSION}"


def set_custom_pricing(
    custom_table: dict[str, dict[str, tuple[float, float]]],
    source: str = "customer_price_book",
    price_book_id: Optional[str] = None,
) -> None:
    """
    Override the pricing table with customer-specific rates.

    Args:
        custom_table: Same format as PRICING_TABLE — {provider: {model: (input_per_1m, output_per_1m)}}
        source: Label for receipts — "customer_price_book", "enterprise_agreement", etc.
        price_book_id: Optional ID for the price book (for audit trail).

    Example:
        set_custom_pricing({
            "openai": {
                "gpt-4o": (1.25, 5.00),  # 50% enterprise discount
            }
        }, source="customer_price_book", price_book_id="acme_openai_q2_2026")
    """
    global _active_pricing, _active_pricing_source, _active_price_book_id, _usage_only_mode
    _usage_only_mode = False
    # Merge custom over defaults (custom overrides, defaults fill gaps)
    merged = dict(PRICING_TABLE)
    for provider, models in custom_table.items():
        if provider not in merged:
            merged[provider] = {}
        else:
            merged[provider] = dict(merged[provider])
        merged[provider].update(models)
    _active_pricing = merged
    _active_pricing_source = source
    _active_price_book_id = price_book_id or f"custom_{source}"


def set_usage_only_mode() -> None:
    """Disable cost calculation entirely. Receipts track usage only."""
    global _usage_only_mode, _active_pricing_source
    _usage_only_mode = True
    _active_pricing_source = "usage_only"


def reset_pricing() -> None:
    """Reset to default public estimated prices."""
    global _active_pricing, _active_pricing_source, _active_price_book_id, _usage_only_mode
    _active_pricing = dict(PRICING_TABLE)
    _active_pricing_source = "public_estimate"
    _active_price_book_id = f"proofmeter_public_{PRICING_VERSION}"
    _usage_only_mode = False


def get_pricing_source() -> str:
    """Return current pricing basis label for receipts."""
    return _active_pricing_source


def get_price_book_id() -> str:
    """Return current price book identifier."""
    return _active_price_book_id


def get_price_book_hash() -> str:
    """Compute SHA-256 hash of the active pricing table for reproducibility."""
    import hashlib
    import json
    canonical = json.dumps(_active_pricing, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def is_usage_only() -> bool:
    """Check if cost calculation is disabled."""
    return _usage_only_mode


def estimate_cost_cents(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> int:
    """
    Estimate cost in cents for a model call.

    Uses the active pricing table (list prices by default, or customer-provided).
    Returns 0 if usage_only mode, model not in table, or tokens are 0.
    """
    if _usage_only_mode:
        return 0

    provider_prices = _active_pricing.get(provider, {})
    prices = provider_prices.get(model)

    if prices is None:
        # Try prefix match (e.g., "gpt-4o-mini-2024" matches "gpt-4o-mini")
        for model_key, model_prices in provider_prices.items():
            if model.startswith(model_key):
                prices = model_prices
                break

    if prices is None:
        return 0

    if input_tokens == 0 and output_tokens == 0:
        return 0

    input_per_1m, output_per_1m = prices
    input_cost = (input_tokens / 1_000_000) * input_per_1m
    output_cost = (output_tokens / 1_000_000) * output_per_1m
    total_usd = input_cost + output_cost

    # Round to nearest cent, minimum 1 cent if any usage occurred
    return max(1, round(total_usd * 100))


def estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate cost in USD with full precision (for human display).

    Uses the active pricing table (list prices by default, or customer-provided).
    Returns float like 0.00234 — use for per-question cost display.
    """
    provider_prices = _active_pricing.get(provider, {})
    prices = provider_prices.get(model)

    if prices is None:
        for model_key, model_prices in provider_prices.items():
            if model.startswith(model_key):
                prices = model_prices
                break

    if prices is None:
        return 0.0

    input_per_1m, output_per_1m = prices
    input_cost = (input_tokens / 1_000_000) * input_per_1m
    output_cost = (output_tokens / 1_000_000) * output_per_1m
    return round(input_cost + output_cost, 6)


def format_cost_human(cost_usd: float) -> str:
    """
    Format a cost for human display.

    Examples:
        0.0 → "Free"
        0.00012 → "$0.0001"
        0.0234 → "$0.023"
        1.50 → "$1.50"
        42.81 → "$42.81"
    """
    if cost_usd == 0:
        return "Free"
    if cost_usd < 0.001:
        return f"${cost_usd:.4f}"
    if cost_usd < 0.10:
        return f"${cost_usd:.3f}"
    return f"${cost_usd:.2f}"


def format_tokens_human(tokens: int) -> str:
    """
    Format token count for human display.

    Examples:
        500 → "500"
        1200 → "1.2K"
        45000 → "45K"
        1500000 → "1.5M"
    """
    if tokens < 1000:
        return str(tokens)
    if tokens < 10000:
        return f"{tokens / 1000:.1f}K"
    if tokens < 1000000:
        return f"{tokens // 1000}K"
    return f"{tokens / 1000000:.1f}M"
