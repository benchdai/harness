"""
Bench'd Capability Declarations v1.0

Systems declare what they support. Benchmarks only score applicable categories.
A markdown system without delete support isn't penalized on deletion compliance.

Usage in benchd.yaml:
  capabilities:
    reset: full
    delete: true
    timestamps: true
    namespaces: false
    multi_agent: false
    citations: true
    token_budget: false
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemCapabilities:
    """What a memory system can and cannot do."""
    reset: str = "full"                # full | partial | none
    delete: bool = False               # Can delete specific memories
    update: bool = False               # Can update existing memories
    timestamps: bool = False           # Preserves temporal information
    namespaces: bool = False           # Supports logical isolation
    multi_agent: bool = False          # Supports agent-scoped memory
    citations: bool = False            # Returns source references
    token_budget: bool = False         # Respects token/result limits
    metadata_filters: bool = False     # Supports metadata-based filtering
    memory_ids: bool = False           # Returns stable memory item IDs
    temporal_scope: bool = False       # Tracks valid_from/valid_until
    provenance: bool = False           # Tracks who created each memory
    access_control: bool = False       # Role-based memory access


# Which capabilities are required for each benchmark
BENCHMARK_REQUIREMENTS: dict[str, list[str]] = {
    "longmemeval-v1": [],                        # No special requirements
    "locomo-v1": [],
    "reliability-v1": [],
    "knowledge-retrieval-v1": [],
    "knowledge-scale-v1": [],
    "semantic-rbac-v1": ["access_control"],       # Needs RBAC support
    "truth-arbitration-v1": ["timestamps"],       # Needs temporal awareness
    "memory-poisoning-v1": [],                    # All systems should resist this
    "smoke-memory-v0": [],
}

# Which capabilities each benchmark SCORES (vs just uses)
BENCHMARK_SCORED_CAPABILITIES: dict[str, list[str]] = {
    "reliability-v1": ["delete"],                 # Deletion compliance test
    "semantic-rbac-v1": ["access_control", "multi_agent"],
    "truth-arbitration-v1": ["timestamps", "temporal_scope"],
}


def parse_capabilities(config: dict[str, Any]) -> SystemCapabilities:
    """Parse capabilities from a YAML manifest."""
    caps = config.get("capabilities", {})
    return SystemCapabilities(
        reset=caps.get("reset", "full"),
        delete=caps.get("delete", False),
        update=caps.get("update", False),
        timestamps=caps.get("timestamps", False),
        namespaces=caps.get("namespaces", False),
        multi_agent=caps.get("multi_agent", False),
        citations=caps.get("citations", False),
        token_budget=caps.get("token_budget", False),
        metadata_filters=caps.get("metadata_filters", False),
        memory_ids=caps.get("memory_ids", False),
        temporal_scope=caps.get("temporal_scope", False),
        provenance=caps.get("provenance", False),
        access_control=caps.get("access_control", False),
    )


def check_eligibility(
    capabilities: SystemCapabilities,
    benchmark_slug: str,
) -> tuple[bool, list[str]]:
    """
    Check if a system is eligible for a benchmark based on capabilities.

    Returns (eligible, missing_capabilities).
    """
    requirements = BENCHMARK_REQUIREMENTS.get(benchmark_slug, [])
    missing = []

    for req in requirements:
        if not getattr(capabilities, req, False):
            missing.append(req)

    return len(missing) == 0, missing
