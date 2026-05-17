# Bench'd Harness

The neutral benchmark harness for AI memory systems. Every score is independently run, cryptographically signed, and verifiable by anyone. Spend attestation powered by ProofMeter (Patent Pending).

**[Leaderboard](https://benchd.ai/leaderboard)** | **[Docs](https://benchd.ai/docs)** | **[Methodology](https://benchd.ai/methodology)** | **[Submit Results](https://benchd.ai/submit)**

## Quick Start

```bash
pip install benchd-harness

# Generate signing keys
benchd keys generate --out ./keys

# Set your LLM API key (for the judge)
export OPENROUTER_API_KEY=sk-or-...

# Run LongMemEval against your MCP-compatible memory system
benchd run -a mcp -b longmemeval-v1 --judge --key ./keys/private.key \
  --adapter-config '{"endpoint": "http://localhost:3000/mcp"}'

# Submit results to the leaderboard
benchd submit ./runs/run_xxx/manifest.signed.json
```

## MCP Systems: Zero-Code Testing

If your memory system exposes an MCP server with `ingest` and `query` tools, you don't need to write any adapter code:

```bash
benchd run -a mcp -b longmemeval-v1 --judge \
  --adapter-config '{"endpoint": "http://localhost:3000/mcp"}'
```

The MCP adapter auto-discovers your tools and maps them to Bench'd's interface.

## Available Benchmarks

| Benchmark | Slug | Questions | What it tests |
|-----------|------|-----------|---------------|
| LongMemEval | `longmemeval-v1` | 500 | Recall, temporal reasoning, knowledge updates |
| LoCoMo | `locomo-v1` | 1,540 | Multi-session conversational memory |
| Smoke | `smoke-memory-v0` | 10 | Quick sanity check |

## Built-in Adapters

| Adapter | System | Install |
|---------|--------|---------|
| `mcp` | Any MCP server | Built-in |
| `mem0-local` | Mem0 OSS | `pip install benchd-harness[mem0]` |
| `langchain-memory` | LangChain | `pip install benchd-harness[langchain]` |
| `llamaindex-memory` | LlamaIndex | `pip install benchd-harness[llamaindex]` |
| `llm-baseline` | Raw LLM (no memory) | `pip install openai` |
| `echo` | Test adapter | Built-in |

## Writing a Custom Adapter

```python
from benchd_harness.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "my-system"

    def setup(self) -> None:
        self.client = MyMemoryClient()

    def ingest(self, turns: list[dict]) -> None:
        for turn in turns:
            self.client.add(role=turn["role"], content=turn["content"])

    def recall(self, query: str) -> str:
        return self.client.search(query).text

    def reset(self) -> None:
        self.client.clear()
```

Register in `benchd_harness/adapters/__init__.py` and run with `benchd run -a my-system`.

## Commands

| Command | Description |
|---------|-------------|
| `benchd run` | Run a benchmark against a memory system |
| `benchd submit` | Submit signed results to benchd.ai |
| `benchd verify` | Verify a signed manifest |
| `benchd keys generate` | Generate Ed25519 signing keys |
| `benchd list` | List available adapters and benchmarks |
| `benchd baselines` | Recompute track baselines from scored manifests |

## ProofMeter — Spend Attestation (Patent Pending)

Track the cost of benchmark runs with cryptographic receipts. ProofMeter separates proven usage (tokens, provider, model) from estimated cost (derived from a declared pricing table).

```bash
# Run with $5 budget and spend tracking
benchd run -a verifiedstate -b reliability-v1 --budget 5.00

# Custom enterprise pricing
benchd run -a my-adapter -b longmemeval-v1 --budget 10.00 --pricing my-rates.json

# Usage tracking only (no cost calculation)
benchd run -a my-adapter -b smoke-memory-v0 --pricing-mode usage-only
```

The signed manifest includes a `proofmeter` section with proven token counts, estimated cost (labeled with pricing basis and confidence), and a Merkle-rooted settlement.

[Receipt Specification](https://benchd.ai/methodology/receipt-spec) | [Trust Boundaries](https://benchd.ai/methodology/trust-boundaries)

## Signing & Verification

Every run produces an Ed25519-signed manifest containing all inputs, outputs, scores, and failure traces. Anyone can verify:

```bash
benchd verify ./runs/run_xxx/manifest.signed.json
```

## Current Results (May 2026)

| # | System | LongMemEval | Status |
|---|--------|-------------|--------|
| 1 | LlamaIndex | 59.0% | Verified |
| 1 | LangChain | 59.0% | Verified |
| 3 | LLM Baseline | 57.6% | Verified |
| 4 | Mem0 OSS | 32.4% | Verified |

Full results at [benchd.ai/leaderboard](https://benchd.ai/leaderboard).

## Links

- **Website**: [benchd.ai](https://benchd.ai)
- **Leaderboard**: [benchd.ai/leaderboard](https://benchd.ai/leaderboard)
- **Docs**: [benchd.ai/docs](https://benchd.ai/docs)
- **Submit**: [benchd.ai/submit](https://benchd.ai/submit)
