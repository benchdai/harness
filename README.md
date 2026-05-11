# Bench'd Harness

Neutral benchmark runner for AI memory systems. Runs standardized benchmarks against any memory system adapter, scores deterministically, produces signed manifests with full failure traces.

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Generate signing keys
benchd keys generate --out ./keys

# Run smoke benchmark against the echo adapter
benchd run --adapter echo --benchmark smoke-memory-v0 --out ./runs --key ./keys/private.key

# Verify the signed receipt
benchd verify ./runs/<run_id>/manifest.signed.json
```

## Commands

| Command | Description |
|---------|-------------|
| `benchd run` | Run a benchmark against a memory system adapter |
| `benchd verify` | Verify a signed manifest's cryptographic signature |
| `benchd keys generate` | Generate an Ed25519 signing keypair |
| `benchd list` | List available adapters and benchmarks |

## Architecture

```
benchd_harness/
  adapters/       ← Memory system interfaces (ingest/recall)
  benchmarks/     ← Benchmark datasets and loaders
  scoring/        ← Deterministic scoring (exact, regex, LLM-pending)
  signing/        ← Ed25519 signing (local dev + VerifiedState production)
  runner.py       ← Orchestrator: ingest → recall → score → manifest → sign
  manifest.py     ← Manifest schema and builder
  cli.py          ← CLI entry point
```

Every piece is independent. Swap an adapter without touching the scorer. Add a benchmark without touching the runner. Change signing backend without touching anything else.

## Adapter Interface

Every memory system adapter implements two methods:

```python
class BaseAdapter(ABC):
    def ingest(self, turns: list[dict]) -> None:
        """Feed conversation turns into the memory system."""
        ...

    def recall(self, query: str) -> str:
        """Query the memory system, return a plain string."""
        ...
```

Turn format:
```json
{"role": "user", "content": "...", "timestamp": "2026-05-01T10:00:00Z"}
```

### Built-in Adapters

| Adapter | Purpose |
|---------|---------|
| `echo` | Stores turns in memory, keyword-matches on recall. For testing the harness. |
| `null` | Returns empty string. Proves failure traces work. |

### Adding an Adapter

Create a file in `benchd_harness/adapters/`, implement `BaseAdapter`, register in `__init__.py`. That's it.

## Benchmarks

### Built-in

| Benchmark | Slug | Questions | Purpose |
|-----------|------|-----------|---------|
| Smoke Memory v0 | `smoke-memory-v0` | 10 | Harness integration test fixture |

### Planned Real Benchmarks

| Benchmark | What it tests |
|-----------|---------------|
| LongMemEval | Long-term memory across sessions (Microsoft Research) |
| LoCoMo | Memory over long conversations |
| PersonaMem | Persona consistency and preference tracking |
| MemoChat | Memory-augmented dialogue quality |

### Adding a Benchmark

Create a file in `benchd_harness/benchmarks/`, implement `BaseBenchmark.load_items()` returning `BenchmarkItem` objects, register in `__init__.py`.

## Scoring

Two modes, used by dimension:

| Method | How it works | Used for |
|--------|-------------|----------|
| **Exact match** | Normalized comparison + containment check | Recall dimension |
| **Regex** | Pattern search against response | Temporal dimension |
| **LLM judge** | Pending — not yet connected | Reasoning dimension (nuance score) |

The **Verified Score** only counts exact/regex items. The **Nuance Score** only counts LLM-judged items (currently `null`/pending).

## Manifest Schema

Every run produces a signed JSON manifest:

```json
{
  "manifest": {
    "version": "1.0.0",
    "run_id": "run_xxxxxxxxxxxx",
    "system": { "name": "...", "adapter": "...", "version": "..." },
    "benchmark": { "slug": "...", "name": "...", "version": "..." },
    "harness": { "version": "0.1.0" },
    "scores": {
      "verified": { "recall": 85.7, "temporal": 66.7, "reasoning": null, "overall": 76.2 },
      "nuance": { "recall": null, "temporal": null, "reasoning": null, "overall": null }
    },
    "summary": { "total_questions": 10, "scored_questions": 7, "pending_questions": 3, "passed": 6, "failed": 1 },
    "traces": [...]
  },
  "manifest_hash": "sha256...",
  "signature": "ed25519...",
  "public_key": "hex...",
  "signing_key_fingerprint": "07f49651a8736f77",
  "signed_at": "2026-05-10T...",
  "signing_mode": "local"
}
```

## Signing

### Local (Development)

Ed25519 via PyNaCl. Generate a keypair, sign manifests locally, verify independently.

```bash
benchd keys generate --out ./keys
benchd run --adapter echo --benchmark smoke-memory-v0 --key ./keys/private.key
benchd verify ./runs/<run_id>/manifest.signed.json
```

### VerifiedState (Production)

Production signing delegates to VerifiedState for independent third-party verification. Receipts are verifiable through VS explorer. _Not yet connected — stub in place._

## Tests

```bash
python -m pytest tests/ -v
```

29 tests covering scoring, adapters, signing, manifest generation, and full runner pipeline.

## What's NOT Built Yet

- Real benchmark loaders (LongMemEval, LoCoMo, PersonaMem)
- Real memory system adapters (Mem0, Letta, Zep)
- LLM judge for nuance scoring
- VerifiedState production signing integration
- Modal sandbox execution
- Frontend data ingestion pipeline

This harness is the foundation. Everything else plugs in.
