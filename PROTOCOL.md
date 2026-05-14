# Bench'd Evaluation Protocol v0.1

The formal specification for how Bench'd evaluates AI memory systems. This document is the constitution — it defines the rules that make results comparable, reproducible, and trustworthy.

## 1. Adapter Contract

Every memory system must implement the Bench'd Adapter Interface:

```python
class BaseAdapter(ABC):
    def setup(self) -> None
    def reset(self) -> None
    def ingest(self, turns: list[dict]) -> None
    def recall(self, query: str) -> str
    def teardown(self) -> None
```

### Turn Format

```json
{
  "role": "user" | "assistant" | "system",
  "content": "string",
  "timestamp": "ISO 8601 (optional)",
  "metadata": {} (optional)
}
```

### Recall Response

The `recall()` method returns a plain string. The harness handles answer generation and scoring separately.

## 2. Benchmark Datasets

### Registered Benchmarks

| Slug | Name | Version | Questions | Dimensions |
|------|------|---------|-----------|------------|
| `longmemeval-v1` | LongMemEval | 1.0 | 500 | recall, temporal, reasoning |
| `locomo-v1` | LoCoMo | 1.0 | 1,540 | recall, temporal, reasoning |
| `reliability-v1` | Reliability | 1.0 | 25 | hallucination, stale_memory, entity_confusion, deletion |
| `smoke-memory-v0` | Smoke | 0.1 | 10 | recall, temporal, reasoning |

### Dataset Integrity

Each benchmark dataset has a fixed hash. The harness verifies dataset integrity before every run. Dataset changes require a new version number.

## 3. Run Pipeline

Every official run follows this exact sequence:

```
1. Provision clean environment
2. Install adapter and system
3. Run healthcheck (adapter.setup())
4. For each benchmark question:
   a. Reset memory state (adapter.reset())
   b. Ingest conversation turns (adapter.ingest(turns))
   c. Execute recall query (adapter.recall(query))
   d. Record raw recall output
   e. Generate answer using locked answerer model
   f. Score against expected answer
   g. Record trace (input, output, score, reasoning)
5. Compute aggregate scores
6. Generate failure traces
7. Sign manifest with Ed25519
8. Publish receipt
```

## 4. Scoring Dimensions

### Accuracy Family

| Dimension | What it measures | Scoring method |
|-----------|-----------------|----------------|
| Recall | Can it retrieve specific facts? | Exact match + containment |
| Temporal | Does it understand time order and updates? | Regex + pattern matching |
| Reasoning | Can it synthesize across memories? | LLM judge |

### Reliability Family

| Dimension | What it measures | Scoring method |
|-----------|-----------------|----------------|
| Hallucination resistance | Says "I don't know" when it should | Trap question scoring |
| Stale memory handling | Uses latest version of changed facts | Trap question scoring |
| Entity confusion | Doesn't mix up similar entities | Trap question scoring |
| Deletion compliance | Forgets when told to forget | Trap question scoring |

### Efficiency Metrics

| Metric | What it measures |
|--------|-----------------|
| Avg latency (ms) | Time per recall query |
| Tokens per correct answer | Token cost per correct answer |
| Avg recall tokens | Tokens returned per retrieval |

### Bench'd Memory Index (BMI)

```
BMI = (0.70 × Accuracy) + (0.30 × Efficiency)
```

Where:
- Accuracy = overall verified score (0-100)
- Efficiency = 100 - min(tokens_per_correct / 100, 100)

BMI version: 1.0. Weight changes require a new version.

## 5. Model Locking

Every official run locks:

| Component | Locked Value |
|-----------|-------------|
| Answerer model | openai/gpt-4o-mini |
| Judge model | openai/gpt-4o-mini |
| Temperature | 0.0 |
| Answerer prompt | Versioned (v1.0) |
| Judge prompt | Versioned (v1.0) |

Changing any model or prompt requires a new benchmark version.

## 6. Trust Tiers

| Tier | Who runs it | Infrastructure | Signing |
|------|------------|----------------|---------|
| Listed | Nobody yet | N/A | N/A |
| Self-Reported | Vendor claims | Vendor's infra | No Bench'd signature |
| Community-Verified | Bench'd | Bench'd infra | Bench'd key |
| Vendor-Verified | Bench'd | Vendor's endpoint | Bench'd + vendor keys |
| Partner-Audited | Bench'd + auditor | Controlled environment | Triple-signed |

## 7. Manifest Format

Every run produces a signed JSON manifest:

```json
{
  "manifest": {
    "version": "1.0.0",
    "run_id": "run_xxxxxxxxxxxx",
    "system": { "name": "...", "adapter": "...", "version": "..." },
    "benchmark": { "slug": "...", "name": "...", "version": "..." },
    "harness": { "version": "0.2.0" },
    "protocol": { "version": "0.1.0" },
    "models": {
      "answerer": "openai/gpt-4o-mini",
      "judge": "openai/gpt-4o-mini",
      "temperature": 0.0
    },
    "scores": { ... },
    "traces": [ ... ]
  },
  "manifest_hash": "sha256:...",
  "signature": "ed25519:...",
  "public_key": "hex:...",
  "signing_key_fingerprint": "...",
  "signed_at": "ISO 8601",
  "signing_mode": "local | verifiedstate"
}
```

## 8. Versioning Rules

- Dataset changes → new benchmark version
- Judge/answerer model changes → new benchmark version
- Scoring weight changes → new BMI version
- Adapter contract changes → new protocol version
- Historical scores are never rewritten
- Version changes are recorded in CHANGELOG.md

## 9. Fairness Rules

1. Same protocol for every official run
2. Same answerer and judge per benchmark version
3. Same recall budget (no vendor-specific shortcuts)
4. Full trace preservation (every input/output recorded)
5. Vendor disputes are visible but cannot overwrite scores
6. Score changes require a rerun, not manual edits
7. Official results always include signed receipts

## 10. Dispute Process

1. Vendor identifies a specific trace they believe is scored unfairly
2. Bench'd reviews the trace (input, recall, answer, expected, reasoning)
3. If the scoring method is flawed, the benchmark version is updated
4. All systems are re-scored under the new version
5. No system gets special treatment

---

**Protocol version:** 0.1.0
**Effective date:** 2026-05-13
**Maintained by:** Bench'd (benchd.ai)
