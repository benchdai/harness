# Bench'd Independent Benchmark Results for {{SYSTEM_NAME}}

Hi team! We're [Bench'd](https://benchd.ai) — an independent benchmark platform for AI memory systems.

We ran **{{SYSTEM_NAME}}** through our LongMemEval benchmark (500 questions testing recall, temporal reasoning, and knowledge updates) using our [open-source harness](https://github.com/benchdai/harness).

## Results

| Benchmark | Score | Questions | Status |
|-----------|-------|-----------|--------|
| LongMemEval v1.0 | **{{SCORE}}%** | 500 | Verified |

Full results and methodology: [benchd.ai/system/{{SLUG}}](https://benchd.ai/system/{{SLUG}})

## Context

- All systems are tested under identical conditions (same LLM judge, same questions, same scoring)
- A plain GPT-4o-mini with no memory layer scores **57.6%** as the baseline
- Every run is cryptographically signed and publicly verifiable
- The harness is fully open source: [github.com/benchdai/harness](https://github.com/benchdai/harness)

## Run it yourself

```bash
pip install benchd-harness
benchd run -a {{ADAPTER}} -b longmemeval-v1 --judge --key ./keys/private.key
```

## Claim your profile

If you'd like to run an official vendor-verified benchmark:
1. Visit [benchd.ai/claim](https://benchd.ai/claim)
2. Or run the harness and submit at [benchd.ai/submit](https://benchd.ai/submit)

We're happy to work with you on adapter improvements. The goal is fair, reproducible comparison — not gotchas.

---

*This issue was created by the [Bench'd](https://benchd.ai) team. We're building the neutral benchmark standard for AI memory systems.*
