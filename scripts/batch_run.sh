#!/bin/bash
set -e

# Bench'd batch runner — runs all available adapters against longmemeval-v1.
# Skips adapters that need infrastructure we don't have locally.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ---------- env vars ----------
[ -f "$HOME/.env.benchd" ] && source "$HOME/.env.benchd"
[ -f "$HOME/.env" ] && source "$HOME/.env"

# ---------- venv ----------
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "Error: .venv not found. Run: python -m venv .venv && pip install -e ." >&2
  exit 1
fi

# ---------- signing key ----------
KEY_PATH="./keys/private.key"
if [ ! -f "$KEY_PATH" ]; then
  echo "No signing key found at $KEY_PATH — generating one..."
  benchd keys generate --out ./keys
fi

# ---------- config ----------
BENCHMARK="longmemeval-v1"
MAX_ITEMS=50
OUT_DIR="./runs"
mkdir -p "$OUT_DIR"

# Adapters to run. Skipping letta, zep, graphiti (need external infra).
ADAPTERS="llm-baseline mem0-local cognee langchain-memory llamaindex-memory verifiedstate"

PASSED=0
FAILED=0
SKIPPED=0
RESULTS=""

# ---------- run loop ----------
for adapter in $ADAPTERS; do
  echo ""
  echo "=============================="
  echo "Running: $adapter"
  echo "=============================="

  START_TIME=$(date +%s)

  if benchd run \
    --adapter "$adapter" \
    --benchmark "$BENCHMARK" \
    --max-items "$MAX_ITEMS" \
    --judge \
    --out "$OUT_DIR" \
    --key "$KEY_PATH"; then

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    RESULTS="$RESULTS\n  ✓ $adapter  (${ELAPSED}s)"
    PASSED=$((PASSED + 1))
  else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    RESULTS="$RESULTS\n  ✗ $adapter  FAILED (${ELAPSED}s)"
    FAILED=$((FAILED + 1))
  fi

  echo "---"
done

# ---------- summary ----------
TOTAL=$((PASSED + FAILED))
echo ""
echo "=============================="
echo "BATCH RUN COMPLETE"
echo "=============================="
echo "  Benchmark: $BENCHMARK (max $MAX_ITEMS items)"
echo "  Adapters:  $TOTAL run, $PASSED succeeded, $FAILED failed"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Run data saved to: $OUT_DIR/"
echo "Next: python scripts/summarize_results.py $OUT_DIR/"
