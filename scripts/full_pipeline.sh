#!/bin/bash
set -e

# Full Bench'd pipeline: batch run -> summarize -> export to frontend.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Step 1: Batch run ==="
./scripts/batch_run.sh

echo ""
echo "=== Step 2: Summarize results ==="
python scripts/summarize_results.py ./runs/

echo ""
echo "=== Step 3: Export to frontend ==="
python scripts/export_to_frontend.py ./runs/ --output-dir ../benchd/lib/data/generated/

echo ""
echo "Done. Restart the frontend dev server to see real data."
