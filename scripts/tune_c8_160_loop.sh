#!/usr/bin/env bash
# Tune no-MTP aggregate throughput toward 160 tok/s (c=8..32 grid)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
cd "$REPO_DIR"

export C8_TARGET="${C8_TARGET:-160}"
export MAX_LOOPS="${MAX_LOOPS:-20}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs/tune/c8160}"
mkdir -p "$LOG_ROOT"

{
  echo "=== tune_c8_160_loop ==="
  echo "  started: $(date -Iseconds)"
  echo "  target: ${C8_TARGET} tok/s aggregate"
  echo "  max_loops: ${MAX_LOOPS}"
  echo "  GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  mx-smi 2>/dev/null | sed -n '/Sliced GPU/,/Process/p' || true
  echo ""
} | tee "$LOG_ROOT/run.log"

PYTHONPATH=. "$PYTHON" scripts/tune_c8_160_loop.py \
  --repo "$REPO_DIR" \
  --log-root "$LOG_ROOT" \
  "$@" 2>&1 | tee -a "$LOG_ROOT/run.log"

echo "done $(date -Iseconds)" | tee -a "$LOG_ROOT/run.log"
