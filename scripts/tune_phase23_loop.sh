#!/usr/bin/env bash
# Phase 2 op + Phase 3 speculative tuning loops
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs/tune/phase23}"
PHASE2_LOOPS="${PHASE2_LOOPS:-12}"
PHASE3_LOOPS="${PHASE3_LOOPS:-5}"

mkdir -p "$LOG_ROOT"
cd "$REPO_DIR"

exec "$PYTHON" scripts/tune_phase23_loop.py \
  --repo "$REPO_DIR" \
  --log-root "$LOG_ROOT" \
  --phase2-loops "$PHASE2_LOOPS" \
  --phase3-loops "$PHASE3_LOOPS" \
  "$@"
