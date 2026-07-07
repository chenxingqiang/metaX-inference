#!/usr/bin/env bash
# Automated tuning loop toward AGENT.md §12.5 targets (Phase 0 + Phase 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs/tune}"
MAX_LOOPS="${MAX_LOOPS:-8}"

mkdir -p "$LOG_ROOT"
cd "$REPO_DIR"
echo "  repo=$REPO_DIR"
echo "  log=$LOG_ROOT"
echo "  max_loops=$MAX_LOOPS"
echo ""

exec "$PYTHON" scripts/tune_targets_loop.py \
  --repo "$REPO_DIR" \
  --log-root "$LOG_ROOT" \
  --max-loops "$MAX_LOOPS" \
  "$@"
