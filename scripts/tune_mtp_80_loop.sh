#!/usr/bin/env bash
# MTP speculative tuning toward 80 tok/s aggregate throughput
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs/tune/mtp80}"
MTP_TARGET="${MTP_TARGET:-80}"
MAX_LOOPS="${MAX_LOOPS:-12}"

mkdir -p "$LOG_ROOT"
cd "$REPO_DIR"

echo "=== tune_mtp_80_loop target=${MTP_TARGET} tok/s max_loops=${MAX_LOOPS} ==="
exec "$PYTHON" scripts/tune_mtp_80_loop.py \
  --repo "$REPO_DIR" \
  --log-root "$LOG_ROOT" \
  --target "$MTP_TARGET" \
  --max-loops "$MAX_LOOPS" \
  "$@"
