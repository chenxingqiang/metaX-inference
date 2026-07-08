#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/trustclaw}"
PID_FILE="${PID_FILE:-$LOG_DIR/gateway.pid}"
PORT="${TRUSTCLAW_PORT:-19001}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "Stopped TrustClaw pid=$pid"
    exit 0
  fi
fi
pkill -f "openclaw.mjs gateway.*$PORT" 2>/dev/null || true
echo "Stopped (if running)."
