#!/usr/bin/env bash
# Stop vLLM deployed for Unsloth Studio.
set -euo pipefail

LOG_DIR="${LOG_DIR:-/data/metax-test-logs/unsloth-studio}"
PID_FILE="${PID_FILE:-$LOG_DIR/vllm.pid}"
PORT="${PORT:-8000}"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping vLLM pid=$pid"
    kill "$pid" 2>/dev/null || true
    exit 0
  fi
fi

pkill -f "vllm serve.*--port $PORT" 2>/dev/null || pkill -f "vllm serve.* $PORT" 2>/dev/null || true
echo "Stopped (if running)."
