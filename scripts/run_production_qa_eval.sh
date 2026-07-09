#!/usr/bin/env bash
# Production Q&A quality eval — start vLLM (no MTP) and run eval_production_qa.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/production-qa}"
RESULT="$LOG_DIR/PRODUCTION_QA_EVAL.md"
API="${API:-chat}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

start_vllm() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  echo "Starting production vLLM (no MTP): $MODEL"
  nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    > "$LOG_DIR/vllm-serve.log" 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed to start" >&2
  tail -40 "$LOG_DIR/vllm-serve.log" >&2 || true
  return 1
}

if ! curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  start_vllm
fi

PYTHONPATH=. python scripts/eval_production_qa.py \
  --url "http://${HOST}:${PORT}" \
  --api "$API" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --no-think \
  --warmup \
  --output "$RESULT" | tee "$LOG_DIR/eval-run.log"

echo ""
echo "Report: $RESULT"
