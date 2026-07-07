#!/usr/bin/env bash
# Retest best c8160 config (high-mem-batch16k, c=18) — target 160 tok/s aggregate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/tune/c8160-retest}"
PROMPT="${PROMPT:-请用中文写一段约120字的自我介绍，不要换行。}"
CONCURRENCY="${CONCURRENCY:-18}"
RUNS="${RUNS:-3}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

pkill -f "vllm serve" 2>/dev/null || true
sleep 3

echo "Starting vLLM high-mem-batch16k ..."
nohup vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype auto \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  > "$LOG_DIR/vllm.log" 2>&1 &

for i in $(seq 1 180); do
  curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 5
done

{
  echo "# C8160 Best Config Retest — $(date -Iseconds)"
  echo ""
  echo "- vLLM: gpu_mem=0.95 max_seqs=128 max_batched=16384"
  echo "- concurrency: $CONCURRENCY"
  echo "- prompt: $PROMPT"
  echo ""
} > "$LOG_DIR/RETEST.md"

for r in $(seq 1 "$RUNS"); do
  echo "## Run $r" | tee -a "$LOG_DIR/RETEST.md"
  python scripts/bench_qwen36.py \
    --url "http://${HOST}:${PORT}" \
    --prompt "$PROMPT" \
    --max-tokens 128 \
    --temperature 0 \
    --warmup-requests 1 \
    --concurrency "$CONCURRENCY" \
    --requests "$CONCURRENCY" \
    --stream --json 2>&1 | tee -a "$LOG_DIR/RETEST.md"
  echo "" | tee -a "$LOG_DIR/RETEST.md"
done

pkill -f "vllm serve" 2>/dev/null || true
echo "Retest done: $LOG_DIR/RETEST.md"
