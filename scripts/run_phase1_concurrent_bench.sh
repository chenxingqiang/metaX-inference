#!/usr/bin/env bash
# Phase 1 concurrent batch benchmark (AGENT.md §12.5 — target >40 tok/s @ 8 req)
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/phase1}"
REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
RESULT="$LOG_DIR/PHASE1_CONCURRENT_BENCH.md"
PROMPT="${PROMPT:-请用中文写一段约120字的自我介绍，不要换行。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
BENCH_API="${BENCH_API:-completions}"
BENCH_NO_THINK="${BENCH_NO_THINK:-0}"

mkdir -p "$LOG_DIR"

{
  echo "# Phase 1 Concurrent Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Model: $MODEL"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo "- Prompt: $PROMPT"
  echo "- max_tokens: $MAX_TOKENS"
  echo "- temperature: $TEMPERATURE"
  echo "- api: $BENCH_API"
  echo ""
} > "$RESULT"

start_vllm() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
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
    > "$LOG_DIR/vllm-concurrent-${RANDOM}.log" 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed to start" | tee -a "$RESULT"
  return 1
}

run_concurrent() {
  local label="$1"
  local concurrency="$2"
  echo "## $label (concurrency=$concurrency)" | tee -a "$RESULT"
  local extra=()
  if [[ "$BENCH_NO_THINK" == "1" ]]; then
    extra+=(--no-think)
  fi
  python "$REPO_DIR/scripts/bench_qwen36.py" \
    --url "http://${HOST}:${PORT}" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --api "$BENCH_API" \
    "${extra[@]}" \
    --concurrency "$concurrency" \
    --requests "$concurrency" \
    --stream \
    --json 2>&1 | tee -a "$RESULT"
  echo "" | tee -a "$RESULT"
}

start_vllm

run_concurrent "Single request baseline" 1
run_concurrent "Concurrent x4" 4
run_concurrent "Concurrent x8 (Phase 1 target)" 8

pkill -f "vllm serve" 2>/dev/null || true
echo "Phase 1 concurrent sweep done." | tee -a "$RESULT"
