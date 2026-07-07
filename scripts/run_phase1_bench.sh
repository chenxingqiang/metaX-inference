#!/usr/bin/env bash
# Phase 1 vLLM parameter sweep + E2E benchmark (AGENT.md §12)
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
RESULT="$LOG_DIR/PHASE1_BENCH.md"

mkdir -p "$LOG_DIR"

{
  echo "# Phase 1 Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Model: $MODEL"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo ""
} > "$RESULT"

run_bench() {
  local label="$1"
  shift
  echo "## $label" | tee -a "$RESULT"
  if [[ -f "$REPO_DIR/scripts/bench_qwen36.py" ]]; then
    python "$REPO_DIR/scripts/bench_qwen36.py" --url "http://${HOST}:${PORT}" "$@" 2>&1 | tee -a "$RESULT"
  else
    curl -sf "http://${HOST}:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"prompt":"你好，请用一句话介绍你自己。","max_tokens":128,"temperature":0.7}' \
      2>&1 | tee -a "$RESULT"
  fi
  echo "" | tee -a "$RESULT"
}

start_vllm() {
  local extra_args="$1"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --trust-remote-code \
    $extra_args \
    > "$LOG_DIR/vllm-${RANDOM}.log" 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed to start" | tee -a "$RESULT"
  return 1
}

# Baseline (Phase 0 config)
start_vllm "--dtype auto"
run_bench "Baseline (default)"

# Phase 1 tuned
start_vllm "--dtype auto --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 64 --enable-chunked-prefill --enable-prefix-caching"
run_bench "Phase 1 tuned (chunked prefill + prefix cache)"

pkill -f "vllm serve" 2>/dev/null || true
echo "Phase 1 sweep done." | tee -a "$RESULT"
