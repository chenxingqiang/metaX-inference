#!/usr/bin/env bash
# Phase 3 MTP / speculative decoding benchmark (AGENT.md §12.3)
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/phase3}"
REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
RESULT="$LOG_DIR/PHASE3_MTP_BENCH.md"
PROMPT="${PROMPT:-请用中文写一段约120字的自我介绍，不要换行。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MTP_TOKENS="${MTP_TOKENS:-2}"
DISABLE_CUDAGRAPH="${DISABLE_CUDAGRAPH:-1}"
COMP_CONFIG_NONE='{"cudagraph_mode":"none"}'

mkdir -p "$LOG_DIR"

{
  echo "# Phase 3 MTP Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Model: $MODEL"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo "- MTP num_speculative_tokens: $MTP_TOKENS"
  echo "- temperature: $TEMPERATURE"
  echo "- DISABLE_CUDAGRAPH: $DISABLE_CUDAGRAPH"
  echo ""
} > "$RESULT"

start_vllm() {
  local label="$1"
  local use_no_cg="${2:-0}"
  shift 2
  echo "Starting vLLM: $label" | tee -a "$RESULT"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  local cg_args=()
  if [[ "$use_no_cg" == "1" && "$DISABLE_CUDAGRAPH" == "1" ]]; then
    cg_args=(--compilation-config "$COMP_CONFIG_NONE")
  fi
  # shellcheck disable=SC2068
  nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --trust-remote-code \
    "${cg_args[@]}" \
    "$@" \
    > "$LOG_DIR/vllm-phase3-${RANDOM}.log" 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed to start ($label)" | tee -a "$RESULT"
  tail -40 "$LOG_DIR"/vllm-phase3-*.log 2>/dev/null | tee -a "$RESULT" || true
  return 1
}

run_bench() {
  local label="$1"
  echo "## $label" | tee -a "$RESULT"
  python "$REPO_DIR/scripts/bench_qwen36.py" \
    --url "http://${HOST}:${PORT}" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --concurrency 1 \
    --stream \
    --json 2>&1 | tee -a "$RESULT"
  echo "" | tee -a "$RESULT"
}

start_vllm "baseline" 0
run_bench "Baseline (no speculative)"

SPEC_MTP="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
if start_vllm "mtp" 1 --speculative-config "$SPEC_MTP" --reasoning-parser qwen3; then
  run_bench "MTP (num_speculative_tokens=${MTP_TOKENS})"
fi

SPEC_NGRAM='{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}'
if start_vllm "ngram" 1 --speculative-config "$SPEC_NGRAM"; then
  run_bench "N-gram speculative (fallback)"
fi

pkill -f "vllm serve" 2>/dev/null || true
echo "Phase 3 MTP sweep done." | tee -a "$RESULT"
