#!/usr/bin/env bash
# Phase 3 MTP / speculative decoding benchmark (AGENT.md §12.3)
# Compares baseline vs MTP vs ngram fallback on MetaX C500.
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
PROMPT="${PROMPT:-你好，请用一句话介绍你自己。}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MTP_TOKENS="${MTP_TOKENS:-2}"
# Speculative decode + CUDA graph capture breaks Triton autotune on MACA
# (maca_fused_recurrent_gated_delta_rule: "operation not permitted when stream is capturing").
DISABLE_CUDAGRAPH="${DISABLE_CUDAGRAPH:-1}"

mkdir -p "$LOG_DIR"

{
  echo "# Phase 3 MTP Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Model: $MODEL"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo "- MTP num_speculative_tokens: $MTP_TOKENS"
  echo ""
  echo "> AWQ 量化模型可能缺少可用 MTP head（draft acceptance 0%）。"
  echo "> 若 MTP 无提升，请换带 \`mtp.*\` BF16 权重的 checkpoint。"
  echo "> DISABLE_CUDAGRAPH=$DISABLE_CUDAGRAPH (workaround for MACA Triton autotune + cudagraph capture)."
  echo ""
} > "$RESULT"

cudagraph_args() {
  if [[ "$DISABLE_CUDAGRAPH" == "1" ]]; then
    echo --max-cudagraph-capture-size 0
  fi
}

start_vllm() {
  local label="$1"
  shift
  echo "Starting vLLM: $label" | tee -a "$RESULT"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  # shellcheck disable=SC2046,SC2068
  nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --trust-remote-code \
    $(cudagraph_args) \
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
    --concurrency 1 \
    --stream \
    --json 2>&1 | tee -a "$RESULT"
  echo "" | tee -a "$RESULT"
}

# Baseline (no speculative)
start_vllm "baseline"
run_bench "Baseline (no speculative)"

# MTP native head
SPEC_MTP="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
if start_vllm "mtp" --speculative-config "$SPEC_MTP" --reasoning-parser qwen3; then
  run_bench "MTP (num_speculative_tokens=${MTP_TOKENS})"
fi

# ngram fallback (works without MTP weights)
SPEC_NGRAM='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
if start_vllm "ngram" --speculative-config "$SPEC_NGRAM"; then
  run_bench "N-gram speculative (fallback)"
fi

pkill -f "vllm serve" 2>/dev/null || true
echo "Phase 3 MTP sweep done." | tee -a "$RESULT"
echo "Target: >20 tok/s equivalent (AGENT.md §12.5)" | tee -a "$RESULT"
