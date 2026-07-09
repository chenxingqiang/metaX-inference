#!/usr/bin/env bash
# Benchmark BF16 MTP checkpoint vs AWQ baseline (Phase 3 + 80 tok/s target)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ-MTP-BF16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/mtp-bf16}"
RESULT="$LOG_DIR/MTP_BF16_BENCH.md"
MTP_TOKENS="${MTP_TOKENS:-2}"
PROMPT="${PROMPT:-你好，请用一句话介绍你自己。}"
PROMPT_LONG="${PROMPT_LONG:-请用中文写一段约120字的自我介绍，不要换行。}"
COMP_NONE='{"cudagraph_mode":"none"}'

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: model not found at $MODEL — run scripts/download_qwen36_mtp_bf16.sh first" >&2
  echo "  (grafts BF16 mtp.* from hampsonw onto MetaX-compatible AWQ base)" >&2
  exit 1
fi

{
  echo "# BF16 MTP Checkpoint Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Model: $MODEL"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo "- MTP num_speculative_tokens: $MTP_TOKENS"
  echo ""
} > "$RESULT"

start_vllm() {
  local label="$1" use_cg="$2"
  shift 2
  echo "Starting vLLM: $label" | tee -a "$RESULT"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  local cg=()
  [[ "$use_cg" == "1" ]] && cg=(--compilation-config "$COMP_NONE")
  # shellcheck disable=SC2068
  nohup vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --dtype auto \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    "${cg[@]}" \
    "$@" \
    > "$LOG_DIR/vllm-${label}-${RANDOM}.log" 2>&1 &
  for i in $(seq 1 180); do
    curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "vLLM failed ($label)" | tee -a "$RESULT"
  tail -30 "$LOG_DIR"/vllm-${label}-*.log 2>/dev/null | tee -a "$RESULT" || true
  return 1
}

bench() {
  local label="$1" prompt="$2" conc="$3"
  echo "## $label (c=$conc)" | tee -a "$RESULT"
  python scripts/bench_qwen36.py \
    --url "http://${HOST}:${PORT}" \
    --prompt "$prompt" \
    --max-tokens 128 \
    --temperature 0 \
    --warmup-requests 1 \
    --concurrency "$conc" \
    --requests "$conc" \
    --stream --json 2>&1 | tee -a "$RESULT"
  echo "" | tee -a "$RESULT"
}

echo "=== Pre-check MTP head ===" | tee -a "$RESULT"
PYTHONPATH=. python scripts/check_mtp_head.py "$MODEL" --json 2>&1 | tee -a "$RESULT"
echo "" | tee -a "$RESULT"

# Baseline no speculative
if start_vllm "baseline" 0; then
  bench "Baseline short c=1" "$PROMPT" 1
  bench "Baseline long c=8" "$PROMPT_LONG" 8
fi

# MTP speculative
SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
if start_vllm "mtp" 1 --speculative-config "$SPEC" --reasoning-parser qwen3; then
  bench "MTP short c=1" "$PROMPT" 1
  bench "MTP short c=8" "$PROMPT" 8
  bench "MTP long c=8" "$PROMPT_LONG" 8
fi

pkill -f "vllm serve" 2>/dev/null || true
echo "BF16 MTP benchmark done." | tee -a "$RESULT"
