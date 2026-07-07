#!/usr/bin/env bash
# Quick Phase 3 retest with vLLM warmup (1 discard req before measure)
set -euo pipefail
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
source /opt/conda/etc/profile.d/conda.sh
conda activate base
REPO=/data/metaX-inference
LOG=/data/metax-test-logs/tune/phase3-warmup.md
mkdir -p /data/metax-test-logs/tune
P="你好，请用一句话介绍你自己。"
echo "# Phase 3 warmup retest — $(date -Iseconds)" > "$LOG"

run_mode() {
  local label="$1" spec="$2" no_cg="$3"
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  local extra=()
  [[ -n "$spec" ]] && extra+=(--speculative-config "$spec")
  [[ "$label" == mtp* ]] && extra+=(--reasoning-parser qwen3)
  [[ "$no_cg" == "1" ]] && extra+=(--compilation-config '{"cudagraph_mode":"none"}')
  nohup vllm serve /data/models/Qwen3.6-27B-AWQ --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --max-model-len 8192 --dtype auto \
    --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 64 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    "${extra[@]}" > /tmp/vllm-p3-$label.log 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
  echo "## $label" | tee -a "$LOG"
  python "$REPO/scripts/bench_qwen36.py" --url http://127.0.0.1:8000 \
    --prompt "$P" --max-tokens 128 --temperature 0 --warmup-requests 1 \
    --concurrency 1 --stream --json 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

run_mode baseline "" 0
run_mode ngram-8 '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}' 1
run_mode mtp-2 '{"method":"mtp","num_speculative_tokens":2}' 1
pkill -f "vllm serve" 2>/dev/null || true
echo "done" | tee -a "$LOG"
