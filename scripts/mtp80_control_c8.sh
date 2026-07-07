#!/usr/bin/env bash
# Control: non-MTP concurrent c=8 (expect ~81 tok/s) vs MTP ceiling
set -euo pipefail
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
source /opt/conda/etc/profile.d/conda.sh
conda activate base
REPO=/data/metaX-inference
LOG=/data/metax-test-logs/tune/mtp80/control-c8.md
P_LONG="请用中文写一段约120字的自我介绍，不要换行。"
mkdir -p /data/metax-test-logs/tune/mtp80

start_vllm() {
  local extra=("$@")
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
  nohup vllm serve /data/models/Qwen3.6-27B-AWQ --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --max-model-len 8192 --dtype auto \
    --gpu-memory-utilization 0.97 --max-num-batched-tokens 16384 --max-num-seqs 128 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    "${extra[@]}" > /tmp/vllm-ctrl.log 2>&1 &
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
}

bench() {
  local label="$1" prompt="$2" conc="$3"
  echo "## $label c=$conc" | tee -a "$LOG"
  python "$REPO/scripts/bench_qwen36.py" --url http://127.0.0.1:8000 \
    --prompt "$prompt" --max-tokens 128 --temperature 0 --warmup-requests 1 \
    --concurrency "$conc" --requests "$conc" --stream --json 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

echo "# MTP80 control — $(date -Iseconds)" > "$LOG"
start_vllm
bench "no-MTP long prompt" "$P_LONG" 8
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
start_vllm --compilation-config '{"cudagraph_mode":"none"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --reasoning-parser qwen3
bench "MTP-2 long prompt" "$P_LONG" 8
pkill -f "vllm serve" 2>/dev/null || true
echo done | tee -a "$LOG"
