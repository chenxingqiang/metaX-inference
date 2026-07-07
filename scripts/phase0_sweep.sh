#!/usr/bin/env bash
# Phase 0 single-request tuning toward >=9.5 tok/s
set -euo pipefail
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
export HF_ENDPOINT=https://hf-mirror.com
source /opt/conda/etc/profile.d/conda.sh
conda activate base

REPO=/data/metaX-inference
MODEL=/data/models/Qwen3.6-27B-AWQ
LOG=/data/metax-test-logs/tune/phase0-sweep.md
mkdir -p /data/metax-test-logs/tune

pkill -f "vllm serve" 2>/dev/null || true
sleep 3
nohup vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --dtype auto \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 \
  --max-num-seqs 64 --enable-chunked-prefill --enable-prefix-caching \
  --trust-remote-code > /tmp/vllm-p0.log 2>&1 &

for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done

{
  echo "# Phase 0 single-request sweep — $(date -Iseconds)"
  echo ""
} > "$LOG"

run_case() {
  local label="$1"; shift
  echo "## $label" | tee -a "$LOG"
  python "$REPO/scripts/bench_qwen36.py" --url http://127.0.0.1:8000 "$@" \
    --concurrency 1 --stream --json 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

run_case "default prompt t0.7 max128" --prompt "你好，请用一句话介绍你自己。" --max-tokens 128 --temperature 0.7
run_case "default prompt t0 max128" --prompt "你好，请用一句话介绍你自己。" --max-tokens 128 --temperature 0
run_case "default prompt t0 max64" --prompt "你好，请用一句话介绍你自己。" --max-tokens 64 --temperature 0
run_case "short prompt t0 max128" --prompt "你好" --max-tokens 128 --temperature 0
run_case "chat no-think max128" --api chat --no-think --prompt "你好，请用一句话介绍你自己。" --max-tokens 128 --temperature 0

pkill -f "vllm serve" 2>/dev/null || true
echo "Phase 0 sweep done." | tee -a "$LOG"
