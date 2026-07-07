#!/usr/bin/env bash
# Best-effort MTP c=8 retest (aggressive vLLM, long startup wait)
set -euo pipefail
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
source /opt/conda/etc/profile.d/conda.sh
conda activate base
REPO=/data/metaX-inference
LOG=/data/metax-test-logs/tune/mtp80/best-mtp-c8.md
P_SHORT="你好，请用一句话介绍你自己。"
echo "# Best MTP c=8 retest — $(date -Iseconds)" > "$LOG"
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
nohup vllm serve /data/models/Qwen3.6-27B-AWQ --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --dtype auto \
  --gpu-memory-utilization 0.97 --max-num-batched-tokens 16384 --max-num-seqs 128 \
  --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"none"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3 > /tmp/vllm-mtp-best.log 2>&1 &
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo ready && break; sleep 5; done
echo "## MTP-2 aggressive c=8 short prompt" | tee -a "$LOG"
python "$REPO/scripts/bench_qwen36.py" --url http://127.0.0.1:8000 \
  --prompt "$P_SHORT" --max-tokens 128 --temperature 0 --warmup-requests 1 \
  --concurrency 8 --requests 8 --stream --json 2>&1 | tee -a "$LOG"
pkill -f "vllm serve" 2>/dev/null || true
