#!/usr/bin/env bash
set -euo pipefail
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
source /opt/conda/etc/profile.d/conda.sh
conda activate base
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
nohup vllm serve /data/models/Qwen3.6-27B-AWQ --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 8192 --dtype auto \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  > /tmp/vllm-warm.log 2>&1 &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
cd /data/metaX-inference
P="你好，请用一句话介绍你自己。"
echo "=== 1st: t0.7 ==="
python scripts/bench_qwen36.py --url http://127.0.0.1:8000 --prompt "$P" --max-tokens 128 --temperature 0.7 --stream --json | python3 -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(s["aggregate_tokens_per_s"], s.get("ttft_s"))'
echo "=== 2nd: t0 ==="
python scripts/bench_qwen36.py --url http://127.0.0.1:8000 --prompt "$P" --max-tokens 128 --temperature 0 --stream --json | python3 -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(s["aggregate_tokens_per_s"], s.get("ttft_s"))'
echo "=== 3rd: t0 again ==="
python scripts/bench_qwen36.py --url http://127.0.0.1:8000 --prompt "$P" --max-tokens 128 --temperature 0 --stream --json | python3 -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(s["aggregate_tokens_per_s"], s.get("ttft_s"))'
pkill -f "vllm serve" 2>/dev/null || true
