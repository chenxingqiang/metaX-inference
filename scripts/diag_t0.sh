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
  --gpu-memory-utilization 0.92 --enable-chunked-prefill --trust-remote-code \
  > /tmp/vllm-curl.log 2>&1 &
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
echo "=== completions t=0 non-stream ==="
curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"你好","max_tokens":16,"temperature":0,"stream":false}' > /tmp/c0.json
python3 -c 'import json; d=json.load(open("/tmp/c0.json")); print("usage", d.get("usage")); print("text", repr(d["choices"][0]["text"][:100]))'
echo "=== chat no-think stream ==="
cd /data/metaX-inference
python scripts/bench_qwen36.py --url http://127.0.0.1:8000 --api chat --no-think \
  --prompt "你好，请用一句话介绍你自己。" --max-tokens 128 --temperature 0 --stream --json \
  > /tmp/chat0.json
python3 -c 'import json; s=json.load(open("/tmp/chat0.json"))["summary"]; print("tok/s", s["aggregate_tokens_per_s"], "ttft", s.get("ttft_s"), "tokens", s["total_completion_tokens"])'
pkill -f "vllm serve" 2>/dev/null || true
