#!/usr/bin/env bash
set -euo pipefail

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="/opt/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.92}"
MAX_SEQS="${MAX_SEQS:-64}"
MAX_BATCHED="${MAX_BATCHED:-8192}"

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  VLLM_API_KEY="sk-unsloth-metax-$(head -c 32 /dev/urandom | xxd -p -c 64)"
  export VLLM_API_KEY
fi

echo "[metax-vllm] model=$MODEL listen=$HOST:$PORT"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code
