#!/usr/bin/env bash
# Production vLLM serve for Qwen3.6-27B-AWQ on MetaX C500 (32GB)
# Usage:
#   ./scripts/serve_qwen36_metax.sh
#   METAX_KERNELS=1 ./scripts/serve_qwen36_metax.sh
#   ENABLE_MTP=1 MTP_TOKENS=2 ./scripts/serve_qwen36_metax.sh
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
MTP_GRAFTED="${MTP_GRAFTED:-/data/models/Qwen3.6-27B-AWQ-MTP-BF16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.92}"
MAX_SEQS="${MAX_SEQS:-64}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
METAX_KERNELS="${METAX_KERNELS:-0}"
METAX_KERNEL_IMPL="${METAX_KERNEL_IMPL:-fused}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_TOKENS="${MTP_TOKENS:-2}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
# Disable CUDA graphs when speculative decode is on (MACA Triton autotune vs cudagraph capture).
DISABLE_CUDAGRAPH="${DISABLE_CUDAGRAPH:-$([[ "$ENABLE_MTP" == "1" ]] && echo 1 || echo 0)}"

EXTRA=()
if [[ "$ENABLE_MTP" == "1" ]]; then
  if [[ -f "$MTP_GRAFTED/config.json" && "$MODEL" == "/data/models/Qwen3.6-27B-AWQ" ]]; then
    MODEL="$MTP_GRAFTED"
    echo "ENABLE_MTP=1 — using grafted BF16 MTP checkpoint: $MODEL"
  fi
  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
  EXTRA+=(--reasoning-parser qwen3)
elif [[ "$PREFIX_CACHE" == "1" ]]; then
  EXTRA+=(--enable-prefix-caching)
fi
if [[ "$DISABLE_CUDAGRAPH" == "1" ]]; then
  EXTRA+=(--compilation-config '{"cudagraph_mode":"none"}')
fi

if [[ "$METAX_KERNELS" == "1" ]]; then
  export METAX_KERNELS=1
  export METAX_KERNEL_IMPL="$METAX_KERNEL_IMPL"
  export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"
  python -c "import engine.vllm_metax_plugin.loader" 2>/dev/null || true
fi

echo "Starting vLLM: model=$MODEL host=$HOST port=$PORT METAX_KERNELS=$METAX_KERNELS ENABLE_MTP=$ENABLE_MTP"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-num-seqs "$MAX_SEQS" \
  --enable-chunked-prefill \
  --trust-remote-code \
  "${EXTRA[@]}"
