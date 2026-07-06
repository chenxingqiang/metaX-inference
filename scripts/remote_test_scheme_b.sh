#!/bin/bash
# Remote end-to-end test on MetaX server (Scheme B primary)
set -euo pipefail
export PS4='+ [${SECONDS}s] '
set -x

source /opt/conda/etc/profile.d/conda.sh
conda activate base
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=$HF_HOME/hub
cd /data

LOG_DIR=/data/metax-test-logs
mkdir -p "$LOG_DIR"
RESULT_FILE="$LOG_DIR/TEST_RESULTS.md"

MODEL_ID="${MODEL_ID:-QuantTrio/Qwen3.6-27B-AWQ}"
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

{
  echo "# MetaX Server Test Results"
  echo ""
  echo "- Date: $(date -Iseconds)"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 'MetaX' || echo unknown)"
  echo "- Model: $MODEL_ID"
  echo ""
} > "$RESULT_FILE"

echo "=== E01-E05 Environment ===" | tee -a "$RESULT_FILE"
mx-smi 2>/dev/null | head -20 | tee -a "$RESULT_FILE" || true
python --version | tee -a "$RESULT_FILE"
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$RESULT_FILE"
python -c "import vllm_metax; import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee -a "$RESULT_FILE"

echo "=== Download model ===" | tee -a "$RESULT_FILE"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  mkdir -p "$MODEL_DIR"
  huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False 2>&1 | tee -a "$LOG_DIR/download.log"
else
  echo "Model already at $MODEL_DIR" | tee -a "$RESULT_FILE"
fi
ls -lh "$MODEL_DIR" | head -20 | tee -a "$RESULT_FILE" || true

echo "=== Start vLLM server ===" | tee -a "$RESULT_FILE"
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

nohup vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --trust-remote-code \
  > "$LOG_DIR/vllm-server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID" | tee -a "$RESULT_FILE"

READY=0
for i in $(seq 1 180); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 || \
     curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early" | tee -a "$RESULT_FILE"
    tail -80 "$LOG_DIR/vllm-server.log" | tee -a "$RESULT_FILE" || true
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "vLLM not ready in time" | tee -a "$RESULT_FILE"
  tail -80 "$LOG_DIR/vllm-server.log" | tee -a "$RESULT_FILE"
  exit 1
fi

echo "=== Inference test ===" | tee -a "$RESULT_FILE"
RESP=$(curl -sf "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好，我是","max_tokens":64,"temperature":0.7}')
echo "$RESP" | tee -a "$RESULT_FILE"

if echo "$RESP" | grep -q '"choices"'; then
  echo "PASS B03: vLLM completion OK" | tee -a "$RESULT_FILE"
else
  echo "FAIL B03" | tee -a "$RESULT_FILE"
  exit 1
fi

mx-smi 2>/dev/null | head -25 | tee -a "$RESULT_FILE" || true
echo "Scheme B test completed successfully." | tee -a "$RESULT_FILE"

kill "$VLLM_PID" 2>/dev/null || true
