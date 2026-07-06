#!/bin/bash
# Fix transformers + run vLLM inference after model download
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=$HF_HOME/hub
cd /data

LOG_DIR=/data/metax-test-logs
MODEL_DIR=/data/models/Qwen3.6-27B-AWQ
HOST=127.0.0.1
PORT=8000
RESULT_FILE="$LOG_DIR/TEST_RESULTS.md"

mkdir -p "$LOG_DIR"

echo "=== Wait for model download ===" | tee -a "$RESULT_FILE"
for i in $(seq 1 360); do
  if [[ -f "$MODEL_DIR/config.json" ]] && ls "$MODEL_DIR"/model-*.safetensors >/dev/null 2>&1; then
    COUNT=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
  if [[ "$COUNT" -ge 8 ]]; then
      echo "Model files ready: $COUNT shards" | tee -a "$RESULT_FILE"
      break
    fi
  fi
  sleep 10
done

du -sh "$MODEL_DIR" | tee -a "$RESULT_FILE"

echo "=== Upgrade transformers for qwen3_5 ===" | tee -a "$RESULT_FILE"
pip install -q "git+https://github.com/huggingface/transformers.git" 2>&1 | tail -5 | tee -a "$LOG_DIR/pip-transformers.log"
python -c "import transformers; print('transformers', transformers.__version__)" | tee -a "$RESULT_FILE"

echo "=== Start vLLM ===" | tee -a "$RESULT_FILE"
pkill -f "vllm serve" 2>/dev/null || true
sleep 3

nohup vllm serve "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype auto \
  --trust-remote-code \
  > "$LOG_DIR/vllm-server.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=$VLLM_PID" | tee -a "$RESULT_FILE"

READY=0
for i in $(seq 1 240); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited" | tee -a "$RESULT_FILE"
    tail -50 "$LOG_DIR/vllm-server.log" | tee -a "$RESULT_FILE"
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  tail -50 "$LOG_DIR/vllm-server.log" | tee -a "$RESULT_FILE"
  exit 1
fi

echo "=== Inference ===" | tee -a "$RESULT_FILE"
RESP=$(curl -sf "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好，我是","max_tokens":64,"temperature":0.7}')
echo "$RESP" | tee -a "$RESULT_FILE"

if echo "$RESP" | grep -q '"choices"'; then
  echo "PASS B03: Qwen3.6-27B-AWQ inference on MetaX C500 (32GB)" | tee -a "$RESULT_FILE"
else
  echo "FAIL B03" | tee -a "$RESULT_FILE"
  exit 1
fi

mx-smi 2>/dev/null | head -25 | tee -a "$RESULT_FILE" || true
kill "$VLLM_PID" 2>/dev/null || true
