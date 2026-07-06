#!/usr/bin/env bash
# Scheme B smoke test: Unsloth 4-bit + MacaRT-vLLM — AGENT.md §5
set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-./qwen3.6-27b-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-你好，我是}"
MAX_TOKENS="${MAX_TOKENS:-128}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -d "$VLLM_MODEL" ]]; then
  echo "ERROR: VLLM_MODEL directory not found: $VLLM_MODEL"
  echo "Run: python scripts/quantize-qwen36.py --output $VLLM_MODEL"
  exit 1
fi

echo "=== Scheme B: MacaRT-vLLM ==="
echo "Model: $VLLM_MODEL"

python -m vllm.entrypoints.api_server \
  --model "$VLLM_MODEL" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype auto \
  --trust-remote-code \
  --host "$HOST" --port "$PORT" &
SERVER_PID=$!

echo "Waiting for vLLM on $HOST:$PORT ..."
for i in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/health" &>/dev/null || \
     curl -sf "http://${HOST}:${PORT}/v1/models" &>/dev/null; then
    break
  fi
  sleep 3
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: vLLM server exited early"
    exit 1
  fi
done

MODEL_NAME=$(basename "$VLLM_MODEL")
RESP=$(curl -sf "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS}}" || true)

if [[ -z "$RESP" ]]; then
  echo "FAIL B03: empty response from /v1/completions"
  exit 1
fi

echo "Response snippet:"
echo "$RESP" | head -c 500
echo ""

if echo "$RESP" | grep -q 'choices'; then
  echo "PASS B03: OpenAI-compatible completions OK"
else
  echo "FAIL B03: unexpected response format"
  exit 1
fi

echo "Scheme B smoke test completed."
