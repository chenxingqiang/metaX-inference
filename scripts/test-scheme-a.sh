#!/usr/bin/env bash
# Scheme A smoke test: Unsloth GGUF + llama.cpp (Vulkan) — AGENT.md §4
set -euo pipefail

GGUF_MODEL="${GGUF_MODEL:-}"
LLAMA_SERVER="${LLAMA_SERVER:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
PROMPT="${PROMPT:-你好，我是}"
N_PREDICT="${N_PREDICT:-128}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "$GGUF_MODEL" || ! -f "$GGUF_MODEL" ]]; then
  echo "ERROR: set GGUF_MODEL to a valid .gguf file path"
  exit 1
fi

if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in \
    "./llama.cpp/build/bin/llama-server" \
    "./llama.cpp/build/llama-server" \
    "llama-server"; do
    if [[ -x "$candidate" ]] || command -v "$candidate" &>/dev/null; then
      LLAMA_SERVER="$candidate"
      break
    fi
  done
fi

if [[ -z "$LLAMA_SERVER" ]] || ! command -v "$LLAMA_SERVER" &>/dev/null && [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: set LLAMA_SERVER or build llama.cpp with -DGGML_VULKAN=ON"
  exit 1
fi

echo "=== Scheme A: llama-server + GGUF ==="
echo "Model: $GGUF_MODEL"
echo "Server: $LLAMA_SERVER"

"$LLAMA_SERVER" \
  -m "$GGUF_MODEL" \
  -ngl 99 \
  -c 8192 \
  --spec-type mtp --spec-draft-n-max 3 \
  --host "$HOST" --port "$PORT" &
SERVER_PID=$!

echo "Waiting for server on $HOST:$PORT ..."
for i in $(seq 1 60); do
  if curl -sf "http://${HOST}:${PORT}/health" &>/dev/null || \
     curl -sf "http://${HOST}:${PORT}/" &>/dev/null; then
    break
  fi
  sleep 2
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: llama-server exited early"
    exit 1
  fi
done

RESP=$(curl -sf "http://${HOST}:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"${PROMPT}\",\"n_predict\":${N_PREDICT}}" || true)

if [[ -z "$RESP" ]]; then
  echo "FAIL A03: empty response from /completion"
  exit 1
fi

echo "Response snippet:"
echo "$RESP" | head -c 500
echo ""

if echo "$RESP" | grep -qE 'content|text|generation'; then
  echo "PASS A03: completion endpoint returned expected fields"
else
  echo "FAIL A03: unexpected response format"
  exit 1
fi

echo "Scheme A smoke test completed."
