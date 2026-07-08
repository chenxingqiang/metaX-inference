#!/usr/bin/env bash
# Deploy Qwen3.6-27B-AWQ for Unsloth Studio (vLLM connection mode).
#
# Unsloth Studio → Settings → Connections → vLLM:
#   Base URL: http://<host>:8000/v1
#   API key:  value of VLLM_API_KEY (default generated below)
#
# Also works with any OpenAI-compatible client (Cursor, Open WebUI, curl).
#
# Usage:
#   ./scripts/deploy_unsloth_studio.sh
#   VLLM_API_KEY=sk-my-key HOST=0.0.0.0 PORT=8000 ./scripts/deploy_unsloth_studio.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.92}"
MAX_SEQS="${MAX_SEQS:-64}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/unsloth-studio}"
PID_FILE="${PID_FILE:-$LOG_DIR/vllm.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm.log}"
ENV_FILE="${ENV_FILE:-$LOG_DIR/unsloth-studio.env}"

mkdir -p "$LOG_DIR"

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
fi
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  VLLM_API_KEY="sk-unsloth-metax-$(openssl rand -hex 16 2>/dev/null || head -c 32 /dev/urandom | xxd -p -c 64)"
fi

cat >"$ENV_FILE" <<EOF
# Unsloth Studio / OpenAI client connection
VLLM_API_KEY=$VLLM_API_KEY
HOST=$HOST
PORT=$PORT
MODEL=$MODEL
BASE_URL=http://127.0.0.1:$PORT/v1
EOF
chmod 600 "$ENV_FILE"

stop_existing() {
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Stopping existing vLLM (pid=$old_pid)..."
      kill "$old_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$old_pid" 2>/dev/null || break
        sleep 2
      done
    fi
  fi
  pkill -f "vllm serve.*$PORT" 2>/dev/null || true
  sleep 2
}

stop_existing

echo "Starting vLLM for Unsloth Studio compatibility..."
echo "  model=$MODEL"
echo "  listen=$HOST:$PORT"
echo "  api_key saved to $ENV_FILE"

nohup vllm serve "$MODEL" \
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
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
VLLM_PID="$(cat "$PID_FILE")"
echo "vLLM pid=$VLLM_PID log=$LOG_FILE"

READY=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" \
    -H "Authorization: Bearer $VLLM_API_KEY" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited early. Last 40 log lines:"
    tail -40 "$LOG_FILE" || true
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: vLLM not ready after 450s"
  tail -40 "$LOG_FILE" || true
  exit 1
fi

MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" \
  -H "Authorization: Bearer $VLLM_API_KEY" | python -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "$MODEL")"

{
  echo "# Unsloth Studio Deployment"
  echo ""
  echo "- Deployed: $(date -Iseconds)"
  echo "- Model: $MODEL"
  echo "- Model ID (for clients): $MODEL_ID"
  echo "- Base URL: http://<server-ip>:$PORT/v1"
  echo "- API key: $VLLM_API_KEY"
  echo ""
  echo "## Connect in Unsloth Studio"
  echo "1. Open Unsloth Studio → Settings → Connections → Add Connection"
  echo "2. Choose **vLLM**"
  echo "3. Base URL: \`http://<server-ip>:$PORT/v1\`"
  echo "4. API key: paste key from \`$ENV_FILE\`"
  echo "5. Load Models → select \`$MODEL_ID\`"
  echo ""
  echo "## curl smoke test"
  echo '```bash'
  echo "curl http://127.0.0.1:$PORT/v1/chat/completions \\"
  echo "  -H 'Authorization: Bearer $VLLM_API_KEY' \\"
  echo "  -H 'Content-Type: application/json' \\"
  echo "  -d '{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
  echo '```'
} >"$LOG_DIR/UNSLOTH_STUDIO_DEPLOY.md"

echo ""
echo "=== Deployment OK ==="
echo "Base URL:  http://127.0.0.1:$PORT/v1"
echo "Model ID:  $MODEL_ID"
echo "API key:   $VLLM_API_KEY"
echo "Docs:      $LOG_DIR/UNSLOTH_STUDIO_DEPLOY.md"
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK in one word.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python -m json.tool 2>/dev/null | head -20 || true
