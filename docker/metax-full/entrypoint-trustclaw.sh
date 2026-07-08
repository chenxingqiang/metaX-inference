#!/bin/sh
set -eu

export OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-/home/node/.openclaw}"
export OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$OPENCLAW_STATE_DIR/openclaw.json}"
export OPENCLAW_CONFIG_DIR="${OPENCLAW_CONFIG_DIR:-$OPENCLAW_STATE_DIR}"
export OPENCLAW_WORKSPACE_DIR="${OPENCLAW_WORKSPACE_DIR:-$OPENCLAW_STATE_DIR/workspace}"
export OPENCLAW_DISABLE_BONJOUR="${OPENCLAW_DISABLE_BONJOUR:-1}"
export OPENCLAW_GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-19001}"
export TRUSTCLAW_APP_ROOT="${TRUSTCLAW_APP_ROOT:-/app}"

# Wait for vLLM when using compose service name.
VLLM_WAIT_URL="${VLLM_WAIT_URL:-${VLLM_BASE_URL:-http://vllm:8000/v1}/models}"
if [ -n "${VLLM_API_KEY:-}" ]; then
  for i in $(seq 1 120); do
    if curl -sf "$VLLM_WAIT_URL" -H "Authorization: Bearer $VLLM_API_KEY" >/dev/null 2>&1; then
      echo "[metax-trustclaw] vLLM ready"
      break
    fi
    if [ "$i" -eq 120 ]; then
      echo "[metax-trustclaw] ERROR: vLLM not ready at $VLLM_WAIT_URL" >&2
      exit 1
    fi
    sleep 5
  done
fi

node /opt/metax-full/init-config.mjs

exec node /app/openclaw.mjs gateway \
  --bind lan \
  --port "$OPENCLAW_GATEWAY_PORT"
