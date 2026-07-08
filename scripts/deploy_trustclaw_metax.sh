#!/usr/bin/env bash
# Deploy TrustClaw Gateway on MetaX, backed by local vLLM (Qwen3.6-27B-AWQ).
#
# Prerequisites:
#   - vLLM running (./scripts/deploy_unsloth_studio.sh)
#
# Usage:
#   ./scripts/deploy_trustclaw_metax.sh
#   TRUSTCLAW_PORT=8080 ./scripts/deploy_trustclaw_metax.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRUSTCLAW_DIR="${TRUSTCLAW_DIR:-/data/TrustClaw}"
TRUSTCLAW_REPO="${TRUSTCLAW_REPO:-https://github.com/chenxingqiang/TrustClaw.git}"
NODE_VERSION="${NODE_VERSION:-22.19.0}"
NODE_PREFIX="${NODE_PREFIX:-/opt/node-v${NODE_VERSION}-linux-x64}"
STATE_DIR="${TRUSTCLAW_STATE_DIR:-/data/trustclaw-state}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/trustclaw}"
PID_FILE="${PID_FILE:-$LOG_DIR/gateway.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/gateway.log}"
GATEWAY_PORT="${TRUSTCLAW_PORT:-19001}"
VLLM_ENV="${VLLM_ENV:-/data/metax-test-logs/unsloth-studio/unsloth-studio.env}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

# shellcheck disable=SC1090
[[ -f "$VLLM_ENV" ]] && source "$VLLM_ENV"
VLLM_BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL_ID="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
VLLM_API_KEY="${VLLM_API_KEY:?Set VLLM_API_KEY or run deploy_unsloth_studio.sh first}"

if ! curl -sf "$VLLM_BASE_URL/models" -H "Authorization: Bearer $VLLM_API_KEY" >/dev/null; then
  echo "ERROR: vLLM not reachable at $VLLM_BASE_URL — run deploy_unsloth_studio.sh first"
  exit 1
fi

install_node() {
  if [[ -x "$NODE_PREFIX/bin/node" ]]; then
    export PATH="$NODE_PREFIX/bin:$PATH"
    return 0
  fi
  echo "Installing Node.js $NODE_VERSION ..."
  tmp="/tmp/node-v${NODE_VERSION}-linux-x64.tar.xz"
  curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o "$tmp"
  tar -xJf "$tmp" -C /opt/
  rm -f "$tmp"
  export PATH="$NODE_PREFIX/bin:$PATH"
  node --version
  corepack enable
  corepack prepare pnpm@11.2.2 --activate
}

install_node

if [[ ! -d "$TRUSTCLAW_DIR/.git" ]]; then
  echo "Cloning TrustClaw to $TRUSTCLAW_DIR ..."
  git clone --depth 1 "$TRUSTCLAW_REPO" "$TRUSTCLAW_DIR"
fi

cd "$TRUSTCLAW_DIR"

if [[ ! -f dist/index.js ]]; then
  echo "Building TrustClaw (first run, ~15–30 min) ..."
  pnpm install --config.minimumReleaseAge=0
  pnpm build
  pnpm trustclaw:ui:build
fi

GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
if [[ -z "$GATEWAY_TOKEN" ]]; then
  if [[ -f "$STATE_DIR/gateway.token" ]]; then
    GATEWAY_TOKEN="$(cat "$STATE_DIR/gateway.token")"
  else
    GATEWAY_TOKEN="tc-$(openssl rand -hex 24 2>/dev/null || head -c 48 /dev/urandom | xxd -p -c 96)"
    echo "$GATEWAY_TOKEN" >"$STATE_DIR/gateway.token"
    chmod 600 "$STATE_DIR/gateway.token"
  fi
fi

CONFIG_PATH="$STATE_DIR/openclaw.json"
sed -e "s|REPLACE_GATEWAY_TOKEN|$GATEWAY_TOKEN|g" \
  -e "s|REPLACE_VLLM_MODEL_ID|$VLLM_MODEL_ID|g" \
  -e "s|REPLACE_VLLM_BASE_URL|$VLLM_BASE_URL|g" \
  -e "s|REPLACE_VLLM_API_KEY|$VLLM_API_KEY|g" \
  -e "s|REPLACE_STATE_DIR|$STATE_DIR|g" \
  -e "s|REPLACE_AGENT_PACKS_DIR|$TRUSTCLAW_DIR/trustclaw/agents|g" \
  "$REPO_ROOT/configs/trustclaw-metax-vllm.json" >"$CONFIG_PATH"

mkdir -p "$STATE_DIR/workspace" "$STATE_DIR/agents/main/agent"

export OPENCLAW_STATE_DIR="$STATE_DIR"
export OPENCLAW_CONFIG_PATH="$CONFIG_PATH"
export OPENCLAW_HOME="$STATE_DIR"
export VLLM_API_KEY
export OPENCLAW_DISABLE_BONJOUR=1

pnpm trustclaw:setup 2>/dev/null || node scripts/trustclaw-setup.mjs || true

# Re-apply vLLM config after setup (setup may overwrite model defaults)
sed -e "s|REPLACE_GATEWAY_TOKEN|$GATEWAY_TOKEN|g" \
  -e "s|REPLACE_VLLM_MODEL_ID|$VLLM_MODEL_ID|g" \
  -e "s|REPLACE_VLLM_BASE_URL|$VLLM_BASE_URL|g" \
  -e "s|REPLACE_VLLM_API_KEY|$VLLM_API_KEY|g" \
  -e "s|REPLACE_STATE_DIR|$STATE_DIR|g" \
  -e "s|REPLACE_AGENT_PACKS_DIR|$TRUSTCLAW_DIR/trustclaw/agents|g" \
  "$REPO_ROOT/configs/trustclaw-metax-vllm.json" >"$CONFIG_PATH"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping existing TrustClaw gateway (pid=$old_pid)..."
    kill "$old_pid" 2>/dev/null || true
    sleep 3
  fi
fi
pkill -f "openclaw.mjs gateway.*$GATEWAY_PORT" 2>/dev/null || true
sleep 2

echo "Starting TrustClaw Gateway on 0.0.0.0:$GATEWAY_PORT ..."
nohup node openclaw.mjs gateway --bind lan --port "$GATEWAY_PORT" \
  >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

READY=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 3
done

if [[ "$READY" -ne 1 ]]; then
  echo "ERROR: TrustClaw gateway not ready"
  tail -40 "$LOG_FILE" || true
  exit 1
fi

cat >"$LOG_DIR/TRUSTCLAW_DEPLOY.md" <<EOF
# TrustClaw Deployment (MetaX + vLLM)

- Deployed: $(date -Iseconds)
- Gateway: http://<server-ip>:$GATEWAY_PORT/
- Health: http://127.0.0.1:$GATEWAY_PORT/healthz
- TRA Console: http://<server-ip>:$GATEWAY_PORT/trustclaw/
- Gateway token: $GATEWAY_TOKEN
- LLM backend: $VLLM_BASE_URL
- Model: vllm/$VLLM_MODEL_ID

## Open Control UI

Use tokenized URL (replace host):
\`http://127.0.0.1:$GATEWAY_PORT/?token=$GATEWAY_TOKEN\`

## vLLM connection

TrustClaw uses bundled vLLM provider; key in $CONFIG_PATH env.VLLM_API_KEY
EOF

echo ""
echo "=== TrustClaw Deployment OK ==="
echo "URL:    http://127.0.0.1:$GATEWAY_PORT/?token=$GATEWAY_TOKEN"
echo "Model:  vllm/$VLLM_MODEL_ID"
echo "Token:  $GATEWAY_TOKEN"
echo "Docs:   $LOG_DIR/TRUSTCLAW_DEPLOY.md"
