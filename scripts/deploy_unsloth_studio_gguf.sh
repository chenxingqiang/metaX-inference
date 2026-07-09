#!/usr/bin/env bash
# Optional: native Unsloth Studio API via GGUF (llama-server backend).
# Slower than vLLM+AWQ on MetaX, but exposes sk-unsloth-* API directly.
#
# Requires: curl -fsSL https://unsloth.ai/install.sh | sh
#
# Usage:
#   ./scripts/deploy_unsloth_studio_gguf.sh
#   GGUF_FILE=/data/scheme-a/models/Qwen3.6-27B-Q4_K_M.gguf PORT=8888 ./scripts/deploy_unsloth_studio_gguf.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/huggingface_home}"

GGUF_FILE="${GGUF_FILE:-/data/scheme-a/models/Qwen3.6-27B-Q4_K_M.gguf}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.6-27B-MTP-GGUF}"
GGUF_VARIANT="${GGUF_VARIANT:-Q4_K_M}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
LOG_DIR="${LOG_DIR:-/data/metax-test-logs/unsloth-studio-gguf}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/unsloth-run.log}"

mkdir -p "$LOG_DIR"

if ! command -v unsloth &>/dev/null; then
  echo "Installing Unsloth Studio (non-interactive)..."
  curl -fsSL https://unsloth.ai/install.sh | UNSLOTH_NONINTERACTIVE=1 sh || {
    echo "Install failed. On MetaX, prefer vLLM mode: ./scripts/deploy_unsloth_studio.sh"
    exit 1
  }
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -f "$GGUF_FILE" ]]; then
  echo "Downloading GGUF..."
  mkdir -p "$(dirname "$GGUF_FILE")"
  hf download "$GGUF_REPO" "Qwen3.6-27B-${GGUF_VARIANT}.gguf" --local-dir "$(dirname "$GGUF_FILE")" || exit 1
  GGUF_FILE="$(dirname "$GGUF_FILE")/Qwen3.6-27B-${GGUF_VARIANT}.gguf"
fi

pkill -f "unsloth run" 2>/dev/null || true
pkill -f "unsloth studio" 2>/dev/null || true
sleep 2

echo "Starting Unsloth native API on $HOST:$PORT with $GGUF_FILE"
nohup unsloth run \
  --model "$GGUF_FILE" \
  -H "$HOST" \
  -p "$PORT" \
  --disable-tools \
  -c 8192 \
  --reasoning off \
  >"$LOG_FILE" 2>&1 &

echo $! >"$LOG_DIR/unsloth.pid"
echo "Log: $LOG_FILE — API key printed in log when ready."
echo "Connect: http://<host>:$PORT/v1 with Bearer sk-unsloth-... from log."
