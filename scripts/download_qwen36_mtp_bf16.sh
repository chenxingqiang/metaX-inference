#!/usr/bin/env bash
# Download Qwen3.6-27B AWQ + BF16 MTP head checkpoint (~26GB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

MODEL_ID="${MODEL_ID:-hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16}"
LOCAL_DIR="${LOCAL_DIR:-/data/models/Qwen3.6-27B-MTP-BF16}"
LOG="${LOG:-/data/metax-test-logs/download-mtp-bf16.log}"

mkdir -p "$(dirname "$LOG")" "$LOCAL_DIR"

{
  echo "=== download_qwen36_mtp_bf16 ==="
  echo "  model: $MODEL_ID"
  echo "  local: $LOCAL_DIR"
  echo "  started: $(date -Iseconds)"
  echo ""
} | tee "$LOG"

if command -v hf &>/dev/null; then
  HF_CLI=hf
elif command -v huggingface-cli &>/dev/null; then
  HF_CLI=huggingface-cli
else
  echo "ERROR: hf / huggingface-cli not found" | tee -a "$LOG"
  exit 1
fi

if [[ "$HF_CLI" == "hf" ]]; then
  "$HF_CLI" download "$MODEL_ID" --local-dir "$LOCAL_DIR" \
    2>&1 | tee -a "$LOG"
else
  "$HF_CLI" download "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False \
    2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== MTP head check ===" | tee -a "$LOG"
cd "${REPO_DIR:-/data/metaX-inference}"
PYTHONPATH=. "$PYTHON" scripts/check_mtp_head.py "$LOCAL_DIR" --json 2>&1 | tee -a "$LOG"
echo "done $(date -Iseconds)" | tee -a "$LOG"
