#!/usr/bin/env bash
# Download BF16 MTP source and graft onto MetaX-compatible AWQ base.
#
# hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16 cannot load directly on MetaX
# (compressed-tensors WNA16 uint4 vs Exllama uint4b8). We download it for mtp.*
# tensors only, then graft onto /data/models/Qwen3.6-27B-AWQ.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh"

MODEL_ID="${MODEL_ID:-hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16}"
MTP_SOURCE="${MTP_SOURCE:-/data/models/Qwen3.6-27B-MTP-BF16}"
AWQ_BASE="${AWQ_BASE:-/data/models/Qwen3.6-27B-AWQ}"
LOCAL_DIR="${LOCAL_DIR:-/data/models/Qwen3.6-27B-AWQ-MTP-BF16}"
LOG="${LOG:-/data/metax-test-logs/download-mtp-bf16.log}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE_GRAFT="${FORCE_GRAFT:-0}"

mkdir -p "$(dirname "$LOG")" "$LOCAL_DIR"

{
  echo "=== download_qwen36_mtp_bf16 ==="
  echo "  model: $MODEL_ID"
  echo "  mtp_source: $MTP_SOURCE"
  echo "  awq_base: $AWQ_BASE"
  echo "  graft_out: $LOCAL_DIR"
  echo "  started: $(date -Iseconds)"
  echo ""
} | tee "$LOG"

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  mkdir -p "$(dirname "$LOG")" "$MTP_SOURCE"

  if command -v hf &>/dev/null; then
    HF_CLI=hf
  elif command -v huggingface-cli &>/dev/null; then
    HF_CLI=huggingface-cli
  else
    echo "ERROR: hf / huggingface-cli not found" | tee -a "$LOG"
    exit 1
  fi

  if [[ "$HF_CLI" == "hf" ]]; then
    "$HF_CLI" download "$MODEL_ID" --local-dir "$MTP_SOURCE" \
      2>&1 | tee -a "$LOG"
  else
    "$HF_CLI" download "$MODEL_ID" \
      --local-dir "$MTP_SOURCE" \
      --local-dir-use-symlinks False \
      2>&1 | tee -a "$LOG"
  fi
else
  echo "SKIP_DOWNLOAD=1 — using existing $MTP_SOURCE" | tee -a "$LOG"
fi

if [[ ! -f "$AWQ_BASE/config.json" ]]; then
  echo "ERROR: AWQ base missing at $AWQ_BASE" | tee -a "$LOG"
  exit 1
fi
if [[ ! -f "$MTP_SOURCE/model.safetensors.index.json" ]]; then
  echo "ERROR: MTP source missing at $MTP_SOURCE" | tee -a "$LOG"
  exit 1
fi

echo "" | tee -a "$LOG"
echo "=== Graft BF16 MTP onto AWQ base ===" | tee -a "$LOG"
cd "${REPO_DIR:-/data/metaX-inference}"
GRAFT_ARGS=(--base "$AWQ_BASE" --source "$MTP_SOURCE" --out "$LOCAL_DIR" --json)
[[ "$FORCE_GRAFT" == "1" ]] && GRAFT_ARGS+=(--force)
PYTHONPATH=. "$PYTHON" scripts/graft_mtp_bf16_to_awq.py "${GRAFT_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== MTP head check (grafted) ===" | tee -a "$LOG"
PYTHONPATH=. "$PYTHON" scripts/check_mtp_head.py "$LOCAL_DIR" --json 2>&1 | tee -a "$LOG"
echo "done $(date -Iseconds)" | tee -a "$LOG"
