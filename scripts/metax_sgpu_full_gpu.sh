#!/usr/bin/env bash
# MetaX sGPU: expand to 100% compute / full VRAM (HOST only)
#
# Current VM/container has read-only sysfs — mx-smi sgpu --set/--disable fails with:
#   "Sysfs error: Read-only file system"
# Run this script on the **physical host** or ask the cloud provider to resize the instance.
#
# Inside guest (diagnostic only):
#   mx-smi
#   mx-smi sgpu
#   mx-smi sgpu --show-remain
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=metax_env.sh
source "$SCRIPT_DIR/metax_env.sh" 2>/dev/null || true

SGPU_ID="${SGPU_ID:-2}"
TARGET_COMPUTE="${TARGET_COMPUTE:-100}"
TARGET_VRAM="${TARGET_VRAM:-64G}"
MODE="${MODE:-disable}"  # disable | set

echo "=== MetaX sGPU full-GPU upgrade ==="
echo "  mode: $MODE"
echo "  sgpu_id: $SGPU_ID"
echo "  target compute: ${TARGET_COMPUTE}%"
echo "  target vram: $TARGET_VRAM"
echo ""

if mount | grep -q 'sysfs on /sys type sysfs (ro,'; then
  echo "WARNING: /sys is read-only in this environment (Docker/VM guest)."
  echo "  mx-smi sgpu changes must be done on the HOST or via cloud console."
  echo ""
fi

echo "--- current ---"
mx-smi 2>/dev/null | sed -n '/Sliced GPU/,/Process/p' || true
mx-smi sgpu --show-mode 2>/dev/null || true
mx-smi sgpu --show-remain 2>/dev/null || true
echo ""

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "CHECK_ONLY=1 — no changes applied."
  exit 0
fi

# Stop workloads using GPU
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  echo "Stopping vLLM ..."
  pkill -f "vllm serve" || true
  sleep 5
fi

case "$MODE" in
  disable)
    echo "Disabling sGPU mode (full physical GPU to single tenant) ..."
    mx-smi sgpu --disable
    ;;
  set)
    echo "Expanding sGPU ${SGPU_ID} quota ..."
    mx-smi sgpu --set "$SGPU_ID" --compute "$TARGET_COMPUTE" --vram "$TARGET_VRAM"
    ;;
  *)
    echo "Unknown MODE=$MODE (use disable or set)" >&2
    exit 1
    ;;
esac

echo ""
echo "--- after ---"
mx-smi sgpu --show-mode 2>/dev/null || true
mx-smi 2>/dev/null | sed -n '/Sliced GPU/,/Process/p' || mx-smi 2>/dev/null | head -30

echo ""
echo "Next: re-run benchmark"
echo "  bash scripts/run_phase1_concurrent_bench.sh"
echo "  bash scripts/run_production_qa_eval.sh"
