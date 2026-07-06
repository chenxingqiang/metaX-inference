#!/usr/bin/env bash
# Environment pre-check for MetaX + Unsloth + Qwen3.6 inference (AGENT.md §3)
set -euo pipefail

PASS=0
FAIL=0
WARN=0

check() {
  local id="$1" desc="$2" result="$3" note="${4:-}"
  if [[ "$result" == "PASS" ]]; then
    echo "[PASS] $id $desc${note:+ — $note}"
    ((PASS++)) || true
  elif [[ "$result" == "WARN" ]]; then
    echo "[WARN] $id $desc${note:+ — $note}"
    ((WARN++)) || true
  else
    echo "[FAIL] $id $desc${note:+ — $note}"
    ((FAIL++)) || true
  fi
}

echo "=== MetaX Unsloth Qwen3.6 Environment Check ==="

# E01: MXMACA packages
if dpkg -l 2>/dev/null | grep -q metax-maca-driver; then
  check E01 "MXMACA driver installed" PASS
else
  check E01 "MXMACA driver installed" FAIL "run: sudo apt install metax-maca-driver metax-maca-runtime metax-maca-dev"
fi

# E02: Vulkan
if command -v vulkaninfo &>/dev/null; then
  if vulkaninfo --summary 2>/dev/null | grep -qiE 'device|gpu|metax|muxi'; then
    check E02 "Vulkan GPU device visible" PASS
  else
    check E02 "Vulkan GPU device visible" WARN "vulkaninfo present but no obvious GPU device"
  fi
else
  check E02 "Vulkan SDK (vulkaninfo)" FAIL "install vulkan-sdk"
fi

# E03: Python version
if command -v python &>/dev/null; then
  PY_VER=$(python --version 2>&1)
  if [[ "$PY_VER" == *"3.10"* ]]; then
    check E03 "Python 3.10" PASS "$PY_VER"
  else
    check E03 "Python 3.10" WARN "$PY_VER (3.10 recommended)"
  fi
elif command -v python3 &>/dev/null; then
  PY_VER=$(python3 --version 2>&1)
  check E03 "Python 3.10" WARN "$PY_VER"
else
  check E03 "Python available" FAIL
fi

# E04: Unsloth import
PY_BIN="${PYTHON:-python}"
if $PY_BIN -c "import unsloth" 2>/dev/null; then
  check E04 "Unsloth importable" PASS
else
  check E04 "Unsloth importable" FAIL "pip install unsloth"
fi

# E05: GPU memory hint
if command -v mx-smi &>/dev/null; then
  check E05 "GPU info (mx-smi)" PASS "$(mx-smi 2>/dev/null | head -5 | tr '\n' ' ' || true)"
elif [[ -d /dev/dri ]]; then
  check E05 "DRI devices present" WARN "mx-smi not found; check VRAM manually"
else
  check E05 "GPU / VRAM" WARN "no MetaX GPU detected in this environment"
fi

echo ""
echo "Summary: PASS=$PASS WARN=$WARN FAIL=$FAIL"
if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
