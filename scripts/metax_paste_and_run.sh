#!/usr/bin/env bash
# =============================================================================
# MetaX 实机一键脚本 — SSH 登录后直接粘贴运行（或 curl | bash）
#
#   curl -fsSL "https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/metax_paste_and_run.sh" | bash
#
# 前置：conda base + vLLM + 模型 /data/models/Qwen3.6-27B-AWQ
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs}"
BRANCH="${BRANCH:-main}"
GITHUB_RAW="${GITHUB_RAW:-https://raw.githubusercontent.com/chenxingqiang/metaX-inference}"

if [[ -f "${REPO_DIR}/scripts/metax_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/scripts/metax_env.sh"
else
  curl -fsSL "$GITHUB_RAW/main/scripts/metax_env.sh" -o /tmp/metax_env.sh
  # shellcheck source=/dev/null
  source /tmp/metax_env.sh
fi

echo "=== metaX-inference 实机全套基准 ==="
echo "时间: $(date -Iseconds)"
echo "GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"

mkdir -p "$REPO_DIR" "$LOG_ROOT"

if [[ ! -f "$REPO_DIR/scripts/remote_run_all_benches.sh" ]]; then
  echo "Syncing repo to $REPO_DIR ..."
  if curl -fsSL "$GITHUB_RAW/$BRANCH/scripts/sync_from_github.sh" -o /tmp/sync_from_github.sh; then
    bash /tmp/sync_from_github.sh
  else
    echo "Fallback: GitHub archive..."
    TMPDIR="/tmp/metaX-inference-dl"
    rm -rf "$TMPDIR"
    mkdir -p "$TMPDIR" "$REPO_DIR"
    curl -fsSL "https://github.com/chenxingqiang/metaX-inference/archive/refs/heads/${BRANCH}.tar.gz" \
      | tar -xz -C "$TMPDIR" --strip-components=1
    if command -v rsync &>/dev/null; then
      rsync -a "$TMPDIR/" "$REPO_DIR/"
    else
      find "$REPO_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
      cp -a "$TMPDIR/." "$REPO_DIR/"
    fi
  fi
  # shellcheck source=/dev/null
  source "${REPO_DIR}/scripts/metax_env.sh"
fi

cd "$REPO_DIR"

if [[ "${FAST:-0}" == "1" ]]; then
  echo "FAST=1 — running quick smoke only"
  bash scripts/quick_smoke_metax.sh
  exit 0
fi

bash scripts/validate_repo.sh
bash scripts/remote_run_all_benches.sh

"$PYTHON" scripts/bench_acceptance.py "$LOG_ROOT" --json -o "$LOG_ROOT/ACCEPTANCE.json"
"$PYTHON" scripts/bench_acceptance.py "$LOG_ROOT" | tee -a "$LOG_ROOT/ACCEPTANCE.txt"
"$PYTHON" scripts/bench_acceptance.py "$LOG_ROOT" --markdown -o "$LOG_ROOT/ACCEPTANCE.md"

if [[ -f "$LOG_ROOT/ACCEPTANCE.json" ]]; then
  "$PYTHON" scripts/update_acceptance_baseline.py "$LOG_ROOT/ACCEPTANCE.json" || true
fi

echo ""
echo "=== 完成 ==="
echo "汇总: $LOG_ROOT/ALL_BENCH_SUMMARY.md"
echo "验收: $LOG_ROOT/ACCEPTANCE.json"
echo "打包: /tmp/metax-bench-bundle.tar.gz"
echo ""
echo "下载到本地:"
echo "  scp -P 32222 root+vm-Fcar5bmuV0AjJMTG@140.207.205.81:/tmp/metax-bench-bundle.tar.gz ."
