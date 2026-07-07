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

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== metaX-inference 实机全套基准 ==="
echo "时间: $(date -Iseconds)"
echo "GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"

mkdir -p "$REPO_DIR" "$LOG_ROOT"

if [[ ! -f "$REPO_DIR/scripts/remote_run_all_benches.sh" ]]; then
  echo "Syncing repo to $REPO_DIR ..."
  TMP="/tmp/metaX-inference-sync.tar.gz"
  curl -fsSL "$GITHUB_RAW/$BRANCH/scripts/remote_sync_repo.sh" -o /tmp/sync.sh || true
  if [[ -f /tmp/sync.sh ]]; then
    bash /tmp/sync.sh 2>/dev/null || true
  fi
  if [[ ! -f "$REPO_DIR/scripts/remote_run_all_benches.sh" ]]; then
    echo "Fallback: curl tarball from GitHub archive..."
    curl -fsSL "https://github.com/chenxingqiang/metaX-inference/archive/refs/heads/${BRANCH}.tar.gz" \
      | tar -xz -C /tmp
    cp -r "/tmp/metaX-inference-${BRANCH//\//-}"/* "$REPO_DIR/" 2>/dev/null || \
      cp -r /tmp/metaX-inference-*/* "$REPO_DIR/" 2>/dev/null || true
  fi
fi

cd "$REPO_DIR"
bash scripts/validate_repo.sh
bash scripts/remote_run_all_benches.sh

python scripts/bench_acceptance.py "$LOG_ROOT" --json -o "$LOG_ROOT/ACCEPTANCE.json"
python scripts/bench_acceptance.py "$LOG_ROOT" | tee -a "$LOG_ROOT/ACCEPTANCE.txt"
python scripts/bench_acceptance.py "$LOG_ROOT" --markdown -o "$LOG_ROOT/ACCEPTANCE.md"

echo ""
echo "=== 完成 ==="
echo "汇总: $LOG_ROOT/ALL_BENCH_SUMMARY.md"
echo "验收: $LOG_ROOT/ACCEPTANCE.json"
echo "打包: /tmp/metax-bench-bundle.tar.gz"
echo ""
echo "下载到本地:"
echo "  scp -P 32222 root+vm-Fcar5bmuV0AjJMTG@140.207.205.81:/tmp/metax-bench-bundle.tar.gz ."
